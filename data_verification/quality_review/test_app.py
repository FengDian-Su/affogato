from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import statistics
import tempfile
import unittest

import httpx
from PIL import Image

from data_verification.quality_review.app import (
    connect,
    create_app,
    export_database,
    get_metadata,
    initialize_database,
    krippendorff_alpha,
    score,
    set_metadata,
    update_rubric,
    validate_rating,
)
from data_verification.quality_review.import_v1 import import_v1
from data_verification.quality_review.rubric import AXIS_TAGS

GOOD = {"task": 2, "role": 2, "heatmap": 2}
NO_TAGS = {"task": [], "role": [], "heatmap": []}
TWO, TWO_TAGS = {"task": 2, "role": 2}, {"task": [], "role": []}


def heat_image(path: Path, x: int) -> str:
    """A grey render with one orange patch at column x (enough for the region-IoU check)."""
    image = Image.new("RGB", (64, 32), (142, 142, 147))
    image.paste((245, 166, 35), (x, 8, x + 12, 20))
    image.save(path)
    return str(path)


def write_fixture(root: Path, count: int = 3) -> Path:
    rows = []
    for index in range(count):
        images = {kind: heat_image(root / f"{index}.{kind}.png", 4) for kind in ("rgb", "a", "b")}
        rows.append({
            "sample_id": f"object-{index}/q{index}_test_task",
            "object_id": f"object-{index}",
            "dataset": "daily_used",
            "split": "dev" if index < 2 else "test",
            "object_name": f"Object {index}",
            "task": f"open the lid {index}",
            "roles": [
                {"id": "A", "role": "rotate", "target": "lid", "contact_region": "the lid rim",
                 "function": "unscrew the lid"},
                {"id": "B", "role": "hold", "target": "body", "contact_region": "the jar body",
                 "function": "stabilize the jar"},
            ],
            "view_labels": [f"view {i:05d}" for i in range(8)],
            "rgb_path": images["rgb"],
            "heat_a_path": images["a"],
            "heat_b_path": images["b"],
        })
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest


class Client:
    """Synchronous wrapper around httpx's ASGI transport that keeps one user's cookies."""

    def __init__(self, app, username: str | None = None, address: str = "127.0.0.1"):
        self.app = app
        self.address = address
        self.cookies = httpx.Cookies()
        if username:
            response = self.post("/api/login", json={"username": username})
            assert response.status_code == 200, response.text

    def request(self, method: str, path: str, **kwargs):
        async def run():
            transport = httpx.ASGITransport(app=self.app, client=(self.address, 4321))
            async with httpx.AsyncClient(transport=transport, base_url="http://test",
                                         cookies=self.cookies) as client:
                response = await client.request(method, path, **kwargs)
                self.cookies.update(response.cookies)
                return response
        return asyncio.run(run())

    def get(self, path: str, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs):
        return self.request("POST", path, **kwargs)

    def next(self, skip: str | None = None) -> dict | None:
        response = self.post("/api/queue/next", params={"skip": skip} if skip else None)
        assert response.status_code == 200, response.text
        return response.json()["sample"]

    def rate(self, sample_id: str, labels=None, tags=None, comment="", key=None, endpoint="/api/annotations"):
        return self.post(endpoint, json={
            "sample_id": sample_id, "labels": labels or GOOD, "tags": tags or NO_TAGS,
            "comment": comment, "elapsed_ms": 4200,
            "idempotency_key": key or f"key-{id(self)}-{sample_id}-{endpoint}",
        })


class QualityReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.db = self.root / "review.sqlite3"
        initialize_database(self.db, write_fixture(self.root))
        self.app = create_app(self.db)
        self.ids = [f"object-{index}/q{index}_test_task" for index in range(3)]

    def tearDown(self):
        self.temporary.cleanup()

    # ------------------------------------------------------------------ pure functions

    def test_krippendorff_alpha_matches_published_example(self):
        # Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability": 4 observers, 12 units.
        observers = [
            [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
            [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
            [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
            [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
        ]
        units = [[o[u] for o in observers if o[u] is not None] for u in range(12)]
        categories = (1, 2, 3, 4, 5)
        self.assertAlmostEqual(krippendorff_alpha(units, "nominal", categories), 0.743, places=3)
        self.assertAlmostEqual(krippendorff_alpha(units, "ordinal", categories), 0.815, places=3)
        self.assertAlmostEqual(krippendorff_alpha(units, "interval", categories), 0.849, places=3)
        self.assertEqual(krippendorff_alpha([[2, 2], [1, 1], [0, 0]]), 1.0)
        self.assertIsNone(krippendorff_alpha([[2, 2], [2, 2]]))

    def test_score_is_mean_rating_over_two_times_100(self):
        self.assertAlmostEqual(score([2, 2, 1, 0]), 62.5)
        self.assertEqual(score([1, 1]), score([0, 2]), "all-OK and half-Bad/half-Good share a score")

    def test_rating_validation_rules(self):
        validate_rating(GOOD, NO_TAGS, "", AXIS_TAGS)
        cases = [
            ({"task": 2, "role": 2}, NO_TAGS, "", "cover exactly"),
            ({**GOOD, "role": 1}, NO_TAGS, "", "need at least one reason"),
            (GOOD, {**NO_TAGS, "task": ["implausible_task"]}, "", "Good cannot carry"),
            ({**GOOD, "task": 0}, {**NO_TAGS, "task": ["missing_or_unusable"]}, "", "unknown tags"),
            ({**GOOD, "heatmap": 1}, {**NO_TAGS, "heatmap": ["other"]}, "", "needs a note"),
            ({**GOOD, "task": 3}, NO_TAGS, "", "must be 0"),
        ]
        for labels, tags, comment, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_rating(labels, tags, comment, AXIS_TAGS)
        _, tags, comment = validate_rating({**GOOD, "heatmap": 1}, {**NO_TAGS, "heatmap": ["other", "other"]},
                                           "  odd patch ", AXIS_TAGS)
        self.assertEqual((tags["heatmap"], comment), (["other"], "odd patch"))

    def test_init_records_rubric_and_refuses_overwrite(self):
        connection = connect(self.db)
        try:
            rubric = get_metadata(connection, "rubric")
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0], 3)
        finally:
            connection.close()
        self.assertEqual([axis["id"] for axis in rubric["axes"]], ["task", "role", "heatmap"])
        with self.assertRaises(FileExistsError):
            initialize_database(self.db, self.root / "manifest.jsonl")
        self.assertTrue(initialize_database(self.db, self.root / "manifest.jsonl", dry_run=True)["dry_run"])

    def test_update_rubric_only_adds_reason_tags(self):
        connection = connect(self.db)
        rubric = get_metadata(connection, "rubric")
        rubric["axes"][0]["tags"] = [t for t in rubric["axes"][0]["tags"] if t["id"] != "not_meaningful"]
        rubric["version"] = "older"
        set_metadata(connection, "rubric", rubric)  # pretend the database predates the new tag
        connection.close()
        alice = Client(create_app(self.db), "alice")
        old_style = alice.rate(self.ids[0], {**GOOD, "task": 1}, {**NO_TAGS, "task": ["not_meaningful"]},
                               key="before-update")
        self.assertEqual(old_style.status_code, 422, "the frozen rubric does not know the tag yet")
        self.assertEqual(update_rubric(self.db)["added_tags"], {"task": ["not_meaningful"]})
        alice = Client(create_app(self.db), "alice")
        ok = alice.rate(self.ids[0], {**GOOD, "task": 1}, {**NO_TAGS, "task": ["not_meaningful"]}, key="after-update")
        self.assertEqual(ok.status_code, 200, ok.text)
        with self.assertRaises(SystemExit):
            update_rubric(self.db)  # nothing left to add
        connection = connect(self.db)
        rubric = get_metadata(connection, "rubric")
        rubric["axes"][2]["verdicts"]["2"] = "reworded"
        set_metadata(connection, "rubric", rubric)
        connection.close()
        with self.assertRaises(SystemExit, msg="wording changes are refused"):
            update_rubric(self.db)

    # ------------------------------------------------------------------ workflow

    def test_every_rater_gets_every_pair(self):
        alice, bob, carol = (Client(self.app, name) for name in ("alice", "bob", "carol"))
        first = alice.next()
        self.assertEqual(set(first), {"sample_id", "object_name", "task", "roles", "view_labels", "images",
                                      "axes", "carried"})
        self.assertEqual((first["axes"], first["carried"]), (["task", "role", "heatmap"], {}))
        self.assertEqual(alice.next()["sample_id"], first["sample_id"], "a held pair is shown again")
        self.assertEqual({bob.next()["sample_id"], carol.next()["sample_id"]}, {first["sample_id"]},
                         "nobody is blocked by someone else's hold")
        seen = []
        while (sample := alice.next()) is not None:
            seen.append(sample["sample_id"])
            self.assertEqual(alice.rate(sample["sample_id"]).status_code, 200)
        self.assertEqual(sorted(seen), sorted(self.ids), "each rater goes through every pair once")
        self.assertEqual(alice.get("/api/me").json()["progress"], {"samples": 3, "mine": 3})
        skipped = bob.next()["sample_id"]
        self.assertNotEqual(bob.next(skip=skipped)["sample_id"], skipped)

    def test_scores_average_the_raters(self):
        alice, bob = Client(self.app, "alice"), Client(self.app, "bob")
        for sid in self.ids:
            alice.rate(sid)
            labels = {**GOOD, "task": 0} if sid == self.ids[0] else GOOD
            tags = {**NO_TAGS, "task": ["implausible_task"]} if sid == self.ids[0] else NO_TAGS
            self.assertEqual(bob.rate(sid, labels, tags).status_code, 200)
        task = alice.get("/api/dashboard").json()["results"]["axes"]["task"]
        self.assertEqual({user: round(entry["score"], 2) for user, entry in task["per_rater"].items()},
                         {"alice": 100.0, "bob": 66.67})
        self.assertAlmostEqual(task["score"], (100 + 200 / 3) / 2)
        self.assertAlmostEqual(task["score_sd"], statistics.stdev([100, 200 / 3]))
        self.assertEqual((task["raters"], task["ratings"], task["distribution"]), (2, 6, [1, 0, 5]))

    def test_revisions_any_time_and_one_blind_rating_each(self):
        alice = Client(self.app, "alice")
        self.assertEqual(alice.rate(self.ids[0], {**GOOD, "task": 0}, {**NO_TAGS, "task": ["part_absent"]}).status_code, 200)
        self.assertEqual(alice.rate(self.ids[0], key="alice-twice").status_code, 409, "one blind rating per rater")
        fixed = alice.rate(self.ids[0], key="alice-fixes", endpoint="/api/revisions")
        self.assertEqual(fixed.status_code, 200, fixed.text)
        self.assertEqual(alice.get("/api/mine", params={"sample_id": self.ids[0]}).json()["rating"]["labels"]["task"], 2)
        history = alice.get("/api/history").json()
        self.assertEqual((history["total"], history["items"][0]["kind"]), (1, "revision"))
        self.assertEqual(Client(self.app, "carol").rate(self.ids[0], key="carol-edit-key",
                                                        endpoint="/api/revisions").status_code, 403)

    def test_idempotent_submit(self):
        alice = Client(self.app, "alice")
        sid = alice.next()["sample_id"]
        first = alice.rate(sid, key="same-key-123").json()
        second = alice.rate(sid, key="same-key-123").json()
        self.assertEqual((first["annotation_id"], second["duplicate"]), (second["annotation_id"], True))

    def test_export_round_trip(self):
        for name in ("alice", "bob"):
            client = Client(self.app, name)
            for sid in self.ids:
                client.rate(sid)
        summary = export_database(self.db, self.root / "export")
        pairs = [json.loads(line) for line in (self.root / "export/per_pair.jsonl").read_text().splitlines()]
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0]["task"], {"score": 100.0, "labels": {"alice": 2, "bob": 2}})
        objects = [json.loads(line) for line in (self.root / "export/per_object.jsonl").read_text().splitlines()]
        self.assertEqual({(row["task"], row["role"], row["heatmap"]) for row in objects}, {(100.0, 100.0, 100.0)})
        self.assertEqual(summary["results"]["axes"]["heatmap"]["score"], 100.0)
        table = (self.root / "export/paper_table.tex").read_text()
        for title in ("Task validity", "Role assignment", "Paired heatmaps"):
            self.assertIn(title, table)
        self.assertIn("Good (2) & OK (1) & Bad (0)", (self.root / "export/rubric_table.tex").read_text())
        self.assertNotIn("session_secret", json.dumps(summary))

    def test_frontend_and_assets_are_served(self):
        client = Client(self.app)
        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        for element in ('id="steps"', 'id="views"', 'id="roles"', 'id="lightbox"', 'id="guide"', 'id="meter-mine"'):
            self.assertIn(element, page.text)
        self.assertNotIn("Adjudicate", page.text)
        for asset in ("app.js", "app.css", "fonts/InterVariable.woff2"):
            self.assertEqual(client.get(f"/static/{asset}").status_code, 200, asset)
        self.assertEqual(client.get("/static/../app.py").status_code, 404)
        self.assertEqual(client.get("/api/me").status_code, 401)
        config = client.get("/api/config").json()
        self.assertEqual([axis["id"] for axis in config["rubric"]["axes"]], ["task", "role", "heatmap"])
        alice = Client(self.app, "alice")
        sample = alice.next()
        self.assertEqual(alice.get(sample["images"]["A"]).content, (self.root / "0.a.png").read_bytes())
        self.assertEqual(client.get(sample["images"]["A"]).status_code, 401)


def write_pilot(root: Path, changed_index: int) -> Path:
    """A minimal stage2-human-review-v1.0 database. Its sample IDs differ from the review IDs on
    purpose: the import must link by (object_id, task)."""
    db = root / "pilot.sqlite3"
    connection = sqlite3.connect(db)
    connection.executescript("""
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE samples(sample_id TEXT PRIMARY KEY, object_id TEXT, task TEXT, heat_a_path TEXT, heat_b_path TEXT);
        CREATE TABLE annotations(id INTEGER PRIMARY KEY, sample_id TEXT, username TEXT, kind TEXT, label INTEGER,
                                 defect_tags_json TEXT, comment TEXT, created_at TEXT);
        CREATE TABLE adjudications(id INTEGER PRIMARY KEY, sample_id TEXT, username TEXT, label INTEGER,
                                   defect_tags_json TEXT, comment TEXT, created_at TEXT);
        CREATE TABLE final_labels(sample_id TEXT PRIMARY KEY, label INTEGER);""")
    connection.execute("INSERT INTO metadata VALUES('schema_version', '\"stage2-human-review-v1.0\"')")
    frag = json.dumps(["fragmented_or_incomplete"])
    # pair 0: both Good. pair 1: Good vs OK, adjudicated OK by EN. pair 2: both Good, render changed since.
    votes = {0: (2, 2, None), 1: (2, 1, 1), 2: (2, 2, None)}
    for index, (mine, theirs, decided) in votes.items():
        sid = f"pilot-{index}"
        x = 40 if index == changed_index else 4
        connection.execute("INSERT INTO samples VALUES(?,?,?,?,?)", (
            sid, f"object-{index}", f"open the lid {index}",
            heat_image(root / f"pilot{index}.a.png", x), heat_image(root / f"pilot{index}.b.png", x)))
        for user, label in (("michaellee", mine), ("FengDian-Su", theirs)):
            connection.execute("INSERT INTO annotations(sample_id,username,kind,label,defect_tags_json,comment,created_at) "
                               "VALUES(?,?,'blind',?,?,'','2026-08-10T00:00:00+00:00')",
                               (sid, user, label, frag if label < 2 else "[]"))
        if decided is not None:
            connection.execute("INSERT INTO adjudications(sample_id,username,label,defect_tags_json,comment,created_at) "
                               "VALUES(?,'EN',?,?,'','2026-08-13T00:00:00+00:00')", (sid, decided, frag))
        connection.execute("INSERT INTO final_labels VALUES(?,?)", (sid, decided if decided is not None else mine))
    connection.commit()
    connection.close()
    return db


class PilotCarryOverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.db = self.root / "review.sqlite3"
        initialize_database(self.db, write_fixture(self.root))
        self.summary = import_v1(self.db, write_pilot(self.root, changed_index=2), workers=2)
        self.app = create_app(self.db)
        self.ids = [f"object-{index}/q{index}_test_task" for index in range(3)]

    def tearDown(self):
        self.temporary.cleanup()

    def mine(self, client: Client, sample_id: str) -> dict:
        return client.get("/api/mine", params={"sample_id": sample_id}).json()

    def test_import_links_by_object_and_task_and_skips_changed_renders(self):
        self.assertEqual((self.summary["linked"], self.summary["carried"], self.summary["rerated"]), (3, 2, 1))
        self.assertEqual(self.summary["carried_finals"], {"1": 1, "2": 1}, "the pilot gold is reproduced")
        self.assertEqual(Client(self.app, "zoe").rate(self.ids[0]).status_code, 200)
        with self.assertRaises(SystemExit, msg="no second import once anyone has rated"):
            import_v1(self.db, self.root / "pilot.sqlite3", workers=2)

    def test_pilot_raters_get_their_own_heatmap_prefilled(self):
        lee, en, zoe = Client(self.app, "michaellee"), Client(self.app, "EN"), Client(self.app, "zoe")
        offered = lee.next()
        expected = ({"heatmap": {"label": 2, "tags": []}}, ["task", "role"]) if offered["sample_id"] != self.ids[2] \
            else ({}, ["task", "role", "heatmap"])
        self.assertEqual((offered["carried"], offered["axes"]), expected)
        self.assertEqual(lee.rate(self.ids[0], TWO, TWO_TAGS, key="lee-two-axes").status_code, 200,
                         "the pre-filled pilot heatmap need not be sent again")
        self.assertEqual(self.mine(lee, self.ids[0])["rating"]["pilot"], ["heatmap"])
        self.assertEqual(lee.rate(self.ids[2], TWO, TWO_TAGS, key="lee-two-changed").status_code, 422,
                         "a pair whose render changed is rated in full")
        self.assertEqual(zoe.rate(self.ids[1], TWO, TWO_TAGS, key="zoe-two-axes").status_code, 422,
                         "new raters rate all three")
        self.assertEqual(en.rate(self.ids[1], TWO, TWO_TAGS, key="en-two-axes").status_code, 422,
                         "EN only adjudicated in the pilot, so EN scores the heatmap too")
        self.assertEqual(en.rate(self.ids[1], key="en-three-axes").status_code, 200)

    def test_pilot_raters_may_change_their_heatmap(self):
        lee = Client(self.app, "michaellee")
        self.assertEqual(lee.rate(self.ids[1], TWO, TWO_TAGS, key="lee-one-two").status_code, 200)
        self.assertEqual(self.mine(lee, self.ids[1])["sample"]["carried"], {"heatmap": {"label": 2, "tags": []}})
        changed = lee.rate(self.ids[1], {**TWO, "heatmap": 1}, {**TWO_TAGS, "heatmap": ["fragmented_or_incomplete"]},
                           key="lee-one-heatmap", endpoint="/api/revisions")
        self.assertEqual(changed.status_code, 200, changed.text)
        mine = self.mine(lee, self.ids[1])
        self.assertEqual((mine["rating"]["labels"]["heatmap"], mine["rating"]["pilot"]), (1, []))

    def test_pilot_heatmaps_count_as_those_raters_scores(self):
        heatmap = Client(self.app, "zoe").get("/api/dashboard").json()["results"]["axes"]["heatmap"]
        self.assertEqual({user: (entry["n"], entry["score"]) for user, entry in heatmap["per_rater"].items()},
                         {"FengDian-Su": (2, 75.0), "michaellee": (2, 100.0)},
                         "EN's adjudications are not ratings; the changed pair carries nothing")
        self.assertAlmostEqual(heatmap["score"], 87.5)


class AccessControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db = Path(self.temporary.name) / "review.sqlite3"
        initialize_database(self.db, write_fixture(Path(self.temporary.name)))

    def tearDown(self):
        self.temporary.cleanup()

    def test_only_allowed_networks_get_in(self):
        app = create_app(self.db, allow=["140.113.0.0/16", "2001:f18::/32"])
        status = lambda address, **headers: Client(app, address=address).get("/", headers=headers).status_code
        self.assertEqual(status("140.113.0.229"), 200, "campus address, direct")
        self.assertEqual(status("8.8.8.8"), 403, "outside address, direct")
        self.assertEqual(status("127.0.0.1"), 200, "this machine itself")
        self.assertEqual(status("127.0.0.1", **{"CF-Connecting-IP": "2001:f18:1::5"}), 200, "campus IPv6 via tunnel")
        self.assertEqual(status("127.0.0.1", **{"CF-Connecting-IP": "114.36.223.193"}), 403, "home address via tunnel")
        self.assertEqual(status("8.8.8.8", **{"CF-Connecting-IP": "140.113.1.1"}), 403,
                         "the header is only trusted from the local tunnel")
        self.assertIn("114.36.223.193", Client(app).get("/", headers={"CF-Connecting-IP": "114.36.223.193"}).text)

    def test_team_code_gates_sign_in(self):
        app = create_app(self.db, team_code="mocha42")
        client = Client(app)
        self.assertTrue(client.get("/api/config").json()["needs_code"])
        self.assertEqual(client.post("/api/login", json={"username": "zoe"}).status_code, 403)
        self.assertEqual(client.post("/api/login", json={"username": "zoe", "code": "latte"}).status_code, 403)
        self.assertEqual(client.post("/api/login", json={"username": "zoe", "code": " mocha42 "}).status_code, 200)
        self.assertEqual(client.get("/api/me").status_code, 200)
        self.assertFalse(Client(create_app(self.db)).get("/api/config").json()["needs_code"])


if __name__ == "__main__":
    unittest.main()
