#!/usr/bin/env python3
"""Release-quality human review: a local multi-rater web app that scores DUALingo pairs on three
axes (task validity, role assignment, paired heatmaps), each Bad / OK / Good.

Every rater scores every pair independently. A criterion's score is each rater's own score (mean
rating / 2 x 100) averaged over the raters; there is no consensus or adjudication step. Pilot
heatmap ratings can be carried over (see import_v1.py), so a rating row may cover only some axes.
Ratings live in SQLite; the release files are read-only inputs.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import statistics
import tempfile
from typing import Annotated, Any, Iterator, Literal
from urllib.parse import quote

from fastapi import Cookie, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from data_verification.quality_review.rubric import AXIS_IDS, SCALE, rubric_payload


REPO = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_ROOT = REPO / "data_verification/outputs/quality_review/release1000"
DEFAULT_DB = DEFAULT_ROOT / "review.sqlite3"
DEFAULT_MANIFEST = DEFAULT_ROOT / "manifest.jsonl"
SCHEMA_VERSION = "quality-review-v3.0"
COOKIE_NAME = "quality_review_session"
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{2,32}$")
IMAGE_KINDS = {"rgb": "rgb_path", "A": "heat_a_path", "B": "heat_b_path"}
SessionCookie = Annotated[str | None, Cookie(alias=COOKIE_NAME)]
# Every prefix NYCU announces from its own AS9916 (RIPEstat, 2026-10-06): `--allow nycu`.
NYCU_NETWORKS = (
    "140.113.0.0/16", "140.129.51.0/24", "140.129.52.0/22", "140.129.56.0/21", "140.129.64.0/20",
    "140.129.80.0/24", "120.126.32.0/19", "120.126.64.0/19", "120.126.96.0/20", "2001:f18::/32",
)
ASSETS = {
    "app.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "fonts/InterVariable.woff2": "font/woff2",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def stable_queue_key(seed: int, sample_id: str) -> int:
    raw = hashlib.sha256(f"{seed}\0{sample_id}".encode()).digest()[:8]
    return int.from_bytes(raw, "big") & ((1 << 63) - 1)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS samples (
  sample_id TEXT PRIMARY KEY,
  sample_index INTEGER NOT NULL,
  object_id TEXT NOT NULL,
  dataset TEXT,
  split TEXT,
  object_name TEXT NOT NULL,
  task TEXT NOT NULL,
  roles_json TEXT NOT NULL,
  view_labels_json TEXT NOT NULL,
  rgb_path TEXT NOT NULL,
  heat_a_path TEXT NOT NULL,
  heat_b_path TEXT NOT NULL,
  queue_key INTEGER NOT NULL,
  manifest_json TEXT NOT NULL,
  carried_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS samples_queue_idx ON samples(queue_key, sample_id);
CREATE TABLE IF NOT EXISTS annotations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  sample_id TEXT NOT NULL REFERENCES samples(sample_id) ON DELETE CASCADE,
  username TEXT NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN ('blind','revision','imported')),
  task INTEGER CHECK(task BETWEEN 0 AND 2),
  role INTEGER CHECK(role BETWEEN 0 AND 2),
  heatmap INTEGER CHECK(heatmap BETWEEN 0 AND 2),
  tags_json TEXT NOT NULL,
  comment TEXT NOT NULL DEFAULT '',
  elapsed_ms INTEGER,
  supersedes_id INTEGER REFERENCES annotations(id),
  idempotency_key TEXT NOT NULL UNIQUE,
  source TEXT,
  created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS annotation_one_blind_per_user
  ON annotations(sample_id, username) WHERE kind='blind';
CREATE INDEX IF NOT EXISTS annotation_sample_idx ON annotations(sample_id, id);
CREATE INDEX IF NOT EXISTS annotation_user_idx ON annotations(username, id);
CREATE TABLE IF NOT EXISTS leases (
  username TEXT PRIMARY KEY,
  sample_id TEXT NOT NULL REFERENCES samples(sample_id) ON DELETE CASCADE,
  expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS leases_sample_idx ON leases(sample_id, expires_at);
CREATE TABLE IF NOT EXISTS adjudications (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  sample_id TEXT NOT NULL REFERENCES samples(sample_id) ON DELETE CASCADE,
  username TEXT NOT NULL,
  labels_json TEXT NOT NULL,
  tags_json TEXT NOT NULL,
  comment TEXT NOT NULL DEFAULT '',
  source TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS final_labels (
  sample_id TEXT NOT NULL REFERENCES samples(sample_id) ON DELETE CASCADE,
  axis TEXT NOT NULL CHECK(axis IN ('task','role','heatmap')),
  label INTEGER NOT NULL CHECK(label BETWEEN 0 AND 2),
  resolution TEXT NOT NULL CHECK(resolution IN ('consensus','adjudicated')),
  resolved_by TEXT,
  tags_json TEXT NOT NULL,
  source_ids_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(sample_id, axis)
);
"""
# adjudications / final_labels only hold the pilot's history (import_v1.py); scores never use them.


def connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield connection
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()


def get_metadata(connection: sqlite3.Connection, key: str, default: Any = KeyError) -> Any:
    row = connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    if row is None:
        if default is KeyError:
            raise KeyError(key)
        return default
    return json.loads(row["value"])


def set_metadata(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT INTO metadata(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, json.dumps(value, ensure_ascii=False, sort_keys=True)))


# ----------------------------------------------------------------------------- validation

def validate_rating(
    labels: dict[str, int],
    tags: dict[str, list[str]],
    comment: str,
    axis_tags: dict[str, set[str]],
    axes: tuple[str, ...] | list[str] = AXIS_IDS,
) -> tuple[dict[str, int], dict[str, list[str]], str]:
    """Check one rating over `axes` and return it normalised."""
    if set(labels) != set(axes):
        raise ValueError(f"labels must cover exactly: {', '.join(axes)}")
    if set(tags) - set(axes):
        raise ValueError(f"tags given for unexpected axes: {sorted(set(tags) - set(axes))}")
    clean_labels, clean_tags = {}, {}
    for axis in axes:
        label = labels[axis]
        if type(label) is not int or label not in SCALE:
            raise ValueError(f"{axis}: label must be 0 (Bad), 1 (OK) or 2 (Good)")
        chosen = sorted(set(tags.get(axis, [])))
        unknown = set(chosen) - axis_tags[axis]
        if unknown:
            raise ValueError(f"{axis}: unknown tags {sorted(unknown)}")
        if label < 2 and not chosen:
            raise ValueError(f"{axis}: Bad and OK need at least one reason tag")
        if label == 2 and chosen:
            raise ValueError(f"{axis}: Good cannot carry reason tags")
        clean_labels[axis], clean_tags[axis] = label, chosen
    comment = comment.strip()
    if any("other" in chosen for chosen in clean_tags.values()) and not comment:
        raise ValueError("the 'Other' tag needs a note")
    if len(comment) > 2000:
        raise ValueError("note is longer than 2000 characters")
    return clean_labels, clean_tags, comment


# ----------------------------------------------------------------------------- import

def inspect_manifest(manifest: Path, seed: int, check_images: bool) -> list[dict[str, Any]]:
    rows, seen = [], set()
    for source in read_jsonl(manifest):
        sid = source.get("sample_id")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError(f"{manifest}: missing or duplicate sample ID {sid!r}")
        seen.add(sid)
        roles = source.get("roles") or []
        if len(roles) != 2 or not str(source.get("task") or "").strip():
            raise ValueError(f"{sid}: needs a task and exactly two roles")
        paths = {field: str(source.get(field) or "") for field in IMAGE_KINDS.values()}
        if check_images:
            for field, value in paths.items():
                path = Path(value)
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"{sid}: missing or empty {field}: {value}")
        rows.append({
            "sample_id": sid,
            "sample_index": len(rows),
            "object_id": str(source.get("object_id") or sid.split("/", 1)[0]),
            "dataset": source.get("dataset"),
            "split": source.get("split"),
            "object_name": str(source.get("object_name") or "Object"),
            "task": str(source["task"]),
            "roles_json": json.dumps(roles, ensure_ascii=False),
            "view_labels_json": json.dumps(source.get("view_labels") or []),
            **paths,
            "queue_key": stable_queue_key(seed, sid),
            "manifest_json": json.dumps(source, ensure_ascii=False, sort_keys=True),
        })
    if not rows:
        raise ValueError(f"{manifest}: no samples")
    return rows


def initialize_database(
    db_path: Path,
    manifest: Path,
    *,
    raters_per_sample: int = 2,
    seed: int = 20261006,
    check_images: bool = True,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    # raters_per_sample is kept for import_v1.py, which re-derives the pilot's two-rater finals.
    rows = inspect_manifest(manifest, seed, check_images)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "rubric": rubric_payload(),
        "raters_per_sample": raters_per_sample,
        "queue_seed": seed,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": file_sha256(manifest),
        "samples": len(rows),
        "split_distribution": dict(Counter(str(row["split"]) for row in rows)),
    }
    if dry_run:
        return {**provenance, "dry_run": True}
    if db_path.exists() and not force:
        raise FileExistsError(f"database already exists: {db_path}; use --force to replace it")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = db_path.with_suffix(db_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = connect(temporary)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(SCHEMA)
        columns = ("sample_id", "sample_index", "object_id", "dataset", "split", "object_name",
                   "task", "roles_json", "view_labels_json", "rgb_path", "heat_a_path",
                   "heat_b_path", "queue_key", "manifest_json")
        with transaction(connection):
            connection.executemany(
                f"INSERT INTO samples({','.join(columns)}) VALUES({','.join('?' * len(columns))})",
                [tuple(row[column] for column in columns) for row in rows],
            )
            provenance["created_at"] = utcnow()
            for key, value in {**provenance, "session_secret": secrets.token_hex(32)}.items():
                set_metadata(connection, key, value)
    finally:
        connection.close()
    for suffix in ("-wal", "-shm"):
        stale = Path(str(db_path) + suffix)
        if stale.exists():
            stale.unlink()
    os.replace(temporary, db_path)
    return provenance


# ----------------------------------------------------------------------------- ratings

def axis_ratings(connection: sqlite3.Connection, sample_id: str | None = None) -> dict[str, dict]:
    """sample -> axis -> rater -> that rater's current rating on that axis.

    A row may cover only some axes (a pilot import carries the heatmap alone), so a rater's current
    label is their newest non-null value per axis. `first_id` orders raters by when they first
    rated that axis."""
    where = "WHERE sample_id=?" if sample_id else ""
    ratings: dict[str, dict] = defaultdict(lambda: {axis: {} for axis in AXIS_IDS})
    for row in connection.execute(f"SELECT * FROM annotations {where} ORDER BY id",
                                  (sample_id,) if sample_id else ()):
        tags = json.loads(row["tags_json"])
        for axis in AXIS_IDS:
            if row[axis] is None:
                continue
            previous = ratings[row["sample_id"]][axis].get(row["username"])
            ratings[row["sample_id"]][axis][row["username"]] = {
                "label": row[axis], "tags": tags.get(axis, []), "id": row["id"], "kind": row["kind"],
                "first_id": previous["first_id"] if previous else row["id"],
            }
    return ratings


def raters_view(connection: sqlite3.Connection, sample_id: str) -> list[dict[str, Any]]:
    """One entry per rater of a sample: current label per axis, which axes came from the pilot,
    and their latest note."""
    per_axis = axis_ratings(connection, sample_id)[sample_id]
    notes = {row["username"]: row["comment"] for row in connection.execute(
        "SELECT username, comment FROM annotations WHERE sample_id=? AND comment != '' ORDER BY id",
        (sample_id,))}
    order: dict[str, int] = {}
    for axis in AXIS_IDS:
        for username, rating in per_axis[axis].items():
            order[username] = min(order.get(username, rating["first_id"]), rating["first_id"])
    return [{
        "username": username,
        "labels": {axis: per_axis[axis].get(username, {}).get("label") for axis in AXIS_IDS},
        "tags": {axis: per_axis[axis].get(username, {}).get("tags", []) for axis in AXIS_IDS},
        "pilot": [axis for axis in AXIS_IDS if per_axis[axis].get(username, {}).get("kind") == "imported"],
        "comment": notes.get(username, ""),
    } for username in sorted(order, key=order.get)]


def recompute_finals(connection: sqlite3.Connection, sample_id: str, raters_per_sample: int) -> None:
    """The pilot's rule, used by import_v1.py to check the carried ratings: per axis the first N
    raters decide, unanimous means a consensus final; adjudicated axes are left alone."""
    per_axis = axis_ratings(connection, sample_id)[sample_id]
    adjudicated = {row["axis"] for row in connection.execute(
        "SELECT axis FROM final_labels WHERE sample_id=? AND resolution='adjudicated'", (sample_id,))}
    for axis in AXIS_IDS:
        if axis in adjudicated:
            continue
        deciders = sorted(per_axis[axis].values(), key=lambda rating: rating["first_id"])[:raters_per_sample]
        labels = {rating["label"] for rating in deciders}
        if len(deciders) == raters_per_sample and len(labels) == 1:
            connection.execute(
                """INSERT INTO final_labels(sample_id,axis,label,resolution,resolved_by,tags_json,
                                            source_ids_json,created_at)
                   VALUES(?,?,?,'consensus',NULL,?,?,?)
                   ON CONFLICT(sample_id,axis) DO UPDATE SET label=excluded.label,
                     tags_json=excluded.tags_json,source_ids_json=excluded.source_ids_json,
                     created_at=excluded.created_at""",
                (sample_id, axis, labels.pop(), json.dumps(sorted({t for r in deciders for t in r["tags"]})),
                 json.dumps([rating["id"] for rating in deciders]), utcnow()),
            )
        else:
            connection.execute(
                "DELETE FROM final_labels WHERE sample_id=? AND axis=? AND resolution='consensus'",
                (sample_id, axis))


def rating_plan(connection: sqlite3.Connection, sample: sqlite3.Row,
                username: str) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """(axes this rater must score, axes pre-filled for them). An axis carried over from the pilot
    starts pre-filled with the rater's own pilot rating, which they may change; a rater without one
    (including the pilot's adjudicator, whose decisions were not independent) scores it like
    everyone else."""
    settled = set(json.loads(sample["carried_json"]))
    per_axis = axis_ratings(connection, sample["sample_id"])[sample["sample_id"]]
    carried = {axis: {"label": per_axis[axis][username]["label"], "tags": per_axis[axis][username]["tags"]}
               for axis in AXIS_IDS if axis in settled and username in per_axis[axis]}
    return [axis for axis in AXIS_IDS if axis not in carried], carried


def public_sample(row: sqlite3.Row, axes: list[str] | tuple[str, ...] = AXIS_IDS,
                  carried: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    sid = quote(row["sample_id"], safe="")
    return {
        "sample_id": row["sample_id"],
        "object_name": row["object_name"],
        "task": row["task"],
        "roles": json.loads(row["roles_json"]),
        "view_labels": json.loads(row["view_labels_json"]),
        "images": {kind: f"/api/image?sample_id={sid}&kind={kind}" for kind in IMAGE_KINDS},
        "axes": list(axes),
        "carried": carried or {},
    }


def progress(connection: sqlite3.Connection, username: str) -> dict[str, int]:
    return {
        "samples": connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0],
        "mine": connection.execute("SELECT COUNT(*) FROM annotations WHERE kind='blind' AND username=?",
                                   (username,)).fetchone()[0],
    }


def claim_next(
    connection: sqlite3.Connection,
    username: str,
    lease_minutes: int = 20,
    exclude_sample_id: str | None = None,
) -> dict[str, Any] | None:
    """The rater's next unrated pair, in the shared queue order. The hold only remembers the pair a
    rater is on (so a reload shows it again); it never blocks anyone else."""
    now = utcnow()
    expiry = (datetime.now(timezone.utc) + timedelta(minutes=lease_minutes)).isoformat()
    with transaction(connection):
        connection.execute("DELETE FROM leases WHERE expires_at <= ?", (now,))
        held = connection.execute(
            "SELECT s.* FROM leases l JOIN samples s USING(sample_id) WHERE l.username=?", (username,)
        ).fetchone()
        if held and held["sample_id"] != exclude_sample_id:
            return public_sample(held, *rating_plan(connection, held, username))
        connection.execute("DELETE FROM leases WHERE username=?", (username,))
        row = connection.execute(
            """SELECT s.* FROM samples s
               WHERE (:skip IS NULL OR s.sample_id != :skip)
               AND NOT EXISTS(SELECT 1 FROM annotations a WHERE a.sample_id=s.sample_id
                              AND a.username=:u AND a.kind IN ('blind','revision'))
               ORDER BY s.queue_key, s.sample_id LIMIT 1""",
            {"skip": exclude_sample_id, "u": username},
        ).fetchone()
        if row is None:
            return None
        connection.execute("INSERT INTO leases(username,sample_id,expires_at) VALUES(?,?,?)",
                           (username, row["sample_id"], expiry))
        return public_sample(row, *rating_plan(connection, row, username))


# ----------------------------------------------------------------------------- statistics

def krippendorff_alpha(units: list[list[Any]], metric: str = "ordinal",
                       categories: tuple = (0, 1, 2)) -> float | None:
    """Krippendorff's alpha over units holding any number of values (units with <2 are skipped)."""
    index = {value: i for i, value in enumerate(categories)}
    k = len(categories)
    coincidence = [[0.0] * k for _ in range(k)]
    for values in units:
        if len(values) < 2:
            continue
        counts = Counter(index[value] for value in values)
        for c, n_c in counts.items():
            for d, n_d in counts.items():
                coincidence[c][d] += n_c * (n_d - (c == d)) / (len(values) - 1)
    marginals = [sum(row) for row in coincidence]
    total = sum(marginals)

    def delta(c: int, d: int) -> float:
        if metric == "nominal":
            return float(c != d)
        if metric == "interval":
            return float(categories[c] - categories[d]) ** 2
        low, high = min(c, d), max(c, d)
        return (sum(marginals[low:high + 1]) - (marginals[low] + marginals[high]) / 2) ** 2

    observed = sum(coincidence[c][d] * delta(c, d) for c in range(k) for d in range(k))
    expected = sum(marginals[c] * marginals[d] * delta(c, d) for c in range(k) for d in range(k))
    if total <= 1 or expected == 0:
        return None
    return 1.0 - (total - 1) * observed / expected


def score(labels: list[int]) -> float:
    """Mean rating (Bad 0 / OK 1 / Good 2) on a 0-100 scale."""
    return sum(labels) / len(labels) / 2 * 100


def results(connection: sqlite3.Connection, everything: dict | None = None) -> dict[str, Any]:
    """Per criterion: every rater's own score over the pairs they rated, and the plain average of
    those scores (each rater counts once), with their spread. Good % is averaged the same way."""
    everything = everything if everything is not None else axis_ratings(connection)
    labels: dict[str, dict[str, list[int]]] = defaultdict(lambda: {axis: [] for axis in AXIS_IDS})
    for per_axis in everything.values():
        for axis in AXIS_IDS:
            for username, rating in per_axis[axis].items():
                labels[username][axis].append(rating["label"])
    axes = {}
    for axis in AXIS_IDS:
        per_rater = {username: {"n": len(mine[axis]), "score": score(mine[axis]),
                                "good_pct": 100 * mine[axis].count(2) / len(mine[axis])}
                     for username, mine in sorted(labels.items()) if mine[axis]}
        scores = [entry["score"] for entry in per_rater.values()]
        pooled = [label for mine in labels.values() for label in mine[axis]]
        axes[axis] = {
            "raters": len(per_rater),
            "ratings": len(pooled),
            "score": statistics.mean(scores) if scores else None,
            "score_sd": statistics.stdev(scores) if len(scores) > 1 else None,
            "good_pct": statistics.mean(entry["good_pct"] for entry in per_rater.values()) if per_rater else None,
            "distribution": [pooled.count(value) for value in (0, 1, 2)],
            "per_rater": per_rater,
        }
    return {"axes": axes}


def agreement(connection: sqlite3.Connection, everything: dict | None = None) -> dict[str, Any]:
    """Inter-rater reliability per axis over every rater's current label."""
    everything = everything if everything is not None else axis_ratings(connection)
    output: dict[str, Any] = {}
    for axis in AXIS_IDS:
        units = [[rating["label"] for rating in per_axis[axis].values()]
                 for per_axis in everything.values() if len(per_axis[axis]) >= 2]
        pairs = [(a, b) for unit in units for i, a in enumerate(unit) for b in unit[i + 1:]]
        output[axis] = {
            "units": len(units),
            "alpha_ordinal": krippendorff_alpha(units, "ordinal"),
            "exact_agreement": sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
        }
    return output


def rater_table(connection: sqlite3.Connection, everything: dict | None = None) -> list[dict[str, Any]]:
    scores = results(connection, everything)["axes"]
    rows = connection.execute("SELECT username, kind, elapsed_ms, created_at FROM annotations").fetchall()
    table = []
    for username in sorted({row["username"] for row in rows}):
        mine = [row for row in rows if row["username"] == username]
        blind = [row for row in mine if row["kind"] == "blind"]
        elapsed = [row["elapsed_ms"] for row in blind if row["elapsed_ms"]]
        table.append({
            "username": username,
            "rated": len(blind),
            "pilot": sum(row["kind"] == "imported" for row in mine),
            "revised": sum(row["kind"] == "revision" for row in mine),
            "median_seconds": statistics.median(elapsed) / 1000 if elapsed else None,
            "score": {axis: scores[axis]["per_rater"].get(username, {}).get("score") for axis in AXIS_IDS},
            "last_at": max((row["created_at"] for row in blind), default=None),
        })
    return sorted(table, key=lambda row: (-row["rated"], row["username"]))


def tag_counts(everything: dict) -> dict[str, list[list[Any]]]:
    """How often each reason was given, over every rater's current Bad / OK labels."""
    counts: dict[str, Counter] = {axis: Counter() for axis in AXIS_IDS}
    for per_axis in everything.values():
        for axis in AXIS_IDS:
            for rating in per_axis[axis].values():
                counts[axis].update(rating["tags"])
    return {axis: [list(item) for item in counter.most_common()] for axis, counter in counts.items()}


# ----------------------------------------------------------------------------- web app

class LoginPayload(BaseModel):
    username: str
    code: str = ""

    @field_validator("username")
    @classmethod
    def valid_username(cls, value: str) -> str:
        value = value.strip()
        if not USERNAME_RE.fullmatch(value):
            raise ValueError("use 2-32 letters, numbers, dot, dash or underscore")
        return value


class RatingPayload(BaseModel):
    sample_id: str
    labels: dict[str, int]
    tags: dict[str, list[str]] = Field(default_factory=dict)
    comment: str = ""
    elapsed_ms: int | None = Field(default=None, ge=0)
    idempotency_key: str = Field(min_length=8, max_length=128)


def sign_session(username: str, secret: str) -> str:
    payload = base64.urlsafe_b64encode(username.encode()).decode().rstrip("=")
    signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


def verify_session(token: str | None, secret: str) -> str | None:
    if not token or "." not in token:
        return None
    payload, signature = token.rsplit(".", 1)
    expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        username = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)).decode()
    except Exception:
        return None
    return username if USERNAME_RE.fullmatch(username) else None


def client_ip(request: Request) -> str:
    """The visitor's address. Behind a Cloudflare tunnel every request arrives from cloudflared on
    this machine, which passes the real address in CF-Connecting-IP."""
    peer = request.client.host if request.client else ""
    forwarded = request.headers.get("cf-connecting-ip")
    return forwarded.strip() if forwarded and peer in ("127.0.0.1", "::1") else peer


def create_app(db_path: Path | str, lease_minutes: int = 20, allow: list[str] | None = None,
               team_code: str | None = None) -> FastAPI:
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"review database does not exist: {db_path}")
    probe = connect(db_path)
    try:
        secret = get_metadata(probe, "session_secret")
        rubric = get_metadata(probe, "rubric")
    finally:
        probe.close()
    if tuple(axis["id"] for axis in rubric["axes"]) != AXIS_IDS:
        raise ValueError(f"database rubric axes do not match this schema: {rubric['axes']}")
    axis_tags = {axis["id"]: {tag["id"] for tag in axis["tags"]} for axis in rubric["axes"]}
    config = {"schema_version": SCHEMA_VERSION, "rubric": rubric, "needs_code": bool(team_code)}
    networks = [ipaddress.ip_network(cidr, strict=False) for cidr in allow or []]

    app = FastAPI(title="DUALingo Quality Review", docs_url=None, redoc_url=None)

    if networks:
        @app.middleware("http")
        async def allowed_networks_only(request: Request, call_next):
            address = client_ip(request)
            try:
                ip = ipaddress.ip_address(address)
            except ValueError:
                ip = None
            if ip is not None and (any(ip in net for net in networks) or
                                   (ip.is_loopback and "cf-connecting-ip" not in request.headers)):
                return await call_next(request)
            return PlainTextResponse(
                f"This review site only accepts connections from the NYCU network (your address: {address}).\n"
                "Off campus, connect through the NYCU VPN first.\n", status_code=403)

    def user(token: str | None) -> str:
        username = verify_session(token, secret)
        if username is None:
            raise HTTPException(401, "not signed in")
        return username

    @contextmanager
    def db() -> Iterator[sqlite3.Connection]:
        connection = connect(db_path)
        try:
            yield connection
        finally:
            connection.close()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"),
                            headers={"Cache-Control": "no-store"})

    @app.get("/static/{asset:path}")
    async def static_asset(asset: str) -> Response:
        if asset not in ASSETS:
            raise HTTPException(404, "not found")
        return Response((STATIC / asset).read_bytes(), media_type=ASSETS[asset],
                        headers={"Cache-Control": "no-cache"})

    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        return config

    @app.post("/api/login")
    async def login(payload: LoginPayload, response: Response) -> dict[str, str]:
        if team_code and not hmac.compare_digest(payload.code.strip().encode(), team_code.encode()):
            raise HTTPException(403, "wrong team code")
        response.set_cookie(COOKIE_NAME, sign_session(payload.username, secret), httponly=True,
                            samesite="strict", secure=False, max_age=60 * 60 * 24 * 30)
        return {"username": payload.username}

    @app.post("/api/logout")
    async def logout(response: Response) -> dict[str, bool]:
        response.delete_cookie(COOKIE_NAME)
        return {"ok": True}

    @app.get("/api/me")
    async def me(session: SessionCookie = None) -> dict[str, Any]:
        username = user(session)
        with db() as connection:
            return {"username": username, "progress": progress(connection, username)}

    @app.post("/api/queue/next")
    async def next_sample(skip: str | None = Query(default=None),
                          session: SessionCookie = None) -> dict[str, Any]:
        username = user(session)
        with db() as connection:
            return {"sample": claim_next(connection, username, lease_minutes, skip),
                    "progress": progress(connection, username)}

    @app.get("/api/image")
    async def image(sample_id: str, kind: Literal["rgb", "A", "B"],
                    session: SessionCookie = None) -> Response:
        user(session)
        with db() as connection:
            row = connection.execute(f"SELECT {IMAGE_KINDS[kind]} path FROM samples WHERE sample_id=?",
                                     (sample_id,)).fetchone()
        if row is None or not Path(row["path"]).is_file():
            raise HTTPException(404, "image not found")
        path = Path(row["path"])
        media = {".webp": "image/webp", ".png": "image/png", ".jpg": "image/jpeg"}.get(path.suffix)
        return Response(path.read_bytes(), media_type=media or "application/octet-stream",
                        headers={"Cache-Control": "private, max-age=604800"})

    def save(payload: RatingPayload, username: str, kind: str) -> dict[str, Any]:
        with db() as connection, transaction(connection):
            sample = connection.execute("SELECT * FROM samples WHERE sample_id=?",
                                        (payload.sample_id,)).fetchone()
            if sample is None:
                raise HTTPException(404, "sample not found")
            required, _ = rating_plan(connection, sample, username)
            if not set(required) <= set(payload.labels) <= set(AXIS_IDS):
                raise HTTPException(422, f"rate {', '.join(required)}")
            axes = [axis for axis in AXIS_IDS if axis in payload.labels]
            try:
                labels, tags, comment = validate_rating(payload.labels, payload.tags, payload.comment,
                                                        axis_tags, axes)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            duplicate = connection.execute("SELECT id FROM annotations WHERE idempotency_key=?",
                                           (payload.idempotency_key,)).fetchone()
            if duplicate:
                return {"annotation_id": duplicate["id"], "duplicate": True}
            mine = connection.execute(
                """SELECT id FROM annotations WHERE sample_id=? AND username=?
                   AND kind IN ('blind','revision') ORDER BY id DESC LIMIT 1""",
                (payload.sample_id, username)).fetchone()
            if kind == "blind" and mine:
                raise HTTPException(409, "you have already rated this sample; edit it from History")
            if kind == "revision" and mine is None:
                raise HTTPException(403, "you can only edit samples you have rated")
            cursor = connection.execute(
                """INSERT INTO annotations(sample_id,username,kind,task,role,heatmap,tags_json,comment,
                                           elapsed_ms,supersedes_id,idempotency_key,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (payload.sample_id, username, kind, labels.get("task"), labels.get("role"),
                 labels.get("heatmap"), json.dumps(tags), comment, payload.elapsed_ms,
                 mine["id"] if kind == "revision" else None, payload.idempotency_key, utcnow()),
            )
            if kind == "blind":
                connection.execute("DELETE FROM leases WHERE username=? AND sample_id=?",
                                   (username, payload.sample_id))
            return {"annotation_id": cursor.lastrowid, "duplicate": False}

    @app.post("/api/annotations")
    async def annotate(payload: RatingPayload, session: SessionCookie = None) -> dict[str, Any]:
        return save(payload, user(session), "blind")

    @app.post("/api/revisions")
    async def revise(payload: RatingPayload, session: SessionCookie = None) -> dict[str, Any]:
        return save(payload, user(session), "revision")

    @app.get("/api/mine")
    async def my_rating(sample_id: str, session: SessionCookie = None) -> dict[str, Any]:
        username = user(session)
        with db() as connection:
            sample = connection.execute("SELECT * FROM samples WHERE sample_id=?", (sample_id,)).fetchone()
            rated = connection.execute(
                "SELECT 1 FROM annotations WHERE sample_id=? AND username=? AND kind IN ('blind','revision')",
                (sample_id, username)).fetchone()
            if sample is None or rated is None:
                raise HTTPException(404, "you have not rated this sample")
            mine = next(r for r in raters_view(connection, sample_id) if r["username"] == username)
            return {"sample": public_sample(sample, *rating_plan(connection, sample, username)), "rating": mine}

    @app.get("/api/history")
    async def history(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
                      session: SessionCookie = None) -> dict[str, Any]:
        username = user(session)
        with db() as connection:
            rows = connection.execute(
                """SELECT a.*, s.object_name, s.task AS task_text
                   FROM annotations a JOIN samples s USING(sample_id)
                   WHERE a.id IN (SELECT MAX(id) FROM annotations WHERE username=?
                                  AND kind IN ('blind','revision') GROUP BY sample_id)
                   ORDER BY a.id DESC LIMIT ? OFFSET ?""", (username, limit, offset)).fetchall()
            total = connection.execute(
                "SELECT COUNT(DISTINCT sample_id) FROM annotations WHERE username=? AND kind IN ('blind','revision')",
                (username,)).fetchone()[0]
            items = []
            for row in rows:
                mine = next(r for r in raters_view(connection, row["sample_id"]) if r["username"] == username)
                items.append({**mine, "sample_id": row["sample_id"], "object_name": row["object_name"],
                              "task": row["task_text"], "kind": row["kind"], "created_at": row["created_at"]})
        return {"total": total, "items": items}

    @app.get("/api/dashboard")
    async def dashboard(session: SessionCookie = None) -> dict[str, Any]:
        username = user(session)
        with db() as connection:
            everything = axis_ratings(connection)
            return {
                "progress": progress(connection, username),
                "results": results(connection, everything),
                "agreement": agreement(connection, everything),
                "raters": rater_table(connection, everything),
                "tags": tag_counts(everything),
                "carried": get_metadata(connection, "import_v1", {}).get("summary"),
            }

    @app.exception_handler(sqlite3.IntegrityError)
    async def sqlite_conflict(_request: Request, exc: sqlite3.IntegrityError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    return app


def update_rubric(db_path: Path) -> dict[str, Any]:
    """Swap in rubric.py's rubric when it only ADDS reason tags, so every existing rating stays valid.
    Anything else (axes, wording, verdicts, scale) must be unchanged."""
    connection = connect(db_path)
    try:
        with transaction(connection):
            old, new = get_metadata(connection, "rubric"), rubric_payload()
            added: dict[str, list[str]] = {}
            same = len(old["axes"]) == len(new["axes"]) and all(
                old[key] == new[key] for key in ("scale", "principles"))
            for before, after in zip(old["axes"], new["axes"]):
                kept_ids = [tag["id"] for tag in before["tags"]]
                kept = [tag for tag in after["tags"] if tag["id"] in kept_ids]
                same = same and {**after, "tags": kept} == before
                added[after["id"]] = [tag["id"] for tag in after["tags"] if tag["id"] not in kept_ids]
            if not same:
                raise SystemExit("rubric.py differs from the database by more than added reason tags")
            if not any(added.values()):
                raise SystemExit(f"nothing to add; the database already uses {old['version']}")
            change = {"from": old["version"], "to": new["version"], "at": utcnow(),
                      "added_tags": {axis: tags for axis, tags in added.items() if tags}}
            set_metadata(connection, "rubric", new)
            set_metadata(connection, "rubric_history", get_metadata(connection, "rubric_history", []) + [change])
    finally:
        connection.close()
    return change


# ----------------------------------------------------------------------------- export

def latex_escape(text: str) -> str:
    for char, replacement in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
                              ("_", r"\_"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}")):
        text = text.replace(char, replacement)
    return text


def paper_table(summary: dict[str, Any]) -> str:
    def fmt(value: float | None, digits: int) -> str:
        return "--" if value is None else f"{value:.{digits}f}"
    rows, raters = [], 0
    for axis in summary["rubric"]["axes"]:
        stats = summary["results"]["axes"][axis["id"]]
        raters = max(raters, stats["raters"])
        spread = f" $\\pm$ {fmt(stats['score_sd'], 1)}" if stats["score_sd"] is not None else ""
        rows.append(f"{latex_escape(axis['title'])} & {fmt(stats['score'], 1)}{spread} & "
                    f"{fmt(stats['good_pct'], 1)} & {fmt(summary['agreement'][axis['id']]['alpha_ordinal'], 2)} \\\\")
    return "\n".join([
        "% Generated by `python -m data_verification.quality_review.app export`; do not edit by hand.",
        f"% {raters} raters each scored every pair; score = each rater's mean rating (0/1/2) / 2 x 100,",
        "% averaged over raters (+- = std across raters); Good (%) averaged the same way;",
        "% alpha = ordinal Krippendorff's alpha across raters.",
        "\\begin{tabular}{@{}lccc@{}}",
        "\\toprule",
        "Criterion & Score $\\uparrow$ & Good (\\%) $\\uparrow$ & Krippendorff's $\\alpha$ \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ])


def rubric_table(rubric: dict[str, Any]) -> str:
    rows = []
    for axis in rubric["axes"]:
        cells = " & ".join(latex_escape(axis["verdicts"][key]) for key in ("2", "1", "0"))
        rows.append(f"\\textbf{{{latex_escape(axis['title'])}}}\\newline"
                    f"{{\\footnotesize\\itshape {latex_escape(axis['question'])}}} & {cells}")
    return "\n".join([
        f"% Rubric {rubric['version']}; generated by the quality_review export (needs booktabs).",
        "\\begin{tabular}{@{}p{0.19\\linewidth}p{0.24\\linewidth}p{0.24\\linewidth}p{0.24\\linewidth}@{}}",
        "\\toprule",
        "Criterion & Good (2) & OK (1) & Bad (0) \\\\",
        "\\midrule",
        " \\\\[3pt]\n".join(rows) + " \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ])


def export_database(db_path: Path, out_dir: Path) -> dict[str, Any]:
    connection = connect(db_path)
    try:
        provenance = {row["key"]: json.loads(row["value"]) for row in connection.execute(
            "SELECT key,value FROM metadata WHERE key != 'session_secret' ORDER BY key")}
        everything = axis_ratings(connection)
        per_pair = []
        for sample in connection.execute("SELECT * FROM samples ORDER BY sample_index").fetchall():
            per_axis = everything.get(sample["sample_id"])
            if not per_axis or not any(per_axis[axis] for axis in AXIS_IDS):
                continue
            manifest = json.loads(sample["manifest_json"])
            per_pair.append({
                "sample_id": sample["sample_id"],
                "object_id": sample["object_id"],
                "object_name": sample["object_name"],
                "dataset": sample["dataset"],
                "split": sample["split"],
                **{axis: {"score": score([r["label"] for r in per_axis[axis].values()]) if per_axis[axis] else None,
                          "labels": {username: r["label"] for username, r in sorted(per_axis[axis].items())}}
                   for axis in AXIS_IDS},
                "carried_from_pilot": json.loads(sample["carried_json"]),
                "release": {key: manifest.get(key) for key in ("meta_sha256", "scores_sha256")},
            })
        objects: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in per_pair:
            objects[row["object_id"]].append(row)
        per_object = [{
            "object_id": object_id,
            "object_name": rows[0]["object_name"],
            "dataset": rows[0]["dataset"],
            "pairs": len(rows),
            "sample_ids": [row["sample_id"] for row in rows],
            **{axis: (statistics.mean(scored) if (scored := [row[axis]["score"] for row in rows
                                                             if row[axis]["score"] is not None]) else None)
               for axis in AXIS_IDS},
        } for object_id, rows in objects.items()]
        raw = [{**dict(row), "tags": json.loads(row["tags_json"])}
               for row in connection.execute("SELECT * FROM annotations ORDER BY id")]
        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "exported_at": utcnow(),
            "database": str(db_path.resolve()),
            "rubric": provenance["rubric"],
            "pairs_rated": len(per_pair),
            "results": results(connection, everything),
            "agreement": agreement(connection, everything),
            "raters": rater_table(connection, everything),
            "tags": tag_counts(everything),
            "definitions": {
                "design": "every rater scores every pair independently; no consensus or adjudication",
                "score": "per criterion: each rater's mean rating (Bad 0 / OK 1 / Good 2) / 2 x 100, "
                         "then averaged over raters (each rater counts once); score_sd is the spread "
                         "across raters",
                "good_pct": "each rater's share of Good, averaged over raters",
                "per_pair": "per pair and criterion, the mean over its raters on the same 0-100 scale",
                "per_object": "per object and criterion, the mean of its pairs' scores",
                "alpha_ordinal": "Krippendorff's alpha (ordinal) across raters",
                "pilot": "two pilot raters' heatmap ratings are carried over where the released "
                         "render is unchanged (import_v1); they count as those raters' own ratings",
            },
            "provenance": provenance,
        }
    finally:
        connection.close()

    def jsonl(rows: list[dict[str, Any]]) -> str:
        return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)

    files = {
        "per_pair": (out_dir / "per_pair.jsonl", jsonl(per_pair)),
        "per_object": (out_dir / "per_object.jsonl", jsonl(per_object)),
        "annotations": (out_dir / "annotations.audit.jsonl", jsonl(raw)),
        "paper_table": (out_dir / "paper_table.tex", paper_table(summary)),
        "rubric_table": (out_dir / "rubric_table.tex", rubric_table(summary["rubric"])),
    }
    for path, content in files.values():
        atomic_write(path, content)
    summary["files"] = {key: {"path": str(path.resolve()), "sha256": file_sha256(path)}
                        for key, (path, _) in files.items()}
    atomic_write(out_dir / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


# ----------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="validate the manifest and create a review database")
    init.add_argument("--db", type=Path, default=DEFAULT_DB)
    init.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    init.add_argument("--seed", type=int, default=20261006)
    init.add_argument("--skip-image-check", action="store_true")
    init.add_argument("--force", action="store_true")
    init.add_argument("--dry-run", action="store_true")

    serve = sub.add_parser("serve", help="serve the review UI")
    serve.add_argument("--db", type=Path, default=DEFAULT_DB)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8770)
    serve.add_argument("--lease-minutes", type=int, default=20)
    serve.add_argument("--allow", action="append", default=[], metavar="CIDR",
                       help="only accept these networks (repeatable); 'nycu' = NYCU's own prefixes")
    serve.add_argument("--team-code-file", type=Path,
                       help="file holding a shared code that sign-in requires")

    update = sub.add_parser("update-rubric", help="add new reason tags from rubric.py to a live database")
    update.add_argument("--db", type=Path, default=DEFAULT_DB)

    export = sub.add_parser("export", help="write per-pair / per-object scores, audit trail and LaTeX tables")
    export.add_argument("--db", type=Path, default=DEFAULT_DB)
    export.add_argument("--out", type=Path, default=DEFAULT_ROOT / "export")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "init":
        result = initialize_database(args.db, args.manifest, seed=args.seed,
                                     check_images=not args.skip_image_check, force=args.force,
                                     dry_run=args.dry_run)
        result.pop("rubric", None)
    elif args.command == "serve":
        import uvicorn
        allow = [cidr for item in args.allow for cidr in (NYCU_NETWORKS if item == "nycu" else [item])]
        code = args.team_code_file.read_text(encoding="utf-8").strip() if args.team_code_file else None
        # proxy_headers off: the client address must be the real TCP peer for client_ip() to trust it.
        uvicorn.run(create_app(args.db, args.lease_minutes, allow, code), host=args.host, port=args.port,
                    proxy_headers=False)
        return
    elif args.command == "update-rubric":
        result = update_rubric(args.db)
    else:
        summary = export_database(args.db, args.out)
        result = {key: summary[key] for key in ("results", "agreement", "files")}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
