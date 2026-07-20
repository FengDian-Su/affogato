#!/usr/bin/env python
"""Deterministic full-scan audit of stage0_full outputs: schema, field completeness,
internal consistency, anomaly signals. Writes flagged records + stats to scratchpad."""
import json, os, sys
from collections import Counter, defaultdict

OUT = os.environ.get('STAGE0_DIR',
    '/home/michaellee/mclee/affogato/data_generation/outputs/stage0/daily_used')
SCRATCH = os.path.dirname(os.path.abspath(__file__))
MAP = '/home/michaellee/mclee/affogato/data_generation/dataset/daily_used_to_affogato.json'

data = [d for d in json.load(open(MAP)) if d.get('dst')]
exp_ids = [d['object_id'] for d in data]

flags = defaultdict(list)   # issue -> [(part, object_id, detail)]
stats = Counter()
comp_counts = Counter()
name_counter = Counter()
kept_single_body = []

REQ = {"object_id", "object_name", "views_used", "keep", "filter", "components"}
FILT_REQ = {"everyday_object", "bimanual_task_exists", "example_task", "keep"}

def flag(issue, part, oid, detail=""):
    flags[issue].append((part, oid, str(detail)[:160]))

import glob
PARTS = sorted(f for f in glob.glob(f'{OUT}/stage0_part*.json') if not f.endswith('.kept.json'))
for i, pf in enumerate(PARTS):
    rs = json.load(open(pf))
    for r in rs:
        oid = r.get('object_id', '?')
        stats['records'] += 1
        keys = set(r.keys())
        is_skip = 'skip_reason' in r
        is_err = 'error' in r
        if is_err:
            flag('error_record', i, oid, r.get('error')); continue

        # -- schema
        missing = REQ - keys
        if missing and not is_skip:
            flag('missing_fields', i, oid, f'missing {sorted(missing)}')
        if is_skip and r.get('skip_reason') not in (None,) and not isinstance(r.get('skip_reason'), str):
            flag('bad_skip_reason_type', i, oid, type(r.get('skip_reason')))

        name = r.get('object_name')
        keep = r.get('keep')
        comps = r.get('components')
        filt = r.get('filter')
        views = r.get('views_used')

        # -- skip records: legitimate shapes
        if is_skip and r['skip_reason'].startswith('name/views'):
            stats['skip_nameviews'] += 1
            continue
        if is_skip and r['skip_reason'] == 'corrupt/unreadable renders':
            stats['skip_corrupt'] += 1
            continue

        # -- full records
        stats['processed'] += 1
        name_counter[str(name)] += 1
        if not isinstance(name, str) or not name.strip():
            flag('empty_name', i, oid)
        if not isinstance(views, list) or len(views) < 5:
            flag('too_few_views', i, oid, f'{len(views) if isinstance(views, list) else views}')
        if not isinstance(filt, dict):
            flag('filter_not_dict', i, oid, type(filt))
        else:
            if 'error' in filt or 'raw' in filt:
                flag('filter_parse_failed', i, oid, filt.get('error', ''))
            else:
                fm = FILT_REQ - set(filt.keys())
                if fm:
                    flag('filter_missing_keys', i, oid, sorted(fm))
                for bkey in ('everyday_object', 'bimanual_task_exists', 'keep'):
                    if bkey in filt and not isinstance(filt[bkey], bool):
                        flag('filter_nonbool', i, oid, f'{bkey}={filt[bkey]!r}')
                # coherence: model said keep but fields contradict
                if filt.get('keep') and not (filt.get('everyday_object') and filt.get('bimanual_task_exists')):
                    flag('filter_incoherent_keep', i, oid,
                         f"keep=T but everyday={filt.get('everyday_object')} bimanual={filt.get('bimanual_task_exists')}")
                if filt.get('keep') and not filt.get('example_task'):
                    flag('keep_no_example_task', i, oid)

        if keep:
            stats['kept'] += 1
            if not comps:
                flag('kept_empty_components', i, oid)
            else:
                comp_counts[len(comps)] += 1
                names = [str(c.get('name', '')).strip().lower() for c in comps if isinstance(c, dict)]
                for c in comps:
                    if not isinstance(c, dict):
                        flag('component_not_dict', i, oid, c); continue
                    if not str(c.get('name', '')).strip():
                        flag('component_empty_name', i, oid)
                    if not str(c.get('interaction', '')).strip():
                        flag('component_empty_interaction', i, oid, c.get('name'))
                    extra = set(c.keys()) - {'name', 'interaction'}
                    if extra:
                        flag('component_extra_keys', i, oid, sorted(extra))
                    if len(str(c.get('name', ''))) > 60:
                        flag('component_name_too_long', i, oid, c.get('name'))
                if len(names) != len(set(names)):
                    flag('component_dup_names', i, oid, names)
                if len(comps) > 8:
                    flag('component_count_high', i, oid, len(comps))
                if names == ['body']:
                    kept_single_body.append((i, oid, str(name)))
            # keep=True must mean filter said keep too
            if isinstance(filt, dict) and 'error' not in filt and not filt.get('keep'):
                flag('keep_true_filter_false', i, oid)
        else:
            # keep False: either filter rejected, or no-components downgrade (needs skip_reason)
            if isinstance(filt, dict) and filt.get('keep') and r.get('skip_reason') != 'no components extracted':
                flag('filter_keep_but_record_false', i, oid, r.get('skip_reason'))
            if r.get('skip_reason') == 'no components extracted':
                stats['no_comps_downgrade'] += 1
            if comps:
                flag('rejected_but_has_components', i, oid, len(comps))

print(f"records scanned: {stats['records']} (processed {stats['processed']}, "
      f"skip name/views {stats['skip_nameviews']}, skip corrupt {stats['skip_corrupt']})")
print(f"kept: {stats['kept']}  no-comps downgrades: {stats['no_comps_downgrade']}")
print(f"\ncomponent-count histogram (kept): {dict(sorted(comp_counts.items()))}")
print(f"kept with ONLY 'body' component: {len(kept_single_body)} "
      f"({len(kept_single_body)/max(stats['kept'],1)*100:.1f}% of kept)")
print(f"\nmost common object_names (top 10): {name_counter.most_common(10)}")

print("\n=== FLAGS ===")
for k in sorted(flags, key=lambda k: -len(flags[k])):
    print(f"{k:32} {len(flags[k]):>6}")
json.dump({k: v for k, v in flags.items()},
          open(os.path.join(SCRATCH, 'audit_flags.json'), 'w'), indent=1, ensure_ascii=False)
json.dump(kept_single_body, open(os.path.join(SCRATCH, 'audit_single_body.json'), 'w'), indent=1)
print(f"\nflag details -> {SCRATCH}/audit_flags.json")
