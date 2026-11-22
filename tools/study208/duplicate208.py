"""Opaque duplicate-bee descriptive census; unknown is not a negative."""
from collections import Counter
import json
from pathlib import Path


def replay(path, core=None):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    assert value['schema'] == 'switch208-opaque-duplicate-census-v1'
    result = {}
    for clip in value['clips']:
        rows = clip['rows']
        assert len({r['key'] for r in rows}) == len(rows)
        counts = Counter(r['state'] for r in rows)
        assert set(counts) <= {'yes', 'not_confirmed_other_track', 'unknown'}
        assert dict(counts) == clip['counts']
        if core is not None:
            report = next(c for c in core['clips'] if c['domain'] == clip['domain'] and c['clip'] == clip['clip'])
            denominators = [a['all_known_positive'] for a in report['selected_adjacency'].values()]
            assert all(n == len(rows) for n in denominators)
        hive = result.setdefault(clip['domain'], Counter())
        hive.update(counts)
    return {h: dict(c) for h, c in result.items()}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('evidence', type=Path)
    print(json.dumps(replay(p.parse_args().evidence), indent=2))
