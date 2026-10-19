"""S8 source-only metadata preflight. Default plans; never trains or infers."""

import argparse
import hashlib
import json
import os
import signal
import stat
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import s2_source_replay as replay  # noqa: E402
from spatial_meta_ce.s8_contract import (  # noqa: E402
    HELD_OUT,
    TRAIN,
    panel_indices,
    partition_databases,
)

REFERENCE_SHA256 = '902b0b0bb4594620354ff6ccee58bfdc9e36ebe188dc874913871810a601ab96'

FILES = ('manifest.json', 'cards.npy', 'offsets.npy',
         'group0.npy', 'group1.npy', 'group2.npy', 'group3.npy')


def build_plan(databases):
    return {
        'purpose': 's8-g1-source-metadata-only-not-experiment',
        'state': 'planned', 'split': partition_databases(databases),
        'panel_unique': 19 * 1024, 'training': False, 'inference': False,
        'adaptation': False, 'target_data_read': False, 'formal_ready': False,
        'not_yet_verified': ['fresh F0 panel isolation',
                             'protected runner/checkpoint/nomination/independent endpoint',
                             'resource smoke and formal budget'],
    }


@contextmanager
def safe_binary(path):
    """Open every path component with O_NOFOLLOW, retaining stable directory FDs."""
    path = Path(path).absolute()
    if '..' in path.parts:
        raise ValueError('Unsafe cache path traversal')
    directory, stream = None, None
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        directory = os.open('/', flags)
        for component in path.parts[1:-1]:
            child = os.open(component, flags, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        stream = os.fdopen(fd, 'rb')
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('Cache path must be a regular file')
        yield stream
    except OSError as error:
        raise ValueError(f'Unsafe or unavailable cache path/link: {path}') from error
    finally:
        if stream is not None:
            stream.close()
        if directory is not None:
            os.close(directory)


def safe_hash(path):
    sha = hashlib.sha256()
    with safe_binary(path) as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def safe_array(path):
    with safe_binary(path) as stream:
        return np.load(stream, allow_pickle=False)


def identity_projection(text):
    """Decode only rows/count/content_hash; skip other values as opaque text."""
    decoder = json.JSONDecoder()
    allowed = {'rows', 'unique_queries', 'content_hash'}
    result, keys = {}, set()
    i = 0

    def whitespace(position):
        while position < len(text) and text[position].isspace():
            position += 1
        return position

    i = whitespace(i)
    if i >= len(text) or text[i] != '{':
        raise ValueError('Invalid source manifest object')
    i += 1
    while True:
        i = whitespace(i)
        if i >= len(text):
            raise ValueError('Truncated source manifest')
        if text[i] == '}':
            if whitespace(i + 1) != len(text) or set(result) != allowed:
                raise ValueError('Incomplete source identity projection')
            return result
        key, i = decoder.raw_decode(text, i)
        if not isinstance(key, str) or key in keys:
            raise ValueError('Invalid/duplicate manifest field')
        keys.add(key)
        i = whitespace(i)
        if i >= len(text) or text[i] != ':':
            raise ValueError('Invalid source manifest separator')
        start = whitespace(i + 1)
        i, depth, quoted, escaped = start, 0, False, False
        while i < len(text):
            char = text[i]
            if quoted:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in '[{':
                depth += 1
            elif char in ']}':
                if depth == 0:
                    break
                depth -= 1
            elif char == ',' and depth == 0:
                break
            i += 1
        if key in allowed:
            result[key], end = decoder.raw_decode(text, start)
            if whitespace(end) != i:
                raise ValueError('Invalid projected identity field')
        i = whitespace(i)
        if i < len(text) and text[i] == ',':
            i += 1
        elif i >= len(text) or text[i] != '}':
            raise ValueError('Invalid manifest value boundary')


def verify_token_duplicates(folder):
    """Check full frozen group arrays and compare every repeated identity's tokens."""
    folder = Path(folder)
    with safe_binary(folder / 'manifest.json') as stream:
        manifest = json.loads(stream.read().decode('utf-8'))
    hashes = [row['query_hash'] for row in manifest['rows']]
    offsets = safe_array(folder / 'offsets.npy')
    if (offsets.dtype.kind not in 'iu' or offsets.shape != (len(hashes) + 1, 4)
            or (offsets < 0).any() or (offsets[0] != 0).any()
            or (offsets[1:] < offsets[:-1]).any()):
        raise ValueError('Invalid token group offsets')
    first, repeated = {}, []
    for i, h in enumerate(hashes):
        if h in first:
            repeated.append((first[h], i))
        else:
            first[h] = i
    for group, dim in enumerate((40, 40, 4, 43)):
        values = safe_array(folder / f'group{group}.npy')
        if (values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != dim
                or values.shape[0] != int(offsets[-1, group])):
            raise ValueError('Invalid frozen feature group shape/dtype')
        for start in range(0, len(values), 65536):
            block = values[start:start + 65536]
            if not np.isfinite(block).all():
                raise ValueError('Nonfinite frozen token features')
            if group == 1 and (block < 0).any():
                raise ValueError('Negative frozen fanout tokens')
        for original, duplicate in repeated:
            a, b = (values[offsets[i, group]:offsets[i + 1, group]]
                    for i in (original, duplicate))
            if not np.array_equal(a, b):
                raise ValueError('Duplicate identity has conflicting token content')
    return len(repeated)


def source_identity_metadata(folder, frozen, labels):
    folder = Path(folder)
    with safe_binary(folder / 'manifest.json') as stream:
        raw = stream.read().decode('utf-8')
    manifest = json.loads(raw) if labels else identity_projection(raw)
    if manifest.get('content_hash') != frozen['manifest_hash']:
        raise ValueError('Source manifest identity mismatch')
    if labels and replay.digest({k: v for k, v in manifest.items() if k != 'content_hash'}) != frozen['manifest_hash']:
        raise ValueError('Source manifest semantic hash mismatch')
    hashes = [row['query_hash'] for row in manifest['rows']]
    if (not hashes or len(hashes) != frozen['rows']
            or any(not isinstance(h, str) or not h for h in hashes)):
        raise ValueError('Invalid source identity denominator')
    offsets = safe_array(folder / 'offsets.npy')
    if (offsets.shape != (len(hashes) + 1, 4) or offsets.dtype.kind not in 'iu'
            or (offsets < 0).any() or (offsets[0] != 0).any()
            or (offsets[1:] < offsets[:-1]).any()):
        raise ValueError('Invalid source count offsets')
    tables = offsets[1:, 2] - offsets[:-1, 2]
    if (tables < 1).any():
        raise ValueError('Invalid source table counts')
    _, first, inverse = np.unique(hashes, return_index=True, return_inverse=True)
    if not np.array_equal(tables, tables[first[inverse]]):
        raise ValueError('Duplicate identity has conflicting table counts')
    selected = np.sort(first)
    if len(selected) != manifest['unique_queries']:
        raise ValueError('Source unique identity mismatch')
    result = {'query_hash': [hashes[i] for i in selected], 'indices': selected,
              'n_tables': tables[selected], 'raw_rows': len(hashes)}
    if labels:
        cards = safe_array(folder / 'cards.npy')
        if (cards.shape != (len(hashes), 2) or cards.dtype.kind not in 'fiu'
                or not np.isfinite(cards).all() or (cards < 0).any()):
            raise ValueError('Invalid source label/PG array')
        if not np.array_equal(cards, cards[first[inverse]]):
            raise ValueError('Duplicate identity has conflicting labels/PG')
        result.update({'true': cards[selected, 0], 'pg': cards[selected, 1]})
    return result


def holdout_identity_metadata(folder, frozen):
    """Before nomination H decodes only identity/count, never inspection ranges."""
    return source_identity_metadata(folder, frozen, labels=False)


def preflight(data_root, reference):
    """Authenticate only the 26 known source caches and freeze panel identities."""
    data_root, reference = Path(data_root).absolute(), Path(reference).absolute()
    if safe_hash(reference) != REFERENCE_SHA256:
        raise ValueError('S8 frozen S6 reference authentication mismatch')
    with safe_binary(reference) as stream:
        sources = json.loads(stream.read().decode('utf-8'))['sources']
    permitted = {*replay.TRAIN, *replay.VALIDATION}
    if set(sources) != permitted:
        raise ValueError('S8 incomplete/extra/target source authentication set')
    report = build_plan(list(replay.TRAIN))
    bound = {str(reference): safe_hash(reference)}
    # Paths come only from the literal source whitelist, never input-supplied db names.
    for db in (*replay.TRAIN, *replay.VALIDATION):
        file_hashes = sources[db]['file_sha256']
        if set(file_hashes) != set(FILES):
            raise ValueError('S8 unknown cache file authentication entry')
        for name, sha in file_hashes.items():
            path = data_root / 'artifacts/cache/pretrain' / db / name
            if safe_hash(path) != sha:
                raise ValueError(f'S8 historical source bytes mismatch: {db}/{name}')
            bound[str(path)] = sha
    rows = []
    for db in (*replay.TRAIN, *replay.VALIDATION):
        folder = data_root / 'artifacts/cache/pretrain' / db
        if db in HELD_OUT:
            cache = holdout_identity_metadata(folder, sources[db])
            duplicate_tokens = None
        else:
            cache = source_identity_metadata(folder, sources[db], labels=True)
            duplicate_tokens = verify_token_duplicates(folder)
        n = len(cache['query_hash'])
        p, q = panel_indices(db, cache['query_hash']) if db in TRAIN else ([], [])
        role = 'T' if db in TRAIN else 'H' if db in HELD_OUT else 'V'
        all_four_plus = int(np.count_nonzero(cache['n_tables'] >= 4))
        row = {
            'database': db, 'role': role, 'raw_rows': cache['raw_rows'],
            'unique': n, 'duplicates': cache['raw_rows'] - n,
            'four_plus': all_four_plus, 'one_to_three': n - all_four_plus,
            'identity_digest': replay.digest(cache['query_hash']),
            'duplicate_token_pairs_verified': duplicate_tokens,
        }
        if role == 'T':
            row.update({
                'P_unique': len(p), 'Q_unique': len(q),
                'P_identity_digest': replay.digest([cache['query_hash'][i] for i in p]),
                'Q_identity_digest': replay.digest([cache['query_hash'][i] for i in q]),
                'P_four_plus': int(np.count_nonzero(cache['n_tables'][p] >= 4)),
                'Q_four_plus': int(np.count_nonzero(cache['n_tables'][q] >= 4)),
            })
        rows.append(row)
    for path, expected in bound.items():
        if safe_hash(Path(path)) != expected:
            raise ValueError(f'S8 inputs changed during preflight: {path}')
    report.update({
        'state': 'verified_metadata', 'formal_ready': False, 'rows': rows,
        'input_sha256': bound, 'inputs_unchanged': True,
        'totals': {role: {
            'unique': sum(r['unique'] for r in rows if r['role'] == role),
            'four_plus': sum(r['four_plus'] for r in rows if r['role'] == role),
        } for role in ('T', 'H', 'V')},
        'P_unique': sum(r.get('P_unique', 0) for r in rows),
        'Q_unique': sum(r.get('Q_unique', 0) for r in rows),
        'T_V_duplicate_token_content_verified': True,
        'H_identity_counts_sha_only': True,
        'H_inspection_values_decoded': False,
        'nofollow_directory_file_reads': True,
        'H_token_label_validation_pending_nomination': True,
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Read-only source metadata; not training')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(build_plan(list(replay.TRAIN)), ensure_ascii=False, indent=2))
        return
    if args.data_root is None or args.reference is None:
        parser.error('--execute requires --data-root and authenticated --reference')
    signal.alarm(300)
    started = time.monotonic()
    report = preflight(args.data_root, args.reference)
    report['seconds'] = time.monotonic() - started
    report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
