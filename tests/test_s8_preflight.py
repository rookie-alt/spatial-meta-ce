"""S8 planning/preflight scope tests; actual cache reader is mocked."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_s8_preflight import (
    build_plan,
    holdout_identity_metadata,
    preflight,
    verify_token_duplicates,
)


class PreflightTests(unittest.TestCase):
    def test_self_consistent_but_unfrozen_reference_is_rejected_before_cache_read(self):
        from scripts.s2_source_replay import TRAIN, VALIDATION

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            reference = root / 'fake.json'
            reference.write_text(json.dumps({'sources': {
                db: {'file_sha256': {name: 'fake' for name in (
                    'manifest.json', 'cards.npy', 'offsets.npy', 'group0.npy',
                    'group1.npy', 'group2.npy', 'group3.npy')}}
                for db in (*TRAIN, *VALIDATION)}}))
            with patch('scripts.run_s8_preflight.replay.safe') as resolve:
                with self.assertRaisesRegex(ValueError, 'reference'):
                    preflight(root, reference)
                resolve.assert_not_called()

    def test_plan_never_claims_formal_execution(self):
        from scripts.s2_source_replay import TRAIN

        report = build_plan(list(TRAIN))
        self.assertEqual(report['state'], 'planned')
        self.assertFalse(report['training'])
        self.assertFalse(report['inference'])
        self.assertFalse(report['formal_ready'])
        self.assertEqual(report['panel_unique'], 19456)
        self.assertEqual(len(report['split']['holdout']), 4)
        with self.assertRaises(ValueError):
            build_plan(list(TRAIN) + ['imdb'])

    def test_preflight_rejects_incomplete_source_set_before_reading_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            meta = root / 'preflight.json'
            meta.write_text(json.dumps({'sources': {'imdb': {}}}))
            with patch('scripts.run_s8_preflight.replay.read_cache') as read:
                with self.assertRaises(ValueError):
                    preflight(root, meta)
                read.assert_not_called()

    def test_no_target_path_is_read_from_extra_manifest_source(self):
        from scripts.s2_source_replay import TRAIN, VALIDATION

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            meta = root / 'preflight.json'
            meta.write_text(json.dumps({'sources': {db: {} for db in (*TRAIN, *VALIDATION, 'imdb')}}))
            with patch('scripts.run_s8_preflight.replay.read_cache') as read:
                with self.assertRaises(ValueError):
                    preflight(root, meta)
                read.assert_not_called()


    def test_duplicate_token_conflict_fails_without_dropping_row(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp).resolve()
            (folder / 'manifest.json').write_text(json.dumps({'rows': [
                {'query_hash': 'same'}, {'query_hash': 'same'}]}))
            np.save(folder / 'offsets.npy', np.array([[0]*4, [1]*4, [2]*4]))
            for group, dim in enumerate((40, 40, 4, 43)):
                np.save(folder / f'group{group}.npy', np.ones((2, dim), dtype=np.float32))
            self.assertEqual(verify_token_duplicates(folder), 1)
            bad = np.ones((2, 40), dtype=np.float32)
            bad[1, 0] = 2
            np.save(folder / 'group0.npy', bad)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                verify_token_duplicates(folder)

    def test_nonfinite_and_negative_fanout_reject_before_nomination(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp).resolve()
            (folder / 'manifest.json').write_text(json.dumps({'rows': [{'query_hash': 'one'}]}))
            np.save(folder / 'offsets.npy', np.array([[0]*4, [1]*4]))
            for group, dim in enumerate((40, 40, 4, 43)):
                np.save(folder / f'group{group}.npy', np.ones((1, dim), dtype=np.float32))
            for invalid in (float('nan'), float('inf'), -1.0):
                bad = np.ones((1, 40), dtype=np.float32)
                bad[0, 0] = invalid
                np.save(folder / 'group1.npy', bad)
                with self.assertRaisesRegex(ValueError, 'finite|fanout'):
                    verify_token_duplicates(folder)


    def test_holdout_identity_metadata_never_opens_labels_or_features(self):
        import numpy as np

        from scripts.s2_source_replay import digest

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp).resolve()
            document = {'rows': [{'query_hash': 'same'}, {'query_hash': 'same'}],
                        'unique_queries': 1}
            document['content_hash'] = digest(document)
            (folder / 'manifest.json').write_text(json.dumps(document))
            np.save(folder / 'offsets.npy', np.array([[0]*4, [1]*4, [2]*4]))
            # No cards/group files exist: a premature label/token read must fail.
            value = holdout_identity_metadata(folder, {'rows': 2, 'manifest_hash': document['content_hash']})
            self.assertEqual(value['query_hash'], ['same'])
            self.assertNotIn('true', value)
            self.assertNotIn('pg', value)
            self.assertEqual(value['raw_rows'], 2)


    def test_holdout_inspection_is_never_decoded_or_rehashed_as_label_statistics(self):
        import numpy as np

        from scripts.s2_source_replay import digest

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp).resolve()
            document = {'rows': [{'query_hash': 'one'}], 'unique_queries': 1,
                        'inspection': {'true_min': 123, 'true_max': 987}}
            document['content_hash'] = digest(document)
            (folder / 'manifest.json').write_text(json.dumps(document))
            np.save(folder / 'offsets.npy', np.array([[0]*4, [1]*4]))
            with patch('scripts.run_s8_preflight.json.loads', side_effect=AssertionError('full parse')), patch('scripts.run_s8_preflight.replay.digest', side_effect=AssertionError('label digest')):
                result = holdout_identity_metadata(folder, {'rows': 1, 'manifest_hash': document['content_hash']})
            self.assertEqual(result['query_hash'], ['one'])

    def test_holdout_duplicate_table_count_conflict_is_rejected(self):
        import numpy as np

        from scripts.s2_source_replay import digest

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp).resolve()
            document = {'rows': [{'query_hash': 'same'}, {'query_hash': 'same'}], 'unique_queries': 1}
            document['content_hash'] = digest(document)
            (folder / 'manifest.json').write_text(json.dumps(document))
            np.save(folder / 'offsets.npy', np.array([[0]*4, [1]*4, [2, 2, 5, 2]]))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                holdout_identity_metadata(folder, {'rows': 2, 'manifest_hash': document['content_hash']})

    def test_metadata_reader_rejects_symlink_directory_before_loading(self):
        import numpy as np

        from scripts.s2_source_replay import digest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            outside = root / 'outside'
            outside.mkdir()
            document = {'rows': [{'query_hash': 'one'}], 'unique_queries': 1}
            document['content_hash'] = digest(document)
            (outside / 'manifest.json').write_text(json.dumps(document))
            np.save(outside / 'offsets.npy', np.array([[0]*4, [1]*4]))
            alias = root / 'alias'
            alias.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'link|path'):
                holdout_identity_metadata(alias, {'rows': 1, 'manifest_hash': document['content_hash']})


    def test_projection_decodes_only_identity_fields_with_nested_opaque_values(self):
        from scripts.run_s8_preflight import identity_projection

        text = json.dumps({'inspection': {'true_min': 123, 'nested': [{'s': '} , "esc"'}]},
                           'rows': [{'query_hash': 'q'}], 'unique_queries': 1, 'content_hash': 'bound'})
        projected = identity_projection(text)
        self.assertEqual(projected, {'rows': [{'query_hash': 'q'}], 'unique_queries': 1, 'content_hash': 'bound'})
        for invalid in ('{}', '{"rows":[]', '{"rows":[],"rows":[]}'):
            with self.assertRaises(ValueError):
                identity_projection(invalid)

    def test_secure_open_retains_original_directory_when_renamed(self):
        import os

        from scripts.run_s8_preflight import safe_binary

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            inside, outside = root / 'inside', root / 'outside'
            inside.mkdir()
            outside.mkdir()
            (inside / 'value').write_bytes(b'original')
            (outside / 'value').write_bytes(b'OUTSIDE_SENTINEL')
            original_open = os.open
            replaced = False

            def open_after_directory_fd(path, flags, *args, **kwargs):
                nonlocal replaced
                fd = original_open(path, flags, *args, **kwargs)
                if path == 'inside' and not replaced:
                    replaced = True
                    inside.rename(root / 'held-original')
                    inside.symlink_to(outside, target_is_directory=True)
                return fd

            with patch('scripts.run_s8_preflight.os.open', side_effect=open_after_directory_fd), safe_binary(inside / 'value') as stream:
                self.assertEqual(stream.read(), b'original')
            self.assertTrue(replaced)
            with self.assertRaises(ValueError), safe_binary(inside / 'value'):
                self.fail('Must reject link before content read')


if __name__ == '__main__':
    unittest.main()
