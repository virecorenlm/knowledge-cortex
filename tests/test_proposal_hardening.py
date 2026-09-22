"""Fail-closed proposal regression tests; all authorities are disposable."""
import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ingest.proposals import approve_proposal, create_proposals, compute_proposal_id, reject_proposal
from ingest.vault_writer import hash_managed_body
from tests.test_proposals import FakeObsidianForProposals, make_analyzer_result
from main import propose_vault_changes, _decision_cli


class HardeningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'doc.txt'
        self.source.write_text('source')
        self.state = self.root / 'state.json'
        self.directory = self.root / 'proposals'
        self.local = {str(self.source): {'generated_body': 'generated'}}
        self.save_state()
        self.ob = FakeObsidianForProposals({'test/note.md': 'human'})

    def save_state(self):
        self.state.write_text(json.dumps({'local': self.local}))

    def create(self, **changes):
        for old in self.directory.glob("*.json"):
            old.unlink()
        result = make_analyzer_result(source_path=str(self.source), managed_note_path='test/note.md',
            source_sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(),
            expected_generated_sha256=hash_managed_body('generated'),
            current_vault_sha256=hash_managed_body('human'), live_body='human')
        result.update(changes)
        created = create_proposals([result], local_state=self.local, proposals_dir=self.directory,
                                   state_path=self.state)
        self.proposal_id = created[0]['proposal_id']
        self.path = self.directory / (self.proposal_id + '.json')
        return json.loads(self.path.read_text())

    def approve(self, state=None):
        return asyncio.run(approve_proposal(self.directory, self.proposal_id, self.ob,
                                            state_path=state or self.state))

    def test_missing_note_captures_baseline_despite_null_analyzer_hash(self):
        data = self.create(classification='MISSING', proposed_action='recreate_managed_note',
                           expected_generated_sha256=None, current_vault_sha256=None, _live_vault_body=None)
        self.assertEqual(data['expected_generated_body'], 'generated')
        self.assertEqual(data['fingerprints']['expected_generated_sha256'], hash_managed_body('generated'))
        self.assertEqual(data['state_path'], str(self.state.resolve()))
        self.ob.files.clear()
        self.assertTrue(self.approve()['ok'])

    def test_unrelated_state_file_even_identical_content_is_refused(self):
        self.create()
        other = self.root / 'other.json'
        other.write_bytes(self.state.read_bytes())
        result = self.approve(other)
        self.assertFalse(result['ok'])
        self.assertIn('state', result['reason'])

    def test_tampered_records_are_unchanged_and_read_no_authorities(self):
        mutations = [
            ('fingerprints', []), ('fingerprints', 'bad'),
            ('reviewed_live_vault_body', 42), ('expected_generated_body', ['bad']),
            ('expected_generated_body', 'tampered'), ('expected_generated_body', None), ('schema_version', True),
            ('schema_version', 999), ('status', []), ('status', 'bogus'),
            ('source_path', []), ('managed_note_path', None), ('state_path', None),
            ('decision', {'status': 'approved'}),
        ]
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                data = self.create()
                data[field] = value
                self.path.write_text(json.dumps(data))
                before = self.path.read_bytes()
                with patch('ingest.proposals._current_fingerprints', side_effect=AssertionError('external read')):
                    result = self.approve()
                self.assertFalse(result['ok'])
                self.assertEqual(self.path.read_bytes(), before)
                self.path.unlink()

    def test_null_and_malformed_hashes_cannot_bypass_checks_even_with_recomputed_id(self):
        for key in ('source_sha256', 'expected_generated_sha256', 'live_vault_sha256'):
            for value in (None, '', 'not-sha256', 'g' * 64, 42, []):
                with self.subTest(key=key, value=value):
                    data = self.create()
                    data['fingerprints'][key] = value
                    fp = data['fingerprints']
                    data['proposal_id'] = compute_proposal_id(data['source_path'], data['managed_note_path'],
                        data['classification'], data['proposed_action'], fp['source_sha256'],
                        fp['expected_generated_sha256'], fp['live_vault_sha256'], state_path=data['state_path'],
                        source_exists=fp['source_exists'])
                    self.path.unlink()
                    self.proposal_id = data['proposal_id']
                    self.path = self.directory / (self.proposal_id + '.json')
                    self.path.write_text(json.dumps(data))
                    before = self.path.read_bytes()
                    with patch('ingest.proposals._current_fingerprints', side_effect=AssertionError('external read')):
                        result = self.approve()
                    self.assertFalse(result['ok'])
                    self.assertEqual(self.path.read_bytes(), before)
                    self.path.unlink()

    def test_filename_must_match_embedded_and_recomputed_id(self):
        self.create()
        renamed = self.directory / ('0' * 16 + '.json')
        self.path.rename(renamed)
        self.proposal_id = '0' * 16
        before = renamed.read_bytes()
        self.assertFalse(self.approve()['ok'])
        self.assertEqual(renamed.read_bytes(), before)

    def test_traversal_id_is_refused(self):
        self.create()
        self.proposal_id = '../proposals/' + self.proposal_id
        self.assertFalse(self.approve()['ok'])

    def test_malformed_or_missing_state_is_stale_without_external_writes(self):
        for value in ('{broken', '[]', 'null', '{"local": []}', '{"local": {}}', None):
            with self.subTest(value=value):
                self.create()
                if value is None:
                    self.state.unlink(missing_ok=True)
                else:
                    self.state.write_text(value)
                before = self.state.read_bytes() if self.state.exists() else None
                source_before, vault_before = self.source.read_bytes(), dict(self.ob.files)
                result = self.approve()
                self.assertEqual(result['status'], 'stale')
                self.assertIn('unverifiable', result['reason'])
                self.assertEqual(self.state.read_bytes() if self.state.exists() else None, before)
                self.assertEqual(self.source.read_bytes(), source_before)
                self.assertEqual(self.ob.files, vault_before)

    def test_missing_note_appearing_is_stale(self):
        self.create(classification='MISSING', current_vault_sha256=None, _live_vault_body=None)
        self.assertEqual(self.approve()['status'], 'stale')

    def test_missing_source_stays_missing_approves_but_reappearance_is_stale(self):
        changes = dict(classification='SOURCE_MISSING', current_source_sha256=None,
                       flags={'source_missing': True})
        data = self.create(**changes)
        self.assertIsNone(data['fingerprints']['source_sha256'])
        self.assertFalse(data['fingerprints']['source_exists'])
        self.source.unlink()
        self.assertTrue(self.approve()['ok'])
        self.source.write_text('source')
        self.create(**changes)
        self.assertEqual(self.approve()['status'], 'stale')

    def test_reject_after_baseline_drift_does_not_read_authorities(self):
        self.create()
        self.local[str(self.source)]['generated_body'] = 'changed'
        self.save_state()
        before = self.state.read_bytes()
        with patch('ingest.proposals._current_fingerprints', side_effect=AssertionError('external read')):
            self.assertTrue(reject_proposal(self.directory, self.proposal_id)['ok'])
        self.assertEqual(self.state.read_bytes(), before)

    def test_approval_never_constructs_qdrant_or_saves_state(self):
        self.create()
        with patch('graph.store.VectorStore', side_effect=AssertionError('VectorStore constructed')), \
             patch('qdrant_client.QdrantClient', side_effect=AssertionError('Qdrant constructed')), \
             patch('sync_cli.save_state', side_effect=AssertionError('state write')):
            self.assertTrue(self.approve()['ok'])

    def test_inconsistent_cached_body_is_not_persisted_as_candidate(self):
        with self.assertRaisesRegex(ValueError, 'generated'):
            self.create(expected_generated_sha256=hash_managed_body('different'))
        self.assertEqual(list(self.directory.glob('*.json')), [])

    def test_existing_malformed_proposal_is_not_repaired_on_regeneration(self):
        self.create()
        self.path.write_text('{corrupt')
        before = self.path.read_bytes()
        result = make_analyzer_result(source_path=str(self.source), managed_note_path='test/note.md',
            source_sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(),
            expected_generated_sha256=hash_managed_body('generated'),
            current_vault_sha256=hash_managed_body('human'), live_body='human')
        with self.assertRaisesRegex(ValueError, 'existing proposal'):
            create_proposals([result], local_state=self.local, proposals_dir=self.directory, state_path=self.state)
        self.assertEqual(self.path.read_bytes(), before)

    def test_cli_parser_forwards_custom_state_to_approval(self):
        import main
        with patch('sys.argv', ['main.py', '--approve-proposal', 'abc', '--state', str(self.state)]), \
             patch('main._decision_cli', return_value=0) as decision:
            self.assertEqual(main.main(), 0)
        self.assertEqual(decision.call_args.kwargs['state_path'], self.state)

    def test_missing_or_nonstring_cached_body_is_unverifiable(self):
        for entry in ({}, {'generated_body': None}, {'generated_body': []}):
            with self.subTest(entry=entry):
                self.create()
                self.state.write_text(json.dumps({'local': {str(self.source): entry}}))
                self.assertEqual(self.approve()['status'], 'stale')

    def test_cli_captures_generated_snapshot(self):
        result = make_analyzer_result(expected_generated_sha256=hash_managed_body('generated'),
                                      source_path=str(self.source))
        with patch('ingest.reverse_analyzer.analyze_managed_notes', return_value=[result]):
            self.assertEqual(propose_vault_changes(self.state, proposals_dir=self.directory, obsidian=self.ob), 0)
        data = json.loads(next(self.directory.glob('*.json')).read_text())
        self.assertEqual(data['expected_generated_body'], 'generated')
        self.assertEqual(data['state_path'], str(self.state.resolve()))
