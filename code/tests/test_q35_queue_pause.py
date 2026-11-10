import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import q35_queue_pause as pause
import q35_after_s05_handoff as handoff
import hard_queue_supervisor as supervisor


class QueuePauseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runs = Path(self.tmp.name)
        self.state = self.runs / 'queue_q35_hard'
        self.state.mkdir()
        self.run = self.runs / 'hard_queue/S05_ucf_seed42'
        self.run.mkdir(parents=True)
        self.marker = self.state / 'pause_after_S05.json'
        self.marker.write_text(json.dumps(dict(active=True, pause_after='S05')))
        self.fence = self.runs / 'untouched_needs_manual.json'
        self.fence.write_text('preserve original fence')

    def invoke(self, source='test'):
        return pause.honor_user_pause(source, self.state, self.runs)

    def test_absent_request_preserves_default_path(self):
        self.marker.unlink()
        self.assertIsNone(self.invoke())
        self.assertFalse((self.state / 'status.json').exists())

    def test_running_S05_is_pending_without_touching_run(self):
        status = self.run / 'status.json'
        status.write_text('{"status":"training","step":380,"pid":295755}')
        before = status.read_bytes()
        result = self.invoke()
        self.assertEqual(result['status'], 'pause_pending_S05_completion')
        self.assertFalse(result['launch_authorized'])
        self.assertEqual(status.read_bytes(), before)

    def test_only_full_verified_400_is_completed_pause(self):
        for receipt in [dict(passed=False, steps=400), dict(passed=True, steps=399)]:
            (self.run / 'verification.json').write_text(json.dumps(receipt))
            self.assertEqual(self.invoke()['status'], 'pause_pending_S05_completion')
        (self.run / 'verification.json').write_text('{"passed":true,"steps":400}')
        self.assertEqual(self.invoke()['status'], 'paused_by_user_after_S05')

    def test_malformed_request_fails_closed(self):
        self.marker.write_text('{"active":false,"pause_after":"S05"}')
        with self.assertRaises(AssertionError):
            self.invoke()

    def check_entry(self, entry):
        (self.run / 'verification.json').write_text('{"passed":true,"steps":400}')
        real_guard = pause.honor_user_pause
        def local_guard(source):
            return real_guard(source, self.state, self.runs)
        with patch.object(pause, 'honor_user_pause', side_effect=local_guard), \
             patch.object(entry, 'STATE', self.state), \
             patch('subprocess.run', side_effect=AssertionError('Unexpected process start')), \
             patch('subprocess.Popen', side_effect=AssertionError('Unexpected process start')), \
             patch('subprocess.check_output', side_effect=AssertionError('Unexpected system call')), \
             patch.object(handoff, 'release_markers', side_effect=AssertionError('Fence release forbidden')):
            entry.main()
        self.assertEqual(self.fence.read_text(), 'preserve original fence')
        result = json.loads((self.state / 'status.json').read_text())
        self.assertEqual(result['status'], 'paused_by_user_after_S05')
        self.assertFalse(result['launch_authorized'])

    def test_handoff_stops_before_fence_release_and_launch(self):
        self.check_entry(handoff)
        self.assertEqual(json.loads((self.state / 'handoff_status.json').read_text())['status'], 'paused_by_user_after_S05')

    def test_supervisor_stops_before_gpu_or_experiment_work(self):
        self.check_entry(supervisor)


if __name__ == '__main__':
    unittest.main()
