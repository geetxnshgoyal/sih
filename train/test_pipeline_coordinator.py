"""Fast tests for batching, priority and resumable coordinator state."""
import tempfile
import types
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import pipeline
from pipeline_words import priority_key


class PipelineCoordinatorTest(unittest.TestCase):
    def test_clinical_words_sort_before_daily_and_remaining(self):
        labels = ["zebra", "hello", "please", "water", "airport"]
        self.assertEqual(sorted(labels, key=priority_key)[:2], ["please", "water"])
        self.assertEqual(priority_key("hello")[0], 1)
        self.assertEqual(priority_key("zebra")[0], 2)

    def test_second_sync_does_not_duplicate_completed_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, meta = root / "data", root / "data/meta"
            source = pipeline.Source("ncert", "fake.py", "ncert_landmarks",
                                     "test", "https://example.invalid")
            args = types.SimpleNamespace(batch_size=1, priority="clinical", source="all")

            def fake_run(*_args, **_kwargs):
                output = data / source.output / "please/one.npz"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.touch(exist_ok=True)
                return types.SimpleNamespace(returncode=0)

            original = (pipeline.ROOT, pipeline.DATA, pipeline.META, pipeline.STATE,
                        pipeline.MANIFEST, pipeline.SOURCES)
            pipeline.ROOT, pipeline.DATA, pipeline.META = root, data, meta
            pipeline.STATE, pipeline.MANIFEST = meta / "state.json", meta / "manifest.jsonl"
            pipeline.SOURCES = (source,)
            try:
                with patch.object(pipeline.subprocess, "run", side_effect=fake_run):
                    self.assertEqual(pipeline.sync(args), 0)
                    self.assertEqual(pipeline.sync(args), 0)
                state = pipeline.load_state()
                self.assertEqual([run["added"] for run in state["runs"]], [1, 0])
                self.assertEqual(pipeline.count_outputs(source), 1)
            finally:
                (pipeline.ROOT, pipeline.DATA, pipeline.META, pipeline.STATE,
                 pipeline.MANIFEST, pipeline.SOURCES) = original


if __name__ == "__main__":
    unittest.main()
