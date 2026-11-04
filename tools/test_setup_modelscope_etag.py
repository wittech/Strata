"""ModelScope CDN ETag changes may be tolerated only when a pinned SHA-256 verifies the complete file."""
from __future__ import annotations

import contextlib
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup  # noqa: E402

sys.path.insert(0, str(Path(setup.__file__).resolve().parent / "tests" / "terminal"))
import test_download_integrity as fixtures


class ModelScopeEtags(unittest.TestCase):
    def setUp(self):
        self.case = fixtures.DownloadCase()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)

    @contextlib.contextmanager
    def source(self, url, pinned=True):
        with mock.patch.object(setup, "model_source", return_value="modelscope"), \
                mock.patch.object(setup, "ms_file", return_value=("ISTA-DASLab/test", "model.gguf")), \
                mock.patch.object(setup, "ms_url", return_value=url), \
                mock.patch.object(setup, "reachable", return_value=True), \
                mock.patch.object(setup, "ms_meta", return_value=None), \
                mock.patch.object(setup, "file_pin", return_value=(fixtures.SIZE, fixtures.SHA) if pinned else None):
            yield

    def serve(self, corrupt=False):
        def plan(n, start, end):
            return {"etag": f'"cdn-edge-{n}"', "corrupt": (1000,) if corrupt else ()}
        return self.case.serve(fixtures.Behavior(plan))

    def test_pinned_modelscope_file_finishes_with_different_edge_etags(self):
        url = self.serve()
        with self.source(url):
            self.case.run_download(url)
        self.case.assertGood()

    def test_existing_chunks_with_an_old_etag_resume_without_starting_over(self):
        url = self.serve()
        self.case.part.write_bytes(fixtures.DATA[:fixtures.PIECE * 3])
        self.assertTrue(setup._save_pieces(self.case.part, fixtures.SIZE, {0, 1, 2}))
        self.case.part.with_name(self.case.part.name + ".src").write_text(json.dumps({"url": url, "etag": "old-cdn-edge"}))
        with self.source(url):
            self.case.run_download(url)
        self.case.assertGood()
        ranges = [(start, end) for start, end in self.case.beh.served if end > start]
        self.assertTrue(ranges)
        self.assertTrue(all(start >= fixtures.PIECE * 3 for start, end in ranges), ranges)

    def test_unpinned_files_keep_strict_etag_checks(self):
        url = self.serve()
        with self.source(url, pinned=False):
            self.case.run_download(url, expect_exit=True)
        self.assertFalse(self.case.dst.exists())
        self.assertFalse(self.case.dst.with_name(self.case.dst.name + ".done").exists())

    def test_changed_content_still_fails_the_final_sha256_check(self):
        url = self.serve(corrupt=True)
        with self.source(url):
            self.case.run_download(url, expect_exit=True)
        self.assertIn("wrong SHA-256", self.case.out)
        self.assertFalse(self.case.dst.exists())
        self.assertFalse(self.case.dst.with_name(self.case.dst.name + ".done").exists())


if __name__ == "__main__":
    unittest.main()
