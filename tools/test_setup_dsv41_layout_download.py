"""Opt-in IST-DAS six-shard downloads keep the existing four-shard default and exact file pins."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup  # noqa: E402
from tools import test_setup_dsv41 as fixtures

REVISION = "65d53bc8be65123e255ff02f3ded958b6dbf54fd"


class DownloadLayout(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.InstallOfficial()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        platform = mock.patch.object(sys, "platform", "linux")
        platform.start()
        self.addCleanup(platform.stop)

    def test_six_downloads_five_iq2_xs_shards_and_current_engram(self):
        f = self.fixture
        with mock.patch.object(setup, "verify_sha256", wraps=setup.verify_sha256) as verify:
            rc, output = f.install_ista(model="IST-DAS-2.25bpw", dsv41_layout="six")
        self.assertEqual(rc, 0, output)
        gets = [c for c in f.calls if c[0] == "get"]
        names = [setup.DSV41_ISTA6_SHARD.format(quant="IQ2_XS", i=i) for i in range(1, 6)]
        self.assertEqual([c[2].name for c in gets[:5]], names)
        engram = next(c for c in gets if c[2].name == setup.DSV41_ISTA_ENGRAM)
        for call in gets[:5] + [engram]:
            self.assertIn(f"/resolve/{REVISION}/", call[1])
            self.assertIsNotNone(setup.file_pin(call[1]))
        checked = next(c for c in verify.call_args_list if c.args[0].name == setup.DSV41_ISTA_ENGRAM)
        self.assertEqual(checked.args[1:3], setup.DSV41_ENGRAM_CONVERTED)
        self.assertIn(setup.DSV41_ISTA_FILES[setup.DSV41_ISTA_ENGRAM][1], checked.kwargs["also"])
        pack = [c for c in f.calls if c[0] == "run"][-1][1]
        self.assertEqual(pack[pack.index("--shards") + 1:pack.index("--engram")], names)

    def test_modelscope_option_is_shown_for_six_shard_downloads(self):
        with mock.patch.dict(os.environ, {"STRATA_SOURCE": "modelscope"}):
            rc, output = self.fixture.install_ista(model="IST-DAS-2.25bpw", dsv41_layout="six")
        self.assertEqual(rc, 0, output)
        self.assertIn("Downloading model files from ModelScope", output)
        gets = [c for c in self.fixture.calls if c[0] == "get"]
        wanted = "/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/resolve/master/"
        for call in gets[:5]:
            repo, path = setup.ms_file(call[1])
            self.assertTrue(setup.ms_url(repo, path).endswith(wanted + call[2].name))

    def test_no_option_keeps_four_shard_urls(self):
        f = self.fixture
        rc, output = f.install_ista(model="IST-DAS-2.25bpw")
        self.assertEqual(rc, 0, output)
        gets = [c for c in f.calls if c[0] == "get"]
        old = setup.MULTI_FILE_FAMILIES["deepseek41"]["models"]["IST-DAS-2.25bpw"]["shards"]
        self.assertEqual([c[2].name for c in gets[:4]], old)
        self.assertTrue(all("/resolve/cb02422c900df489cea5833e8a92171224059b3c/" in c[1] for c in gets[:4]))

    def test_selecting_six_leaves_old_shards_and_partial_files_in_place(self):
        f = self.fixture
        folder = f.root / "models" / "deepseek-v4.1-flash-IST-DAS-2.25bpw"
        folder.mkdir(parents=True)
        old = folder / setup.DSV41_ISTA_SHARD.format(quant="IQ2_XS", i=1)
        old.write_bytes(b"old weights")
        partial = folder / (setup.DSV41_ISTA6_SHARD.format(quant="IQ2_XS", i=2) + ".part")
        partial.write_bytes(b"partial download")
        rc, output = f.install_ista(model="IST-DAS-2.25bpw", dsv41_layout="six")
        self.assertEqual(rc, 0, output)
        self.assertEqual(old.read_bytes(), b"old weights")
        self.assertEqual(partial.read_bytes(), b"partial download")
        self.assertEqual([c[2].name for c in f.calls if c[0] == "get"][:5],
                         [setup.DSV41_ISTA6_SHARD.format(quant="IQ2_XS", i=i) for i in range(1, 6)])

    def test_six_is_rejected_for_an_unsloth_model(self):
        rc, output = self.fixture.install(dsv41_layout="six")
        self.assertEqual(rc, 1, output)
        self.assertIn("IST-DAS", output)
        self.assertFalse([c for c in self.fixture.calls if c[0] == "get"])


class RevisionPins(unittest.TestCase):
    def url(self, name, revision=REVISION):
        return f"https://huggingface.co/{setup.DSV41_ISTA_REPO}/resolve/{revision}/{name}"

    def test_all_six_shards_are_pinned_at_the_new_revision(self):
        for name, pin in setup.DSV41_ISTA6_FILES.items():
            with self.subTest(name=name):
                self.assertEqual(setup.file_pin(self.url(name)), pin)
        name = setup.DSV41_ISTA_ENGRAM
        self.assertEqual(setup.file_pin(self.url(name)), setup.DSV41_ENGRAM_CONVERTED)
        self.assertEqual(setup.file_pin(setup.hf(setup.DSV41_ISTA_REPO) + name), setup.DSV41_ISTA_FILES[name])

    def test_modelscope_keeps_the_selected_revisions_pin(self):
        url = self.url(setup.DSV41_ISTA_ENGRAM)
        # Both Engram revisions have the same size but different hashes. A tiny completed file exercises
        # the real source mapping and pin selection without any network or model data.
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(setup, "model_source", return_value="modelscope"), \
                mock.patch.object(setup, "reachable", return_value=True), \
                mock.patch.object(setup, "_urlopen", return_value=SimpleNamespace(headers={"Content-Length": "8"})) as opened, \
                mock.patch.object(setup, "ms_meta", return_value=None), \
                mock.patch.object(setup, "dl_connections", return_value=1), \
                mock.patch.object(setup, "HF_REVISION_PINS", create=True,
                                  new={(setup.DSV41_ISTA_REPO, REVISION): {setup.DSV41_ISTA_ENGRAM: (8, "a" * 64)}}), \
                mock.patch.dict(setup.DSV41_ISTA_FILES, {setup.DSV41_ISTA_ENGRAM: (8, "b" * 64)}), \
                mock.patch.object(setup, "verify_sha256") as verify:
            dst = Path(folder) / "engram.gguf"
            dst.write_bytes(b"12345678")
            setup.download(url, dst)
        self.assertEqual(verify.call_args.args[1:3], (8, "a" * 64))
        self.assertEqual(opened.call_args.args[0].full_url,
                         setup.ms_url(setup.DSV41_ISTA_REPO, setup.DSV41_ISTA_ENGRAM))


if __name__ == "__main__":
    unittest.main()
