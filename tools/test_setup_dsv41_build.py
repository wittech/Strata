"""DeepSeek --build compiles its own CUDA target before preparing model files; no GPU/downloads needed."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup  # noqa: E402
from tools import test_setup_dsv41 as fixtures


class CudaBuildEntry(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.Install()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def install(self, **kw):
        f = self.fixture
        cards = [{"index": 0, "name": "NVIDIA A800", "arch": "80", "vram_gb": 80.0, "vendor": "nvidia"},
                 {"index": 1, "name": "NVIDIA H200", "arch": "90", "vram_gb": 141.0, "vendor": "nvidia"}]
        options = fixtures.args(model="UD-Q8_K_XL", vision="no", build=True, models_dir=str(f.root / "models"), **kw)
        with mock.patch.object(sys, "platform", "linux"):
            return fixtures.run(setup.install_dsv41, options, "deepseek41", cards[0], False, [], 13, [], 8080, 512.0, None, cards)

    def test_build_compiles_for_selected_cards_and_saves_library_paths(self):
        f = self.fixture
        with mock.patch.object(setup, "build_dsv41_engine", create=True,
                               return_value=(f.exe, ["/cuda/lib64"])) as build, \
                mock.patch.object(setup, "get_dsv41_engine") as fetch, \
                mock.patch.object(setup, "github_reachable") as probe:
            rc, output = self.install()
        self.assertEqual(rc, 0, output)
        build.assert_called_once()
        self.assertEqual(build.call_args.args[0]["archs"], [80, 90])
        self.assertEqual(build.call_args.kwargs["toolkit"], 13)
        fetch.assert_not_called()
        probe.assert_not_called()
        self.assertEqual(f.cfg["exe"], str(f.exe))
        self.assertEqual(f.cfg["lib_dirs"], ["/cuda/lib64"])

    def test_build_runs_even_with_an_installed_engine(self):
        f = self.fixture
        f.exe.parent.mkdir(parents=True)
        f.exe.write_text("old engine")
        with mock.patch.object(setup, "build_dsv41_engine", create=True, return_value=(f.exe, [])) as build:
            rc, output = self.install()
        self.assertEqual(rc, 0, output)
        build.assert_called_once()


class CudaBuildTarget(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.nvcc = self.root / "cuda" / "bin" / "nvcc"
        self.nvcc.parent.mkdir(parents=True)
        (self.nvcc.parent.parent / "lib64").mkdir()
        self.calls = []
        self.produce_binary = True
        for patch in (mock.patch.object(setup, "ROOT", self.root),
                      mock.patch.object(setup, "WIN", False),
                      mock.patch.object(setup.acquisition, "require"),
                      mock.patch.object(setup, "install_build_tools", return_value=(str(self.nvcc), None)),
                      mock.patch.object(setup, "get_llama_cpp", return_value=self.root / "llama"),
                      mock.patch.object(setup, "cpu_info", return_value=("test CPU", True)),
                      mock.patch.object(setup, "cpu_floor", return_value=""),
                      mock.patch.object(setup, "cmake_build", side_effect=self.compile)):
            patch.start()
            self.addCleanup(patch.stop)

    def compile(self, src, folder, target, defs, vcvars, bat):
        self.calls.append((folder, target, defs))
        if self.produce_binary:
            folder.mkdir(parents=True, exist_ok=True)
            binary = folder / "strata-dsv41"
            binary.write_text("compiled DeepSeek")
            binary.chmod(0o755)

    def test_build_uses_own_target_and_preserves_qwen_build_record(self):
        eng = self.root / "engine"
        eng.mkdir()
        stamp = eng / "BUILD.json"
        stamp.write_text(json.dumps({"source": "prebuilt", "version": "qwen"}))
        before = stamp.read_bytes()
        exe, dirs = setup.build_dsv41_engine({"arch": "80", "archs": [90, 80]}, True, toolkit=13)
        self.assertEqual(exe, eng / "strata-dsv41")
        self.assertEqual(exe.read_text(), "compiled DeepSeek")
        self.assertEqual(stamp.read_bytes(), before)
        folder, target, defs = self.calls[0]
        self.assertEqual(folder, self.root / "build-dsv41")
        self.assertEqual(target, "strata-dsv41")
        for definition in ("-DSTRATA_BUILD_DSV41=ON", "-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_NATIVE_EXPERTS=ON",
                           "-DCMAKE_CUDA_ARCHITECTURES=80;90", f"-DCMAKE_CUDA_COMPILER={self.nvcc}"):
            self.assertIn(definition, defs)
        self.assertIn(str(self.nvcc.parent.parent / "lib64"), dirs)

    def test_cuda12_gets_its_own_build_folder(self):
        setup.build_dsv41_engine({"arch": "80"}, True, toolkit=12)
        folder, _, defs = self.calls[0]
        self.assertEqual(folder, self.root / "build-dsv41-cuda12")
        self.assertIn("-DSTRATA_EXPERIMENTAL_SM60=ON", defs)

    def test_missing_build_output_stops_before_installing(self):
        self.produce_binary = False
        with self.assertRaises(SystemExit):
            setup.build_dsv41_engine({"arch": "90"}, True, toolkit=13)
        self.assertFalse((self.root / "engine" / "strata-dsv41").exists())


if __name__ == "__main__":
    unittest.main()
