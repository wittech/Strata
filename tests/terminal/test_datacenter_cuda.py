"""CUDA build recipes cover A100/A800 (sm_80) and H100/H200 (sm_90).

Checks recipe declarations against the real package selector. No GPU or compiler is used.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from strata_control import engine_component as ec, feed

ROOT = Path(__file__).resolve().parents[2]


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def manifest_archs(spec):
    archs = []
    for item in spec.split(";"):
        number, _, kind = item.partition("-")
        if kind != "virtual":
            archs.append(f"sm_{number}")
        if kind != "real":
            archs.append(f"compute_{number}")
    return tuple(archs)


def release_archs(spec):
    """Engine, reader and Docker recipes emit SASS for all cards and PTX only for the last."""
    return tuple(f"sm_{n}" for n in spec.split(";")) + (f"compute_{spec.split(';')[-1]}",)


class DatacenterCuda(unittest.TestCase):
    def assert_cards_select_package(self, archs, platform, cards):
        package = feed.EnginePackage(
            version="0.1.55", platform=platform, backend="cuda", variant="cuda13", archs=tuple(archs),
            min_driver={"nvidia": "580.65"}, min_runtime=None, build_id="test", url="u", format="tar.gz",
            size=1, sha256="a" * 64, manifest_sha256="b" * 64, installed_bytes=1,
        )
        for card in cards:
            with self.subTest(platform=platform, card=card):
                hw = {"platform": platform, "backend": "cuda", "cuda_archs": [card], "nvidia_driver": "590.10"}
                chosen, reasons = feed.select_engine([package], hw)
                self.assertIs(chosen, package, reasons)
                self.assertTrue(ec.arch_native(archs, [card]), "ship the card's own machine code")

    def test_linux_main_engine_and_reader_cover_datacenter_cards(self):
        script = read("packaging/build_cuda_linux.sh")
        spec = re.search(r"cuda13\) archs=\$\{ARCHS:-([^}]+)\}", script).group(1)
        self.assert_cards_select_package(release_archs(spec), "linux-x86_64", ("sm_80", "sm_90"))
        self.assertIn('"-DCMAKE_CUDA_ARCHITECTURES=${real};${last}-virtual"', script)
        self.assertIn('--arch "compute_$last"', script)

    def test_deepseek_linux_windows_and_cmake_cover_the_same_cards(self):
        linux = re.search(r"^ARCH=\$\{ARCH:-([^}]+)\}", read("packaging/build_dsv41_cuda.sh"), re.M).group(1)
        windows = re.search(r'set "DSV41_ARCH=([^"]+)"', read("packaging/build_dsv41_windows.bat")).group(1)
        cmake = re.search(r'if\(STRATA_BUILD_DSV41\).*?set\(CMAKE_CUDA_ARCHITECTURES "([^"]+)"',
                          read("CMakeLists.txt"), re.S).group(1)
        self.assertEqual(linux, windows)
        self.assertEqual(linux, cmake)
        for platform in ("linux-x86_64", "windows-x86_64"):
            self.assert_cards_select_package(manifest_archs(linux), platform,
                                             ("sm_80", "sm_86", "sm_89", "sm_90", "sm_120"))
        self.assertTrue(ec.arch_covers(manifest_archs(linux), ["sm_121"]))
        self.assertFalse(ec.arch_covers(manifest_archs(linux), ["sm_75"]), "do not lower DeepSeek's build floor")

    def test_windows_reader_covers_the_new_engine_architectures(self):
        script = read("packaging/build_vision_windows.bat")
        spec = re.search(r'else if /i "%VARIANT%"=="cuda13" \(\s*set "ARCHS=([^"]+)"', script).group(1)
        self.assert_cards_select_package(release_archs(spec), "windows-x86_64", ("sm_80", "sm_90"))

    def test_docker_source_build_covers_datacenter_cards(self):
        spec = re.search(r'ARG CUDA_ARCHITECTURES="([^"]+)"', read("Dockerfile")).group(1)
        self.assert_cards_select_package(release_archs(spec), "linux-x86_64", ("sm_80", "sm_90"))

    def test_old_consumer_only_package_is_still_rejected(self):
        old = ("sm_86", "sm_89", "sm_120", "compute_120")
        for card in ("sm_80", "sm_90"):
            self.assertFalse(ec.arch_covers(old, [card]), "metadata must not pretend missing kernels exist")


if __name__ == "__main__":
    unittest.main()
