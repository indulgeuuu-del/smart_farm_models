from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "release" / "download_public_release.py"


def load_module():
    spec = importlib.util.spec_from_file_location("download_public_release", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, content: bytes, status: int = 200) -> None:
        self._content = io.BytesIO(content)
        self.status = status

    def read(self, size: int = -1) -> bytes:
        return self._content.read(size)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False


class FakeOpener:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.requests = []

    def __call__(self, request, timeout: float):
        self.requests.append((request, timeout))
        return self.response


class DownloadPublicReleaseTest(unittest.TestCase):
    def test_script_exists(self) -> None:
        self.assertTrue(SCRIPT_PATH.is_file())

    def test_parse_assets_rejects_unsafe_name(self) -> None:
        module = load_module()
        payload = {
            "assets": [
                {
                    "name": "../part-001.tar.gz",
                    "browser_download_url": "https://example.invalid/part-001.tar.gz",
                    "size": 1,
                    "digest": "sha256:" + "0" * 64,
                }
            ]
        }

        with self.assertRaises(module.DownloadError):
            module.parse_assets(payload)

    def test_select_assets_filters_unlisted_release_files(self) -> None:
        module = load_module()
        assets = [
            module.ReleaseAsset("part-001.tar.gz", "https://example.invalid/1", 1, "1" * 64),
            module.ReleaseAsset("SHA256SUMS.txt", "https://example.invalid/sums", 1, "2" * 64),
            module.ReleaseAsset("old-upload.png", "https://example.invalid/old", 1, "3" * 64),
        ]

        selected, warnings = module.select_assets(
            assets,
            ("part-001.tar.gz", "SHA256SUMS.txt"),
        )

        self.assertEqual([asset.name for asset in selected], ["part-001.tar.gz", "SHA256SUMS.txt"])
        self.assertEqual(warnings, ["Unlisted release assets ignored: old-upload.png"])

    def test_select_assets_requires_missing_contract_file(self) -> None:
        module = load_module()
        asset = module.ReleaseAsset("part-001.tar.gz", "https://example.invalid/1", 1, "1" * 64)

        with self.assertRaisesRegex(module.DownloadError, "missing contract assets"):
            module.select_assets([asset], ("part-001.tar.gz", "SHA256SUMS.txt"))

    def test_unlisted_download_requires_explicit_opt_in(self) -> None:
        module = load_module()
        assets = [
            module.ReleaseAsset("part-001.tar.gz", "https://example.invalid/1", 1, "1" * 64),
            module.ReleaseAsset("old.png", "https://example.invalid/old", 1, "2" * 64),
        ]

        with self.assertRaisesRegex(module.DownloadError, "selected release contract"):
            module.select_assets(assets, ("part-001.tar.gz",), ["old.png"])

        selected, _ = module.select_assets(assets, ("part-001.tar.gz",), ["old.png"], True)
        self.assertEqual([asset.name for asset in selected], ["old.png"])

    def test_resumes_partial_download_and_verifies_digest(self) -> None:
        module = load_module()
        content = b"hello world"
        asset = module.ReleaseAsset(
            name="part-001.tar.gz",
            url="https://example.invalid/part-001.tar.gz",
            size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        opener = FakeOpener(FakeResponse(b" world", status=206))
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "part-001.tar.gz.part").write_bytes(b"hello")

            result = module.download_asset(asset, root, timeout=1, retries=1, opener=opener)

            self.assertEqual(result, "downloaded")
            self.assertEqual((root / asset.name).read_bytes(), content)
            self.assertFalse((root / f"{asset.name}.part").exists())
            self.assertEqual(opener.requests[0][0].get_header("Range"), "bytes=5-")

    def test_existing_verified_asset_is_not_downloaded_again(self) -> None:
        module = load_module()
        content = b"already complete"
        asset = module.ReleaseAsset(
            name="part-001.tar.gz",
            url="https://example.invalid/part-001.tar.gz",
            size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        opener = FakeOpener(FakeResponse(b"unexpected"))
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / asset.name).write_bytes(content)

            result = module.download_asset(asset, root, timeout=1, retries=1, opener=opener)

            self.assertEqual(result, "skipped")
            self.assertEqual(opener.requests, [])

    def test_completed_partial_is_verified_and_renamed_without_a_network_request(self) -> None:
        module = load_module()
        content = b"complete before rename"
        asset = module.ReleaseAsset(
            name="part-001.tar.gz",
            url="https://example.invalid/part-001.tar.gz",
            size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        opener = FakeOpener(FakeResponse(b"unexpected"))
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / f"{asset.name}.part").write_bytes(content)

            result = module.download_asset(asset, root, timeout=1, retries=1, opener=opener)

            self.assertEqual(result, "downloaded")
            self.assertEqual((root / asset.name).read_bytes(), content)
            self.assertFalse((root / f"{asset.name}.part").exists())
            self.assertEqual(opener.requests, [])

    def test_rejects_partial_larger_than_published_asset(self) -> None:
        module = load_module()
        content = b"expected"
        asset = module.ReleaseAsset(
            name="part-001.tar.gz",
            url="https://example.invalid/part-001.tar.gz",
            size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / f"{asset.name}.part").write_bytes(content + b"extra")

            with self.assertRaisesRegex(module.DownloadError, "exceeds published size"):
                module.download_asset(asset, root, timeout=1, retries=1)


if __name__ == "__main__":
    unittest.main()
