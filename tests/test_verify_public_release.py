from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "release" / "verify_public_release.py"


def load_module():
    spec = importlib.util.spec_from_file_location("verify_public_release", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_release(root: Path, unsafe_member: bool = False) -> None:
    source_files = {
        "datasets/sample/images/a.txt": b"alpha\n",
        "outputs/run/metrics.json": b'{"ok": true}\n',
    }
    archive_name = "part-001.tar.gz"
    archive_path = root / archive_name
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, data in source_files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        if unsafe_member:
            info = tarfile.TarInfo("../outside.txt")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))

    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    (root / "SHA256SUMS.txt").write_text(f"{digest} *{archive_name}\n", encoding="utf-8")
    with (root / "source_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["path", "bytes"])
        writer.writerows((name, len(data)) for name, data in source_files.items())
    (root / "part_plan.csv").write_text(
        "Part,Files,Bytes,GiB,Archive,ArchiveBytes,ArchiveSHA256\n"
        f"part-001,2,{sum(map(len, source_files.values()))},0.0,{archive_name},{archive_path.stat().st_size},{digest}\n",
        encoding="utf-8",
    )
    (root / "scope.json").write_text(
        json.dumps(
            {
                "Repository": "smart_farm_models",
                "Files": 2,
                "Bytes": sum(map(len, source_files.values())),
                "Parts": 1,
            }
        ),
        encoding="utf-8",
    )
    (root / "RELEASE_NOTES.md").write_text("release\n", encoding="utf-8")


class VerifyPublicReleaseTest(unittest.TestCase):
    def test_valid_release_and_extraction_pass(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "release"
            root.mkdir()
            write_release(root)
            report = module.verify_release(root, Path(tmpdir) / "restored")
            self.assertTrue(report.ok, report.issues)
            self.assertTrue((Path(tmpdir) / "restored" / "datasets/sample/images/a.txt").is_file())

    def test_unsafe_archive_member_is_rejected(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "release"
            root.mkdir()
            write_release(root, unsafe_member=True)
            report = module.verify_release(root)
            self.assertFalse(report.ok)
            self.assertTrue(any("Unsafe archive member" in issue for issue in report.issues))

    def test_manifest_rejects_absolute_path(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "source_manifest.tsv"
            manifest.write_text("path\tbytes\nC:/secret.txt\t1\n", encoding="utf-8")
            parsed, issues = module.read_manifest(manifest)
            self.assertEqual(parsed, {})
            self.assertTrue(any("drive-qualified" in issue for issue in issues))

    def test_duplicate_archive_members_across_parts_are_rejected(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "release"
            root.mkdir()
            write_release(root)
            second = root / "part-002.tar.gz"
            with tarfile.open(second, "w:gz") as archive:
                data = b"duplicate\n"
                info = tarfile.TarInfo("datasets/sample/images/a.txt")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            digest = hashlib.sha256(second.read_bytes()).hexdigest()
            (root / "SHA256SUMS.txt").write_text(
                (root / "SHA256SUMS.txt").read_text(encoding="utf-8")
                + f"{digest} *{second.name}\n",
                encoding="utf-8",
            )
            with (root / "part_plan.csv").open("a", encoding="utf-8") as handle:
                handle.write(f"part-002,1,{len(data)},0.0,{second.name},{second.stat().st_size},{digest}\n")
            report = module.verify_release(root)
            self.assertFalse(report.ok)
            self.assertTrue(any("more than one archive" in issue for issue in report.issues))

    def test_archive_part_numbers_must_be_continuous(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "release"
            root.mkdir()
            write_release(root)

            second = root / "part-003.tar.gz"
            data = b"gap\n"
            with tarfile.open(second, "w:gz") as archive:
                info = tarfile.TarInfo("outputs/run/metrics-2.json")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))

            digest = hashlib.sha256(second.read_bytes()).hexdigest()
            (root / "SHA256SUMS.txt").write_text(
                (root / "SHA256SUMS.txt").read_text(encoding="utf-8")
                + f"{digest} *{second.name}\n",
                encoding="utf-8",
            )
            with (root / "source_manifest.tsv").open("a", encoding="utf-8", newline="") as handle:
                handle.write(f"outputs/run/metrics-2.json\t{len(data)}\n")
            with (root / "part_plan.csv").open("a", encoding="utf-8") as handle:
                handle.write(f"part-003,1,{len(data)},0.0,{second.name},{second.stat().st_size},{digest}\n")
            scope = json.loads((root / "scope.json").read_text(encoding="utf-8"))
            scope["Files"] = 3
            scope["Bytes"] += len(data)
            scope["Parts"] = 2
            (root / "scope.json").write_text(json.dumps(scope), encoding="utf-8")

            report = module.verify_release(root)

            self.assertFalse(report.ok)
            self.assertTrue(any("continuous from 001" in issue for issue in report.issues))


if __name__ == "__main__":
    unittest.main()
