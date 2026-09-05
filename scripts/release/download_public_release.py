"""Download, resume, and verify the assets of one public GitHub Release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_REPOSITORY = "indulgeuuu-del/smart_farm_models"
DEFAULT_API_BASE = "https://api.github.com"
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    url: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ReleaseContract:
    tag: str
    expected_assets: tuple[str, ...]


class DownloadError(RuntimeError):
    """A release asset could not be downloaded or verified safely."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", required=True, help="Destination directory for release assets.")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY, help="GitHub owner/repository name.")
    parser.add_argument("--tag", help="Release tag. Defaults to RELEASE_MANIFEST.json when available.")
    parser.add_argument(
        "--asset",
        action="append",
        help="Download only this asset name. Repeat the option to select multiple assets.",
    )
    parser.add_argument(
        "--release-json",
        help="Use a saved GitHub release API response instead of requesting the GitHub API.",
    )
    parser.add_argument(
        "--allow-unlisted-assets",
        action="store_true",
        help="Also download assets not listed in RELEASE_MANIFEST.json (explicit opt-in).",
    )
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=float, default=60.0, help="Network timeout in seconds per request.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count for transient network failures.")
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_asset_name(value: object) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise DownloadError(f"Invalid release asset name: {value!r}")
    if "/" in value or "\\" in value or Path(value).name != value:
        raise DownloadError(f"Release asset name must not contain a path: {value!r}")
    return value


def parse_assets(payload: object) -> list[ReleaseAsset]:
    if not isinstance(payload, dict) or not isinstance(payload.get("assets"), list):
        raise DownloadError("Release metadata must contain an assets list.")

    assets: list[ReleaseAsset] = []
    names: set[str] = set()
    for item in payload["assets"]:
        if not isinstance(item, dict):
            raise DownloadError("Release asset metadata must be an object.")
        name = safe_asset_name(item.get("name"))
        if name in names:
            raise DownloadError(f"Release metadata contains a duplicate asset name: {name}")
        names.add(name)
        url = item.get("browser_download_url")
        if not isinstance(url, str) or urlparse(url).scheme not in {"http", "https"}:
            raise DownloadError(f"Release asset has no safe browser download URL: {name}")
        try:
            size = int(item.get("size"))
        except (TypeError, ValueError) as exc:
            raise DownloadError(f"Release asset has no valid size: {name}") from exc
        if size < 0:
            raise DownloadError(f"Release asset has a negative size: {name}")
        digest = item.get("digest")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise DownloadError(f"Release asset has no SHA-256 digest: {name}")
        sha256 = digest.removeprefix("sha256:").lower()
        if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
            raise DownloadError(f"Release asset has an invalid SHA-256 digest: {name}")
        assets.append(ReleaseAsset(name=name, url=url, size=size, sha256=sha256))
    return assets


def read_release_contract(manifest_path: Path | None = None) -> ReleaseContract:
    if manifest_path is None:
        manifest_path = Path(__file__).resolve().parents[2] / "RELEASE_MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tag = manifest["release"]["tag_name"]
        expected = manifest["archive"]["expected_assets"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DownloadError(f"Cannot read release contract from {manifest_path}: {exc}") from exc
    if not isinstance(tag, str) or not tag:
        raise DownloadError("RELEASE_MANIFEST.json has no usable release.tag_name.")
    if not isinstance(expected, list) or not expected:
        raise DownloadError("RELEASE_MANIFEST.json has no usable archive.expected_assets list.")
    names: list[str] = []
    seen: set[str] = set()
    for value in expected:
        name = safe_asset_name(value)
        if name in seen:
            raise DownloadError(f"Release contract contains a duplicate asset name: {name}")
        seen.add(name)
        names.append(name)
    return ReleaseContract(tag=tag, expected_assets=tuple(names))


def default_tag() -> str:
    return read_release_contract().tag


def select_assets(
    assets: list[ReleaseAsset],
    expected_assets: tuple[str, ...] | None,
    requested_names: list[str] | None = None,
    allow_unlisted: bool = False,
) -> tuple[list[ReleaseAsset], list[str]]:
    """Select assets and report release files outside the repository contract."""
    by_name = {asset.name: asset for asset in assets}
    warnings: list[str] = []
    if expected_assets is None:
        candidates = list(assets)
    else:
        missing_contract = [name for name in expected_assets if name not in by_name]
        if missing_contract:
            raise DownloadError(
                "Release is missing contract assets: " + ", ".join(missing_contract)
            )
        unlisted = sorted(set(by_name) - set(expected_assets))
        if unlisted:
            warnings.append("Unlisted release assets ignored: " + ", ".join(unlisted))
        candidates = [by_name[name] for name in expected_assets]

    requested = None
    if requested_names:
        requested = {safe_asset_name(name) for name in requested_names}
        known = {asset.name for asset in candidates}
        if allow_unlisted and expected_assets is not None:
            known = set(by_name)
        missing = sorted(requested - known)
        if missing:
            raise DownloadError("Requested assets do not exist in the selected release contract: " + ", ".join(missing))
        if expected_assets is not None and not allow_unlisted:
            candidates = [asset for asset in candidates if asset.name in requested]
        else:
            candidates = [asset for asset in assets if asset.name in requested]
    elif allow_unlisted and expected_assets is not None:
        candidates = list(assets)
    return candidates, warnings


def read_json_file(path: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DownloadError(f"Cannot read release metadata file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DownloadError(f"Release metadata file is not a JSON object: {path}")
    return payload


def read_release_metadata(repository: str, tag: str, api_base: str, timeout: float) -> dict[str, Any]:
    if repository.count("/") != 1 or any(not part for part in repository.split("/")):
        raise DownloadError(f"Repository must have the form owner/name: {repository!r}")
    from urllib.parse import quote

    url = f"{api_base.rstrip('/')}/repos/{repository}/releases/tags/{quote(tag, safe='')}"
    request = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "smart-farm-models-release"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DownloadError(f"Cannot read GitHub release metadata for {repository}@{tag}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DownloadError("GitHub release metadata is not an object.")
    return payload


def response_status(response: object) -> int:
    status = getattr(response, "status", None)
    if isinstance(status, int):
        return status
    getcode = getattr(response, "getcode", None)
    if callable(getcode):
        value = getcode()
        if isinstance(value, int):
            return value
    raise DownloadError("Download response did not provide an HTTP status code.")


def write_response(response: Any, target: Path, start_at: int) -> None:
    status = response_status(response)
    if status not in {200, 206}:
        raise DownloadError(f"Unexpected HTTP status {status} while downloading {target.name}.")
    append = start_at > 0 and status == 206
    mode = "ab" if append else "wb"
    with target.open(mode) as handle:
        for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
            handle.write(chunk)


def download_asset(
    asset: ReleaseAsset,
    release_dir: Path,
    timeout: float,
    retries: int,
    opener: Callable[..., Any] = urlopen,
) -> str:
    release_dir.mkdir(parents=True, exist_ok=True)
    final_path = release_dir / asset.name
    partial_path = release_dir / f"{asset.name}.part"
    if final_path.exists():
        if not final_path.is_file():
            raise DownloadError(f"Existing release target is not a file: {final_path}")
        if final_path.stat().st_size == asset.size and sha256_file(final_path) == asset.sha256:
            return "skipped"
        raise DownloadError(f"Existing release asset does not match published metadata: {final_path}")

    if partial_path.exists() and not partial_path.is_file():
        raise DownloadError(f"Existing partial release target is not a file: {partial_path}")
    if partial_path.exists():
        partial_size = partial_path.stat().st_size
        if partial_size > asset.size:
            raise DownloadError(
                f"Partial release asset exceeds published size for {asset.name}: "
                f"expected {asset.size}, got {partial_size}."
            )
        if partial_size == asset.size:
            actual = sha256_file(partial_path)
            if actual != asset.sha256:
                raise DownloadError(f"SHA-256 mismatch for {asset.name}: expected {asset.sha256}, got {actual}.")
            os.replace(partial_path, final_path)
            return "downloaded"

    attempts = max(1, retries)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        offset = partial_path.stat().st_size if partial_path.exists() else 0
        headers = {"Accept": "application/octet-stream", "User-Agent": "smart-farm-models-release"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(asset.url, headers=headers)
        try:
            with opener(request, timeout=timeout) as response:
                write_response(response, partial_path, offset)
            if partial_path.stat().st_size != asset.size:
                raise DownloadError(
                    f"Downloaded size mismatch for {asset.name}: expected {asset.size}, got {partial_path.stat().st_size}."
                )
            actual = sha256_file(partial_path)
            if actual != asset.sha256:
                raise DownloadError(f"SHA-256 mismatch for {asset.name}: expected {asset.sha256}, got {actual}.")
            os.replace(partial_path, final_path)
            return "downloaded"
        except (HTTPError, URLError, OSError, DownloadError) as exc:
            last_error = exc
            if isinstance(exc, DownloadError) and (
                "size mismatch" in str(exc) or "SHA-256 mismatch" in str(exc) or "Unexpected HTTP" in str(exc)
            ):
                break
            if attempt < attempts:
                time.sleep(min(2.0**(attempt - 1), 8.0))
    raise DownloadError(f"Failed to download {asset.name} after {attempts} attempt(s): {last_error}")


def main() -> int:
    args = build_parser().parse_args()
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive.")
    if args.retries < 1:
        raise SystemExit("--retries must be at least 1.")

    try:
        contract = read_release_contract() if args.tag is None else None
        tag = args.tag or contract.tag
        payload = read_json_file(args.release_json) if args.release_json else read_release_metadata(
            args.repository, tag, args.api_base, args.timeout
        )
        assets = parse_assets(payload)
        if not assets:
            raise DownloadError("Release metadata contains no downloadable assets.")
        selected, warnings = select_assets(
            assets,
            contract.expected_assets if contract is not None else None,
            args.asset,
            args.allow_unlisted_assets,
        )
        if not selected:
            raise DownloadError("No release assets selected for download.")
        release_dir = Path(args.release_dir).resolve()
        for asset in selected:
            result = download_asset(asset, release_dir, args.timeout, args.retries)
            print(f"ASSET={asset.name} STATUS={result} BYTES={asset.size}")
        for warning in warnings:
            print(f"WARNING={warning}")
    except DownloadError as exc:
        print(f"STATUS=FAIL\nISSUE: {exc}")
        return 1
    print(f"STATUS=OK\nTAG={tag}\nASSETS={len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
