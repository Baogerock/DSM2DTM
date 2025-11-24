"""Utilities to download and split the DSM2DTM dataset patches.

The script downloads the original 2000x2000 DSM/DTM tiles from the URLs
listed in the CSV files under ``dataset/DSMs`` and ``dataset/DTMs`` and
creates 256x256 patches according to the provided ``train``, ``val`` and
``test`` text files. The resulting directory structure is::

    <output>/
        raw/<modality>/<canton>/<source>.tif
        patches/<split>/<modality>/<canton>/<source>_<x>_<y>.tif

The download and extraction pipeline follows the dataset description in the
README: Zürich, St. Gallen and Vaud contain ``train`` and ``val`` splits,
while Fribourg is used only for ``test``.

Examples
--------
Download everything to ``data`` using eight threads::

    python prepare_dataset.py --output-root data --workers 8

Only create patches (assumes raw downloads already exist)::

    python prepare_dataset.py --output-root data --skip-download

"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urlparse

import requests
from PIL import Image

# Size of the cropped patch used for training, validation and testing
PATCH_SIZE = 256

# Default location of the dataset metadata
DEFAULT_DATASET_ROOT = Path("dataset")

# Map of modality folder to prefix used in split files
MODALITIES = {
    "DSMs": "dsm",
    "DTMs": "dgm",
}

# Configure a module-level logger
LOGGER = logging.getLogger("prepare_dataset")


class DownloadError(RuntimeError):
    """Raised when a download fails."""


class PatchExtractionError(RuntimeError):
    """Raised when a patch cannot be extracted."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the folder that contains the modality sub-folders (DSMs/DTMs).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("prepared_dataset"),
        help="Root directory where downloads and patches will be written.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume raw images are already present and only extract patches.",
    )
    return parser.parse_args()


def read_urls(csv_path: Path) -> List[str]:
    """Return a list of URLs from a CSV file.

    The CSVs in this repository consist of a single column with one URL per
    line, therefore we treat each row as a single entry.
    """

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        urls = [row[0].strip() for row in reader if row]
    LOGGER.debug("Loaded %d URLs from %s", len(urls), csv_path)
    return urls


def split_from_filename(path: Path) -> str:
    """Infer dataset split from the metadata filename."""

    name = path.name.lower()
    if "train" in name:
        return "train"
    if "val" in name:
        return "val"
    if "test" in name:
        return "test"
    raise ValueError(f"Cannot determine split from file name: {path}")


def parse_split_file(path: Path) -> List[Tuple[str, int, int]]:
    """Parse a split description file.

    Each line follows the pattern ``<filename>,<x>,<y>`` where ``x`` and ``y``
    denote the upper-left pixel coordinates of the 256×256 patch.
    """

    entries: List[Tuple[str, int, int]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Malformed line in {path}: {line}")
            filename, x_str, y_str = parts
            entries.append((filename, int(x_str), int(y_str)))
    LOGGER.debug("Parsed %d patch definitions from %s", len(entries), path)
    return entries


def derive_target_path(url: str, raw_dir: Path) -> Path:
    """Build the destination file path for a download URL."""

    url_path = Path(urlparse(url).path)
    filename = url_path.name
    return raw_dir / filename


def download_one(url: str, destination: Path) -> Path:
    """Download a single file if it does not already exist."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        LOGGER.debug("Skipping existing file %s", destination)
        return destination

    session = requests.Session()
    try:
        with session.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with destination.open("wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except requests.RequestException as exc:
        destination.unlink(missing_ok=True)
        raise DownloadError(f"Failed to download {url}") from exc

    LOGGER.info("Downloaded %s", destination)
    return destination


def download_all(urls: Sequence[str], raw_dir: Path, workers: int) -> List[Path]:
    """Download every URL using a thread pool."""

    tasks: Dict[concurrent.futures.Future[Path], str] = {}
    results: List[Path] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for url in urls:
            destination = derive_target_path(url, raw_dir)
            future = executor.submit(download_one, url, destination)
            tasks[future] = url

        for future in concurrent.futures.as_completed(tasks):
            url = tasks[future]
            try:
                path = future.result()
                results.append(path)
            except Exception as exc:  # noqa: BLE001
                raise DownloadError(f"Download failed for {url}") from exc
    return results


def extract_patches(
    split_entries: Iterable[Tuple[str, int, int]],
    raw_dir: Path,
    output_dir: Path,
) -> None:
    """Extract patches for a given split and canton.

    Parameters
    ----------
    split_entries:
        Iterable of ``(filename, x, y)`` tuples.
    raw_dir:
        Directory where the full-size GeoTIFF images reside.
    output_dir:
        Destination directory for the extracted patches.
    """

    patches_by_file: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for filename, x, y in split_entries:
        patches_by_file[filename].append((x, y))

    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, coordinates in patches_by_file.items():
        source_path = raw_dir / filename
        if not source_path.exists():
            raise PatchExtractionError(f"Missing source image: {source_path}")

        with Image.open(source_path) as image:
            for x, y in coordinates:
                box = (x, y, x + PATCH_SIZE, y + PATCH_SIZE)
                patch = image.crop(box)
                patch_name = f"{source_path.stem}_{x}_{y}{source_path.suffix}"
                patch_path = output_dir / patch_name
                patch.save(patch_path)
                LOGGER.debug("Saved patch %s", patch_path)
        LOGGER.info("Extracted %d patches from %s", len(coordinates), source_path)


def process_modality(
    modality_root: Path,
    output_root: Path,
    workers: int,
    skip_download: bool,
) -> None:
    """Download and extract patches for a single modality (DSM or DTM)."""

    modality_name = modality_root.name
    prefix = MODALITIES[modality_name]
    LOGGER.info("Processing modality %s", modality_name)

    for canton_dir in sorted(p for p in modality_root.iterdir() if p.is_dir()):
        canton_name = canton_dir.name
        LOGGER.info("\n==> Canton: %s", canton_name)

        raw_dir = output_root / "raw" / modality_name / canton_name
        patch_root = output_root / "patches" / modality_name / canton_name

        csv_files = sorted(canton_dir.glob("*.csv"))
        if not csv_files:
            LOGGER.warning("No CSV files found for %s", canton_dir)
            continue

        urls: List[str] = []
        for csv_file in csv_files:
            urls.extend(read_urls(csv_file))

        if not skip_download:
            download_all(urls, raw_dir, workers)
        else:
            LOGGER.info("Skipping download step for %s", canton_name)

        split_files = sorted(canton_dir.glob(f"{prefix}_256x256_*.txt"))
        if not split_files:
            LOGGER.warning("No split files found for %s", canton_dir)
            continue

        for split_file in split_files:
            split = split_from_filename(split_file)
            split_dir = patch_root / split
            entries = parse_split_file(split_file)
            extract_patches(entries, raw_dir, split_dir)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    )

    for modality, prefix in MODALITIES.items():
        modality_root = args.dataset_root / modality
        if not modality_root.exists():
            LOGGER.warning("Skipping missing modality folder %s", modality_root)
            continue
        process_modality(modality_root, args.output_root, args.workers, args.skip_download)


if __name__ == "__main__":
    main()
