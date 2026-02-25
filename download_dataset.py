"""
INCLUDE Dataset Downloader for ISL Gesture Recognition.

Downloads a curated subset of the INCLUDE dataset from Zenodo,
organized by word categories. The full dataset is ~57 GB;
this script downloads only selected categories to keep it manageable.

Usage:
    python download_dataset.py                    # Download default subset
    python download_dataset.py --all              # Download entire dataset (~57 GB)
    python download_dataset.py --categories Greetings Animals Colours
    python download_dataset.py --list             # List available categories
"""

import argparse
import os
import sys
import ssl
import zipfile
import urllib.request

# ─── SSL fix for macOS Python installations ───────────────────────────────────
try:
    _ssl_context = ssl.create_default_context()
    urllib.request.urlopen("https://zenodo.org", context=_ssl_context, timeout=5)
except Exception:
    _ssl_context = ssl._create_unverified_context()
    ssl._create_default_https_context = ssl._create_unverified_context

# ─── INCLUDE Dataset on Zenodo ────────────────────────────────────────────────
ZENODO_BASE = "https://zenodo.org/records/4010759/files"

# Actual zip file names on Zenodo (verified from the record page)
DATASET_CATEGORIES = {
    "Adjectives": [
        "Adjectives_1of8.zip", "Adjectives_2of8.zip", "Adjectives_3of8.zip",
        "Adjectives_4of8.zip", "Adjectives_5of8.zip", "Adjectives_6of8.zip",
        "Adjectives_7of8.zip", "Adjectives_8of8.zip",
    ],
    "Animals": ["Animals_1of2.zip", "Animals_2of2.zip"],
    "Clothes": ["Clothes_1of2.zip", "Clothes_2of2.zip"],
    "Colours": ["Colours_1of2.zip", "Colours_2of2.zip"],
    "Days_and_Time": [
        "Days_and_Time_1of3.zip", "Days_and_Time_2of3.zip",
        "Days_and_Time_3of3.zip",
    ],
    "Electronics": ["Electronics_1of2.zip", "Electronics_2of2.zip"],
    "Greetings": ["Greetings_1of2.zip", "Greetings_2of2.zip"],
    "Home": [
        "Home_1of4.zip", "Home_2of4.zip",
        "Home_3of4.zip", "Home_4of4.zip",
    ],
    "Jobs": ["Jobs_1of2.zip", "Jobs_2of2.zip"],
    "Means_of_Transportation": [
        "Means_of_Transportation_1of2.zip",
        "Means_of_Transportation_2of2.zip",
    ],
    "People": [
        "People_1of5.zip", "People_2of5.zip", "People_3of5.zip",
        "People_4of5.zip", "People_5of5.zip",
    ],
    "Places": [
        "Places_1of4.zip", "Places_2of4.zip",
        "Places_3of4.zip", "Places_4of4.zip",
    ],
    "Pronouns": ["Pronouns_1of2.zip", "Pronouns_2of2.zip"],
    "Seasons": ["Seasons_1of1.zip"],
    "Society": ["Society_1of3.zip", "Society_2of3.zip", "Society_3of3.zip"],
}

# Default: small, useful categories for a demo (~3-5 GB)
DEFAULT_CATEGORIES = [
    "Greetings",      # Hello, Thank You, Sorry, Good Morning, etc.
    "Pronouns",       # I, You, He, She, We, They, etc.
    "Colours",        # Red, Blue, Green, etc.
    "Animals",        # Cat, Dog, Bird, etc.
    "Seasons",        # Summer, Winter, Rain, etc.
]


def download_file(url: str, dest_path: str):
    """Download a file with progress display."""
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(
                f"\r    {mb_down:.1f}/{mb_total:.1f} MB ({pct:.0f}%)")
        else:
            mb_down = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r    {mb_down:.1f} MB downloaded")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, progress_hook)
    print()


def extract_zip(zip_path: str, extract_dir: str):
    """Extract a zip file and remove it afterwards."""
    print(f"    Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
    os.remove(zip_path)


def download_category(category: str, download_dir: str):
    """Download and extract all zip parts for a category."""
    if category not in DATASET_CATEGORIES:
        print(f"  [ERROR] Unknown category: {category}")
        print(f"  Available: {', '.join(DATASET_CATEGORIES.keys())}")
        return False

    zip_files = DATASET_CATEGORIES[category]
    print(f"\n  Downloading category: {category} ({len(zip_files)} parts)")

    for zip_name in zip_files:
        url = f"{ZENODO_BASE}/{zip_name}?download=1"
        zip_path = os.path.join(download_dir, zip_name)

        if os.path.exists(zip_path):
            print(f"    {zip_name} already exists, skipping download.")
        else:
            print(f"    Downloading {zip_name}...")
            try:
                download_file(url, zip_path)
            except Exception as e:
                print(f"    [ERROR] Failed to download {zip_name}: {e}")
                return False

        try:
            extract_zip(zip_path, download_dir)
        except Exception as e:
            print(f"    [ERROR] Failed to extract {zip_name}: {e}")
            return False

    return True


def list_categories():
    """Print available categories and their zip file counts."""
    print("\nAvailable INCLUDE Dataset Categories:")
    print("=" * 55)
    for name, zips in sorted(DATASET_CATEGORIES.items()):
        marker = " ★" if name in DEFAULT_CATEGORIES else ""
        print(f"  {name:<30} ({len(zips)} parts){marker}")
    print(f"\n★ = included in default download")
    print(f"Default categories: {', '.join(DEFAULT_CATEGORIES)}")


def main():
    parser = argparse.ArgumentParser(
        description="Download INCLUDE ISL Dataset from Zenodo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_dataset.py                     # Download default subset
  python download_dataset.py --all               # Download everything (~57 GB)
  python download_dataset.py --categories Greetings Animals Colours
  python download_dataset.py --list              # List categories
        """
    )
    parser.add_argument('--list', action='store_true',
                        help='List available categories')
    parser.add_argument('--all', action='store_true',
                        help='Download entire dataset (~57 GB)')
    parser.add_argument('--categories', nargs='+', default=None,
                        help='Specific categories to download')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: data/include_videos)')

    args = parser.parse_args()

    if args.list:
        list_categories()
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output or os.path.join(base_dir, "data", "include_videos")
    os.makedirs(output_dir, exist_ok=True)

    if args.all:
        categories = list(DATASET_CATEGORIES.keys())
    elif args.categories:
        categories = args.categories
    else:
        categories = DEFAULT_CATEGORIES

    print("=" * 60)
    print("  INCLUDE ISL Dataset Downloader")
    print("=" * 60)
    print(f"  Output: {output_dir}")
    print(f"  Categories: {', '.join(categories)}")
    print()

    success = 0
    for cat in categories:
        if download_category(cat, output_dir):
            success += 1

    print(f"\n{'='*60}")
    print(f"  Download complete: {success}/{len(categories)} categories")
    print(f"  Videos saved to: {output_dir}")
    print(f"\n  Next step: Process the videos into landmark data:")
    print(f"    python process_videos.py --input {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
