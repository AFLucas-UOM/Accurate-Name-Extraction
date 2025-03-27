import os
import re
import json
import requests
import logging
import subprocess
from urllib.parse import urlparse
from tqdm import tqdm
from typing import Optional, Dict, Any, List

# Set up logging configuration
LOG_FILE = "TVIA_downloader.log"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", handlers=[
    logging.FileHandler(LOG_FILE, mode="w"),
    logging.StreamHandler()
])

# Define a common User-Agent header to mimic a browser.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/105.0.0.0 Safari/537.36"
    )
}

# Segment download settings
SEGMENT_DURATION = 185  # Each segment is 185 seconds
MAX_SEGMENTS = 3        # Only download the first 3 segments
MIN_SIZE_BYTES = 200 * 1024  # Minimum size for a valid segment (200 KB)

# Target download directory (all files will be stored here)
TARGET_DIR = "/Volumes/Fili's SSD/FYP/Dataset_Videos/ForeignNews"

def extract_identifier(url: str) -> Optional[str]:
    """Extracts the identifier from an Archive.org URL."""
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if "details" in parts:
        idx = parts.index("details")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    logging.error("Error: 'details' not found or identifier missing in the URL.")
    return None

def fetch_metadata(identifier: str) -> Optional[Dict[str, Any]]:
    """Fetches metadata JSON from Archive.org."""
    metadata_url = f"https://archive.org/metadata/{identifier}"
    try:
        response = requests.get(metadata_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Error fetching metadata from Archive.org: {e}")
        return None

def find_mp4_file(files: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Finds the largest available .mp4 file."""
    mp4_files = [f for f in files if f.get("name", "").lower().endswith(".mp4")]
    return max(mp4_files, key=lambda f: int(f.get("size", 0)), default=None)

def is_stream_only(meta: Dict[str, Any]) -> bool:
    """Checks if the video is stream-only (i.e., not directly downloadable)."""
    return meta.get("is_dark", False) or not any(f["name"].endswith(".mp4") for f in meta.get("files", []))

def download_file(url: str, output_path: str) -> bool:
    """Downloads a file with resume capability and progress bar."""
    try:
        headers = HEADERS.copy()
        mode = "wb"
        current_size = 0

        # Resume download if file exists
        if os.path.exists(output_path):
            current_size = os.path.getsize(output_path)
            headers["Range"] = f"bytes={current_size}-"
            mode = "ab"

        response = requests.get(url, stream=True, headers=headers, timeout=10)
        if response.status_code == 403:  # Stream-only case
            return False

        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0)) + current_size
        progress = tqdm(total=total_size, initial=current_size, unit="B", unit_scale=True, desc=os.path.basename(output_path))

        with open(output_path, mode) as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))

        progress.close()
        return True
    except requests.exceptions.RequestException:
        return False

def download_segments(identifier: str, target_dir: str) -> Optional[tuple]:
    """
    Downloads only the first 3 segments of a stream-only video.
    All temporary files (segments and list file) are saved in target_dir.
    """
    base_url = f"http://archive.org/download/{identifier}/{identifier}.mp4"
    segment_list = []
    list_file_path = os.path.join(target_dir, f"{identifier}_list.txt")

    with open(list_file_path, "w") as list_file:
        for i in range(MAX_SEGMENTS):
            start_time = i * SEGMENT_DURATION
            end_time = start_time + SEGMENT_DURATION
            segment_url = f"{base_url}?exact=1&start={start_time}&end={end_time}"
            segment_file_name = f"{identifier}_part{i+1}.mp4"
            segment_path = os.path.join(target_dir, segment_file_name)

            logging.info(f"Downloading segment {i + 1} ({start_time}-{end_time} sec)")
            response = requests.get(segment_url, stream=True, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                with open(segment_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)

                if os.path.getsize(segment_path) > MIN_SIZE_BYTES:
                    segment_list.append(segment_path)
                    # Write only the base file name (relative to target_dir)
                    list_file.write(f"file '{os.path.basename(segment_path)}'\n")
                else:
                    os.remove(segment_path)
                    logging.warning(f"Segment {i + 1} is too small. Skipping.")
            else:
                logging.error(f"Failed to download segment {i + 1}")
                return None

    return segment_list, list_file_path

def concatenate_segments(identifier: str, segment_list: List[str], list_file_path: str, output_file: str):
    """
    Concatenates the downloaded segments into a single output file using FFmpeg.
    The working directory is set to the target folder so that relative paths in the list file are resolved correctly.
    """
    logging.info("Concatenating segments into final video...")
    # Run FFmpeg with working directory set to TARGET_DIR
    result = subprocess.run([
        'ffmpeg', '-f', 'concat', '-safe', '0', '-i', os.path.basename(list_file_path),
        '-c:v', 'libx264', '-preset', 'fast', '-c:a', 'aac',
        '-movflags', '+faststart', os.path.basename(output_file)
    ], cwd=os.path.dirname(output_file))

    if result.returncode == 0:
        logging.info(f"✅ Concatenation complete! Saved as {output_file}")
        # Cleanup temporary segment files and list file.
        for segment in segment_list:
            os.remove(segment)
        os.remove(list_file_path)
    else:
        logging.error("❌ Error: Failed to concatenate video segments.")

def process_video(url: str, output_path: str):
    """
    Processes a single video download from Archive.org.
    The output file, temporary segments, and list file are all created in the same directory.
    """
    identifier = extract_identifier(url)
    if not identifier:
        logging.error(f"Failed to extract identifier for URL: {url}")
        return

    meta = fetch_metadata(identifier)
    if not meta:
        logging.error(f"Failed to fetch metadata for identifier: {identifier}")
        return

    if is_stream_only(meta):
        logging.info(f"Stream-only video detected for URL: {url}")
        segments_result = download_segments(identifier, target_dir=os.path.dirname(output_path))
        if segments_result:
            segment_list, list_file_path = segments_result
            concatenate_segments(identifier, segment_list, list_file_path, output_file=output_path)
        else:
            logging.error(f"Failed to download segments for identifier: {identifier}")
        return

    best_file = find_mp4_file(meta.get("files", []))
    if not best_file:
        logging.error(f"No .mp4 file found for identifier: {identifier}")
        return

    download_url = f"https://archive.org/download/{identifier}/{best_file['name']}"
    logging.info(f"Downloading: {output_path}")
    if not download_file(download_url, output_path):
        logging.info("Failed to download full video. Trying segment-based approach...")
        segments_result = download_segments(identifier, target_dir=os.path.dirname(output_path))
        if segments_result:
            segment_list, list_file_path = segments_result
            concatenate_segments(identifier, segment_list, list_file_path, output_file=output_path)

def get_starting_count(source: str, target_dir: str) -> int:
    """
    Scans the target directory for files matching the pattern <source>_<number>.mp4
    and returns one more than the highest number found, or 1 if none exist.
    """
    pattern = re.compile(r'^' + re.escape(source) + r'_(\d+)\.mp4$', re.IGNORECASE)
    max_count = 0
    for filename in os.listdir(target_dir):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            if num > max_count:
                max_count = num
    return max_count + 1

def main():
    # Load URLs from urls.json (adjusted file path as needed)
    try:
        with open("1. URLs/urls.json", "r") as f:
            urls_list = json.load(f)
    except Exception as e:
        logging.error(f"Error loading urls.json: {e}")
        return

    # Dictionary to track counts per source during this run.
    source_counts = {}

    for entry in urls_list:
        url = entry.get("url")
        source = entry.get("source")
        if not url or not source:
            logging.error("Invalid entry in urls.json. Skipping entry.")
            continue

        # If source is not yet in our running dictionary, initialize it by checking the target directory.
        if source not in source_counts:
            source_counts[source] = get_starting_count(source, TARGET_DIR)

        # Create output file name using the source and its count.
        output_file_name = f"{source}_{source_counts[source]}.mp4"
        output_path = os.path.join(TARGET_DIR, output_file_name)

        logging.info(f"Processing video: {url} -> {output_file_name}")
        process_video(url, output_path)
        # Increment the counter for this source.
        source_counts[source] += 1

if __name__ == "__main__":
    main()