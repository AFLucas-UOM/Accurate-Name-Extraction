import os
import sys
import json
import re
import yt_dlp
import logging
from datetime import datetime

# Set up logging configuration
LOG_FILE = "TKTK_downloader.log"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s: %(message)s",
                    handlers=[
                        logging.FileHandler(LOG_FILE, mode="w"),
                        logging.StreamHandler()
                    ])

# Target download directory for TikTok videos.
TARGET_DIR = "/Volumes/Fili's SSD/FYP/Dataset_Videos/TiktokNews"

def format_size(bytes):
    """Convert bytes to a human-readable string."""
    if bytes is None:
        return 'Unknown'
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024

def format_time(seconds):
    """Convert seconds to a human-readable string."""
    if seconds is None:
        return 'Unknown'
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    elif minutes:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def create_progress_bar(percentage, width=50):
    """Create a progress bar with specified width and percentage filled."""
    filled = int(width * percentage / 100) if percentage else 0
    return "█" * filled + "░" * (width - filled)

def progress_hook(d):
    """Handle download progress updates."""
    status = d.get('status')
    if status == 'downloading':
        total = d.get('total_bytes') or d.get('total_bytes_estimate')
        downloaded = d.get('downloaded_bytes', 0)
        percentage = (downloaded / total) * 100 if total else 0
        speed = f"{format_size(d.get('speed', 0))}/s"
        eta = format_time(d.get('eta'))
        progress_bar = create_progress_bar(percentage)
        print(f"Downloading: {percentage:5.1f}% {progress_bar} {format_size(downloaded)}/{format_size(total)} {speed} ETA: {eta}", end='\r')
    elif status == 'finished':
        sys.stdout.write('\x1b[2K\r')
        print("✅ Download completed! Processing...")

def check_ffmpeg():
    """Check if FFmpeg is installed and accessible."""
    try:
        import subprocess
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def get_starting_count(source, target_dir):
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

def setup_download_options(download_path, source, count):
    """Return download options for yt-dlp with naming style Source_X."""
    return {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(download_path, f'{source}_{count}.mp4'),
        'progress_hooks': [progress_hook],
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'postprocessors': [{'key': 'FFmpegVideoRemuxer', 'preferedformat': 'mp4'}],
        'prefer_ffmpeg': True
    }

def download_video(url, download_path, provided_source=None):
    """
    Download video from the given TikTok URL using yt-dlp.
    If provided_source is given (from the JSON file), it is used for naming.
    """
    try:
        # Extract information to get title and duration
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown_Title')
            duration = format_time(info.get('duration'))
            # Use the provided source if available; otherwise fallback to uploader or "TikTok"
            source = provided_source if provided_source is not None else (info.get('uploader') or "TikTok")
            print(f"\n📹 Title: {title}\n⏱️  Duration: {duration}\nSource: {source}")
            print("📥 Downloading video...")
        
        # Determine the next available count for this source in the target directory
        count = get_starting_count(source, download_path)
        ydl_opts = setup_download_options(download_path, source, count)
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            return filename if filename.endswith('.mp4') else os.path.splitext(filename)[0] + '.mp4'
    except Exception as e:
        sys.stdout.write('\x1b[2K\r')
        print(f"❌ Error: {str(e)}")
        logging.error(f"Error downloading {url}: {e}")
        return None

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n=== TikTok Video Downloader ===\n")
    
    if not check_ffmpeg():
        print("❌ FFmpeg is not installed. Please install it to continue.")
        print("• Windows: choco install ffmpeg\n• macOS: brew install ffmpeg\n• Linux: sudo apt install ffmpeg")
        sys.exit(1)
    
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    urls_file = "../1. URLs/TKTK_urls.json"
    try:
        with open(urls_file, "r") as f:
            data = json.load(f)
            urls = []
            # Expect each entry to be a dict with "url" and "source" keys or a simple URL string.
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        url = item.get("url")
                        source = item.get("source")
                        if url:
                            urls.append((url, source))
                    elif isinstance(item, str):
                        urls.append((item, None))
            else:
                logging.error("JSON file format is not valid.")
                sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading {urls_file}: {e}")
        sys.exit(1)
    
    start_time = datetime.now()
    
    for url, source in urls:
        logging.info(f"Starting download for URL: {url} with source: {source}")
        downloaded_file = download_video(url, TARGET_DIR, provided_source=source)
        if downloaded_file:
            logging.info(f"Download complete: {downloaded_file}")
        else:
            logging.error(f"Download failed for URL: {url}")
    
    total_time = datetime.now() - start_time
    print(f"\n✅ All downloads complete!\n⏱️  Total time: {format_time(total_time.total_seconds())}\n")
    logging.info(f"All downloads complete in {format_time(total_time.total_seconds())}")

if __name__ == "__main__":
    main()
