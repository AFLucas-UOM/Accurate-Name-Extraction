import os
import sys
import yt_dlp
from datetime import datetime

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
    status = d['status']
    if status == 'downloading':
        total = d.get('total_bytes') or d.get('total_bytes_estimate')
        downloaded = d.get('downloaded_bytes', 0)
        percentage = (downloaded / total) * 100 if total else None
        speed = f"{format_size(d.get('speed', 0))}/s"
        eta = format_time(d.get('eta'))
        progress_bar = create_progress_bar(percentage) if percentage else "≈" * 50
        print(f" Downloading: {percentage:5.1f}% {progress_bar} {format_size(downloaded)}/{format_size(total)} {speed} ETA: {eta}", end='\r')
    elif status == 'finished':
        sys.stdout.write('\x1b[2K\r')
        print(" ✅ Download completed! Processing...")

def check_ffmpeg():
    """Check if FFmpeg is installed and accessible."""
    try:
        import subprocess
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def setup_download_options(download_path):
    """Return download options for yt-dlp."""
    return {
        'format': 'bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': os.path.join(download_path, '%(title)s.%(ext)s'),
        'progress_hooks': [progress_hook],
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{'key': 'FFmpegVideoRemuxer', 'preferedformat': 'mp4'}],
        'prefer_ffmpeg': True
    }

def download_video(url, ydl_opts):
    """Download video from the given URL using yt-dlp."""
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown Title')
            duration = format_time(info.get('duration'))
            print(f" 📹 Title: {title}\n ⏱️  Duration: {duration}")
            print("\n Downloading video...")
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            return filename if filename.endswith('.mp4') else os.path.splitext(filename)[0] + '.mp4'
    except Exception as e:
        sys.stdout.write('\x1b[2K\r')
        print(f" ❌ Error: {str(e)}")
        return None

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n=== YouTube Video Downloader ===\n")
    
    if not check_ffmpeg():
        print(" ❌ FFmpeg is not installed. Please install it to continue.")
        print(" • Windows: choco install ffmpeg\n • macOS: brew/pip install ffmpeg\n • Linux: sudo apt install ffmpeg")
        sys.exit(1)
    
    url = input(" Enter YouTube URL: ")
    print("\n 📥 Starting download process...")
    start_time = datetime.now()

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    ydl_opts = setup_download_options(desktop_path)
    downloaded_file = download_video(url, ydl_opts)
    
    if downloaded_file:
        total_time = datetime.now() - start_time
        print(f" ✅ Download complete!\n 📁 Saved to: {downloaded_file}\n ⏱️  Total time: {format_time(total_time.total_seconds())}\n")
