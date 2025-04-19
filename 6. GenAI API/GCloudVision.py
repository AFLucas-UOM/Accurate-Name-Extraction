import cv2
import os
import json
import requests
import base64
import time
import numpy as np
import concurrent.futures
from datetime import datetime
import sys
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
import google.generativeai as genai
import re

class VideoProcessor:
    def __init__(self, video_path, output_dir, api_key_file="config.json", 
                 similarity_threshold=0.9, sample_rate=1, max_workers=5, save_frames=True):
        """Initialize the VideoProcessor."""
        self.video_path = video_path
        self.output_dir = output_dir
        self.api_key_file = api_key_file
        self.similarity_threshold = similarity_threshold
        self.sample_rate = sample_rate
        self.max_workers = max_workers
        self.save_frames = save_frames
        self.console = Console()
        self.gemini_available = False
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if save_frames and not os.path.exists(os.path.join(output_dir, "frames")):
            os.makedirs(os.path.join(output_dir, "frames"))
            
        # Load API keys from config file
        self.api_keys = self._load_api_keys()
        
        # Check for Vision API key
        if not self.api_keys.get('google_cloud_vision_api_key'):
            self.console.print("[bold red]Google Cloud Vision API key not found in config file![/]")
            raise ValueError(f"Vision API key not found in config file {api_key_file}")
        else:
            self.console.print("[green]Successfully loaded Google Cloud Vision API key[/]")
            
        # Set up Gemini API if available
        if self.api_keys.get('google_gemini_api_key'):
            try:
                genai.configure(api_key=self.api_keys['google_gemini_api_key'])
                # Try different model names (in case API changes)
                model_names = ['gemini-1.5-pro', 'gemini-1.0-pro', 'gemini-pro']
                
                for model_name in model_names:
                    try:
                        self.gemini_model = genai.GenerativeModel(model_name)
                        # Test with a simple prompt to verify it works
                        response = self.gemini_model.generate_content("Hello")
                        if response:
                            self.gemini_available = True
                            self.console.print(f"[green]Successfully connected to Gemini API using model:[/] {model_name}")
                            break
                    except Exception as e:
                        self.console.print(f"[yellow]Model {model_name} not available:[/] {str(e)}")
                
                if not self.gemini_available:
                    self.console.print("[bold red]Could not connect to any Gemini model. Exiting.[/]")
                    raise ValueError("Gemini API is required but not available")
            except Exception as e:
                self.console.print(f"[bold red]Gemini API initialization failed:[/] {str(e)}")
                raise ValueError("Gemini API is required but initialization failed")
        else:
            self.console.print("[bold red]No Gemini API key found in config. Exiting.[/]")
            raise ValueError("Gemini API key is required but not found in config")
            
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize tracking variables
        self.last_frame = None
        self.results = []
        self.processed_count = 0
    
    def _load_api_keys(self):
        """Load API keys from the config file."""
        try:
            if not os.path.exists(self.api_key_file):
                self.console.print(f"[bold red]Config file not found:[/] {self.api_key_file}")
                return {}
                
            with open(self.api_key_file, 'r') as f:
                config = json.load(f)
                
            return config
        except Exception as e:
            self.console.print(f"[bold red]Error loading API keys:[/] {str(e)}")
            return {}
        
    def compute_frame_hash(self, frame):
        """Compute a perceptual hash for a frame using DCT."""
        # Resize and convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32))
        
        # Compute DCT
        dct = cv2.dct(np.float32(resized))
        
        # Keep only the top-left 8x8 coefficients which contain most of the signal energy
        dct_low = dct[:8, :8]
        
        # Compute median value
        median = np.median(dct_low)
        
        # Convert to binary hash (1 if value > median, else 0)
        hash_value = (dct_low > median).flatten()
        
        return hash_value
    
    def frame_similarity(self, hash1, hash2):
        """Calculate similarity between two frame hashes."""
        if hash1 is None or hash2 is None:
            return 0
            
        # Calculate Hamming distance
        distance = np.count_nonzero(hash1 != hash2)
        
        # Calculate similarity (1 - normalized distance)
        similarity = 1 - (distance / len(hash1))
        
        return similarity
        
    def format_timestamp(self, frame_num):
        """Convert frame number to timestamp string."""
        seconds = frame_num / self.fps
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
    def save_frame_image(self, frame, frame_num):
        """Save frame as JPEG image."""
        filename = f"frame_{frame_num:05d}.jpg"
        filepath = os.path.join(self.output_dir, "frames", filename)
        cv2.imwrite(filepath, frame)
        return filename
        
    def detect_text(self, image_content):
        """Detect text in an image using Google Cloud Vision API."""
        url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_keys['google_cloud_vision_api_key']}"
        
        payload = {
            "requests": [
                {
                    "image": {
                        "content": image_content
                    },
                    "features": [
                        {
                            "type": "TEXT_DETECTION"
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code != 200:
            self.console.print(f"[bold red]Error from Vision API:[/] {response.status_code} - {response.text}")
            return ""
            
        result = response.json()
        
        # Extract text annotations if available
        try:
            annotations = result["responses"][0].get("textAnnotations", [])
            if annotations:
                return annotations[0].get("description", "").strip()
            return ""
        except (KeyError, IndexError):
            return ""
            
    def process_frame(self, frame, frame_num):
        """Process a single frame."""
        # Encode image for API request
        _, buffer = cv2.imencode('.jpg', frame)
        image_content = base64.b64encode(buffer).decode('utf-8')
        
        # Detect text
        detected_text = self.detect_text(image_content)
        
        # Save frame image if requested
        frame_filename = None
        if self.save_frames:
            frame_filename = self.save_frame_image(frame, frame_num)
        
        # Create result
        result = {
            "timestamp": self.format_timestamp(frame_num),
            "frame": frame_filename,
            "text": detected_text
        }
        
        return result
    
    def extract_names_with_gemini(self, text_data):
        """Use Google Gemini to extract names from text data."""
        if not self.gemini_available:
            self.console.print("[bold red]Gemini API not available. Exiting.[/]")
            raise ValueError("Gemini API is required but not available")
            
        # Prepare prompt for Gemini
        prompt = f"""
        I need to extract proper names (people's names) from the text data from a video.
        
        The text data is from multiple frames and is as follows:
        
        {json.dumps(text_data, indent=2)}
        
        Analyze the text and extract ONLY real people's names. Skip:
        - Brand names
        - Channel names
        - Show names
        - App names
        - Non-name text

        Return the result as a JSON object with this exact format:
        {{
          "names": [
            {{
              "name": "Full Name",
              "first_appearance": "timestamp",
              "last_appearance": "timestamp",
              "count": number_of_appearances
            }}
          ]
        }}
        
        Only include valid people's names. Return an empty array if no real people's names are found.
        Be precise and return ONLY the JSON object, nothing else.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            result_text = response.text
            
            # Extract JSON portion if there's additional text
            try:
                json_start = result_text.find('{')
                json_end = result_text.rfind('}')
                if json_start >= 0 and json_end >= 0:
                    json_str = result_text[json_start:json_end + 1]
                    return json.loads(json_str)
                else:
                    self.console.print("[bold red]No valid JSON found in Gemini response. Exiting.[/]")
                    raise ValueError("Failed to parse Gemini response")
            except Exception as parse_error:
                self.console.print(f"[bold red]Could not parse Gemini response as JSON:[/] {str(parse_error)}")
                raise ValueError("Failed to parse Gemini response as JSON")
                
        except Exception as e:
            self.console.print(f"[bold red]Error using Gemini API:[/] {str(e)}")
            raise ValueError(f"Gemini API error: {str(e)}")
    
    def process_video(self):
        """Process the video, detecting distinct frames and extracting text."""
        start_time = time.time()
        
        # Create a rich progress display
        self.console.print(Panel.fit(f"[bold blue]Processing Video:[/] {os.path.basename(self.video_path)}", 
                                    title="Video Analysis", subtitle="Starting processing..."))
        
        self.console.print(f"[bold]Video properties:[/] {self.width}x{self.height}, {self.fps:.2f} fps, {self.frame_count} frames")
        
        # Frame extraction progress
        frame_num = 0
        last_hash = None
        distinct_frames = []
        
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console
        ) as progress:
            # Frame sampling task
            frame_task = progress.add_task("[cyan]Sampling frames...", total=self.frame_count)
            
            # Read frames
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Process every nth frame (based on sample rate)
                if frame_num % self.sample_rate == 0:
                    # Compute perceptual hash
                    current_hash = self.compute_frame_hash(frame)
                    
                    # Check if frame is distinct from previous
                    if last_hash is None or self.frame_similarity(current_hash, last_hash) < self.similarity_threshold:
                        distinct_frames.append((frame, frame_num))
                        last_hash = current_hash
                
                frame_num += 1
                progress.update(frame_task, advance=1)
                
            self.cap.release()
        
        self.console.print(f"[bold green]Found {len(distinct_frames)} distinct frames[/]")
        
        # Text detection progress
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            # Text detection task
            text_task = progress.add_task("[cyan]Detecting text...", total=len(distinct_frames))
            
            # Process distinct frames in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.process_frame, frame, frame_num): (frame, frame_num) 
                          for frame, frame_num in distinct_frames}
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result["text"]:  # Only add frames with detected text
                            self.results.append(result)
                    except Exception as e:
                        frame, frame_num = futures[future]
                        self.console.print(f"[bold red]Error processing frame {frame_num}:[/] {str(e)}")
                    
                    progress.update(text_task, advance=1)
        
        # Sort results by timestamp
        self.results.sort(key=lambda x: x["timestamp"])
        
        # Save results as JSON
        with open(os.path.join(self.output_dir, "results.json"), "w") as f:
            json.dump(self.results, f, indent=2)
            
        # Create a simplified representation for name extraction
        frames_with_text = []
        for result in self.results:
            frames_with_text.append({
                "timestamp": result["timestamp"],
                "text": result["text"]
            })
            
        # Extract names using Gemini
        self.console.print("\n[bold]Extracting names using Gemini...[/]")
        name_data = self.extract_names_with_gemini(frames_with_text)
        
        # Generate summary statistics
        elapsed_time = time.time() - start_time
        
        # Create summary
        summary = {
            "total_video_frames": self.frame_count,
            "distinct_frames_processed": len(distinct_frames),
            "frames_with_text": len(self.results),
            "processing_time_seconds": elapsed_time,
            "fps": self.fps,
            "duration_seconds": self.frame_count / self.fps,
            "names": name_data.get("names", [])
        }
        
        # Save summary
        with open(os.path.join(self.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        # Display results table
        table = Table(title="Video Analysis Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Video file", os.path.basename(self.video_path))
        table.add_row("Duration", f"{summary['duration_seconds']:.2f} seconds")
        table.add_row("Total frames", f"{self.frame_count}")
        table.add_row("Distinct frames", f"{len(distinct_frames)}")
        table.add_row("Frames with text", f"{len(self.results)}")
        table.add_row("Unique names found", f"{len(summary['names'])}")
        table.add_row("Processing time", f"{elapsed_time:.2f} seconds")
        
        self.console.print(table)
        
        # Display names if found
        if summary['names']:
            name_table = Table(title="Detected Names")
            name_table.add_column("Name", style="cyan")
            name_table.add_column("First Seen", style="green")
            name_table.add_column("Last Seen", style="green")
            name_table.add_column("Appearances", style="green")
            
            for name_entry in summary['names']:
                name_table.add_row(
                    name_entry["name"],
                    name_entry["first_appearance"],
                    name_entry["last_appearance"],
                    str(name_entry["count"])
                )
            
            self.console.print(name_table)
        else:
            self.console.print("[yellow]No names were detected in the video.[/]")
            
        self.console.print(f"\n[bold green]Results saved to:[/] {os.path.join(self.output_dir, 'results.json')}")
        self.console.print(f"[bold green]Summary saved to:[/] {os.path.join(self.output_dir, 'summary.json')}")
        
        return self.results

if __name__ == "__main__":
    import argparse

    def print_custom_usage_and_exit():
        usage_msg = (
            '\n[bold red]❌ Invalid or missing arguments.[/]\n\n'
            '[bold]Usage:[/] python "6. GenAI API/GCloudVision.py" "6. GenAI API/Videos/XXXX.mp4" "6. GenAI API/results" "6. GenAI API/config.json"\n'
        )
        Console().print(usage_msg)
        sys.exit(1)

    if len(sys.argv) >= 4 and not sys.argv[1].startswith('-'):
        video_path = sys.argv[1]
        output_dir = sys.argv[2]
        api_key_file = sys.argv[3]

        console = Console()
        try:
            processor = VideoProcessor(
                video_path=video_path,
                output_dir=output_dir,
                api_key_file=api_key_file,
                similarity_threshold=0.9,
                sample_rate=5,
                max_workers=5,
                save_frames=True
            )
            processor.process_video()
        except Exception as e:
            console.print(f"[bold red]Error:[/] {str(e)}")
            sys.exit(1)

    elif len(sys.argv) == 1:
        print_custom_usage_and_exit()

    else:
        # Catch bad usage before letting argparse print ugly default
        if any(arg.startswith('-') for arg in sys.argv[1:]):
            # Argparse-style usage
            parser = argparse.ArgumentParser(
                description="Process video frames for text and name detection",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                usage=(
                    '\npython "6. GenAI API/GCloudVision.py" '
                    '"6. GenAI API/Videos/Video_T2.mp4" '
                    '"6. GenAI API/results" '
                    '"6. GenAI API/config.json"\n'
                )
            )
            parser.add_argument("video_path", help="Path to the video file")
            parser.add_argument("--output-dir", "-o", default="output", help="Directory to save results")
            parser.add_argument("--api-key-file", "-k", default="config.json", help="Path to the JSON file containing API keys")
            parser.add_argument("--similarity-threshold", "-s", type=float, default=0.9, help="Threshold for frame similarity (0-1)")
            parser.add_argument("--sample-rate", "-r", type=int, default=5, help="Process every nth frame")
            parser.add_argument("--max-workers", "-w", type=int, default=5, help="Maximum number of concurrent workers")
            parser.add_argument("--no-save-frames", action="store_false", dest="save_frames", help="Don't save frame images")

            try:
                args = parser.parse_args()
                processor = VideoProcessor(
                    video_path=args.video_path,
                    output_dir=args.output_dir,
                    api_key_file=args.api_key_file,
                    similarity_threshold=args.similarity_threshold,
                    sample_rate=args.sample_rate,
                    max_workers=args.max_workers,
                    save_frames=args.save_frames
                )
                processor.process_video()
            except Exception as e:
                Console().print(f"[bold red]Error:[/] {str(e)}")
                sys.exit(1)
        else:
            print_custom_usage_and_exit()
