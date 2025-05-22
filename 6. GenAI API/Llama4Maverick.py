import cv2
import os
import json
import base64
import time
import numpy as np
import concurrent.futures
import sys
from datetime import datetime, timedelta
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import backoff 

# ── OpenAI / OpenRouter client (>=1.0.0) ────────────────────────────────
try:
    from openai import OpenAI  # openai >= 1.0.0
    from openai.types.chat import ChatCompletionMessage
    from openai.types import CompletionUsage
except ImportError as e:
    raise ImportError(
        "The openai Python package ≥ 1.0.0 is required. Install/upgrade with `pip install --upgrade openai`."
    ) from e


@dataclass
class RateLimiter:
    """Manages rate limits with adaptive backoff"""
    # Configuration
    initial_requests_per_minute: int = 30
    min_requests_per_minute: int = 5
    max_requests_per_minute: int = 50
    backoff_factor: float = 0.8
    recovery_factor: float = 1.05
    
    # Runtime state
    current_rpm: float = field(init=False)
    request_times: List[float] = field(default_factory=list)
    consecutive_successes: int = field(default=0, init=False)
    consecutive_failures: int = field(default=0, init=False)
    
    def __post_init__(self):
        self.current_rpm = float(self.initial_requests_per_minute)
        self.console = Console()
    
    def wait_if_needed(self) -> None:
        """Wait if we're exceeding our self-imposed rate limit"""
        now = time.time()
        
        # Clean up old request timestamps (older than 1 minute)
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # Calculate current requests in the last minute
        requests_in_last_minute = len(self.request_times)
        
        # If we're over our limit, wait
        if requests_in_last_minute >= self.current_rpm:
            # Calculate the oldest request in our window + 60s
            oldest = min(self.request_times) if self.request_times else now
            wait_time = max(oldest + 60 - now, 0) + 0.1  # Add a small buffer
            
            wait_until = datetime.fromtimestamp(now + wait_time)
            self.console.print(f"[cyan]⏳ Rate limiting ({requests_in_last_minute}/{self.current_rpm:.1f} RPM). "
                              f"Waiting {wait_time:.1f}s until {wait_until.strftime('%H:%M:%S')}…[/]")
            time.sleep(wait_time)
    
    def record_request(self) -> None:
        """Record a new request timestamp"""
        self.request_times.append(time.time())
    
    def record_success(self) -> None:
        """Record a successful API call"""
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        
        # Gradually increase our rate limit after consistent success
        if self.consecutive_successes >= 10:
            self.current_rpm = min(self.current_rpm * self.recovery_factor, self.max_requests_per_minute)
            self.consecutive_successes = 0
            self.console.print(f"[green]✓ Increasing rate limit to {self.current_rpm:.1f} RPM[/]")
    
    def record_rate_limited(self, reset_time: Optional[float] = None) -> float:
        """
        Record a rate limit failure and calculate backoff time
        Returns the time to wait in seconds
        """
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        
        # Reduce our rate limit
        self.current_rpm = max(self.current_rpm * self.backoff_factor, self.min_requests_per_minute)
        
        # Calculate wait time based on reset_time or our current rate
        if reset_time:
            wait_time = max(reset_time - time.time(), 1)
        else:
            # Calculate based on our new rate limit
            wait_time = max(60 / self.current_rpm * (1 + self.consecutive_failures * 0.5), 5)
            # Cap at 2 minutes
            wait_time = min(wait_time, 120)
        
        self.console.print(f"[yellow]⚠️ Rate limited! Reducing to {self.current_rpm:.1f} RPM[/]")
        return wait_time


@dataclass
class APIMetrics:
    """Tracks API usage metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0
    
    def update_from_usage(self, usage: Optional[CompletionUsage]) -> None:
        """Update metrics from OpenAI usage object"""
        if usage:
            self.total_tokens += usage.total_tokens
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            # Approximate cost: $0.15 per 1M tokens (adjust as needed)
            self.total_cost += usage.total_tokens * 0.00000015


class VideoProcessor:
    def __init__(
        self,
        video_path: str,
        output_dir: str,
        api_key_file: str = "config.json",
        *,
        similarity_threshold: float = 0.9,
        sample_rate: int = 5,
        max_workers: int = 5,
        save_frames: bool = True,
        extract_names: bool = True,
        min_frame_text_length: int = 3,  # Filter out frames with too little text
    ) -> None:
        self.console = Console()
        self.video_path = video_path
        self.output_dir = output_dir
        self.api_key_file = api_key_file
        self.similarity_threshold = similarity_threshold
        self.sample_rate = sample_rate
        self.max_workers = max_workers
        self.save_frames = save_frames
        self.do_extract_names = extract_names
        self.min_frame_text_length = min_frame_text_length

        # ── I/O setup ─────────────────────────────────────────────────────
        os.makedirs(output_dir, exist_ok=True)
        if save_frames:
            os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)

        # ── Keys & OpenRouter setup ──────────────────────────────────────
        self.api_keys = self._load_api_keys()
        if not self.api_keys.get("openrouter_api_key"):
            self.console.print("[bold red]❌ OpenRouter API key missing in config![/]")
            raise ValueError("API key required under 'openrouter_api_key'.")

        self.client = OpenAI(
            api_key=self.api_keys["openrouter_api_key"],
            base_url="https://openrouter.ai/api/v1",  # OpenRouter endpoint
        )
        self.model = "meta-llama/llama-4-maverick:free"
        
        # ── API rate limiting and metrics ─────────────────────────────────
        self.rate_limiter = RateLimiter()
        self.metrics = APIMetrics()

        # ── Video capture ────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30  # fallback
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.results: List[Dict[str, Any]] = []
        
        # ── Feature detection (for better frame comparison) ───────────────
        self.orb = cv2.ORB_create() if hasattr(cv2, 'ORB_create') else None

    # ── Internal helpers ─────────────────────────────────────────────────

    def _process_single_frame(self, frame: np.ndarray, frame_num: int) -> Dict[str, Any]:
        """Process a single frame with OCR, with proper error handling and retries"""
        fname, path = self._save_frame(frame, frame_num) if self.save_frames else ("", "")
        
        if not path:
            return {
                "timestamp": self._format_timestamp(frame_num),
                "frame": fname,
                "text": "",
                "frame_num": frame_num,
                "processed_successfully": False,
            }
            
        # Try OCR with robust retries
        text = self._detect_text_with_retry(path)
        
        # Track if this was processed successfully - empty responses count as not successful
        processed_successfully = bool(text and len(text.strip()) >= self.min_frame_text_length)
        
        return {
            "timestamp": self._format_timestamp(frame_num),
            "frame": fname,
            "text": text,
            "frame_num": frame_num,
            "processed_successfully": processed_successfully,
            "path": path,  # Include path for retry attempts
        }

    def _load_api_keys(self) -> Dict[str, str]:
        if not os.path.exists(self.api_key_file):
            return {}
        with open(self.api_key_file, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def _compute_frame_hash(self, frame: np.ndarray) -> np.ndarray:
        """Improved perceptual hash using both DCT and ORB features for better accuracy"""
        # Method 1: DCT-based perceptual hash
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        dct_low = dct[:8, :8]
        median = np.median(dct_low)
        phash = (dct_low > median).flatten()
        
        # Method 2: ORB features (if available)
        if self.orb:
            try:
                # Use a feature-based approach for detecting actual content changes
                keypoints = self.orb.detect(gray, None)
                # Sort keypoints by response strength and take top 10
                keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:10]
                # Create a feature vector from keypoint coordinates
                features = np.array([(kp.pt[0]/self.width, kp.pt[1]/self.height) for kp in keypoints])
                # If we have features, combine with phash
                if len(features) > 0:
                    # Flatten and binarize features
                    feature_bits = (features.flatten() > 0.5).astype(np.uint8)
                    # Combine with phash
                    return np.concatenate([phash, feature_bits])
            except Exception:
                # Fall back to just phash if ORB fails
                pass
                
        return phash

    def _frame_similarity(self, h1: Optional[np.ndarray], h2: Optional[np.ndarray]) -> float:
        if h1 is None or h2 is None:
            return 0.0
        
        # If lengths don't match, use only the common part (happens when ORB features vary)
        min_len = min(len(h1), len(h2))
        return 1.0 - np.count_nonzero(h1[:min_len] != h2[:min_len]) / min_len

    def _format_timestamp(self, frame_num: int) -> str:
        secs = frame_num / self.fps
        h, m = divmod(secs, 3600)
        m, s = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

    def _save_frame(self, frame: np.ndarray, frame_num: int) -> Tuple[str, str]:
        fname = f"frame_{frame_num:05d}.jpg"
        path = os.path.join(self.output_dir, "frames", fname)
        
        # Save with better compression for smaller files
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return fname, path

    # ── API calls with backoff/retry logic ───────────────────────────────

    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_tries=5,
        giveup=lambda e: "is not a valid api_key" in str(e).lower()
    )
    def _detect_text_with_retry(self, image_path: str, alternate_prompt: bool = False) -> str:
        """OCR text detection with proper rate limiting and retries
        
        Args:
            image_path: Path to the image file
            alternate_prompt: Use an alternate prompt for retry attempts
        """
        # Check if we need to wait before making API call
        self.rate_limiter.wait_if_needed()
        
        with open(image_path, "rb") as img_file:
            b64_img = base64.b64encode(img_file.read()).decode()
            
        # Use different prompts for regular vs retry attempts to improve results
        if not alternate_prompt:
            prompt_text = "Extract and return only visible text from this image. Respond with plain text only."
        else:
            # For retry attempts, use a more detailed prompt to get better results
            prompt_text = (
                "This image may contain hard-to-read text. Please analyze carefully and extract ALL visible text, "
                "even if it's partially obscured or in unusual positions. Be thorough and focus on any text that "
                "might be present. Respond with ONLY the extracted text, nothing else."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                    },
                ],
            }
        ]

        try:
            # Record this API request
            self.rate_limiter.record_request()
            self.metrics.total_requests += 1
            
            # Make API call
            res = self.client.chat.completions.create(model=self.model, messages=messages)
            
            # Process successful result
            if res and res.choices and res.choices[0].message and res.choices[0].message.content:
                self.metrics.successful_requests += 1
                self.rate_limiter.record_success()
                
                # Update usage metrics
                if hasattr(res, 'usage'):
                    self.metrics.update_from_usage(res.usage)
                    
                return res.choices[0].message.content.strip()
            else:
                self.metrics.failed_requests += 1
                self.console.print("[yellow]⚠️ Empty response from LLaMA OCR[/]")
                return ""
                
        except Exception as exc:
            err_msg = str(exc).lower()
            
            # Handle rate limiting
            if "rate limit" in err_msg or (hasattr(exc, "status_code") and exc.status_code == 429):
                self.metrics.rate_limited_requests += 1
                
                # Try to parse rate-limit reset time
                reset_ts = None
                if hasattr(exc, "response") and hasattr(exc.response, "headers"):
                    headers = exc.response.headers
                    if "X-RateLimit-Reset" in headers:
                        try:
                            reset_ts = int(headers["X-RateLimit-Reset"]) / 1000  # to seconds
                        except:
                            pass
                
                # Calculate wait time
                wait_time = self.rate_limiter.record_rate_limited(reset_ts)
                wait_until = datetime.fromtimestamp(time.time() + wait_time)
                
                self.console.print(f"[bold yellow]🛑 Rate limit exceeded![/] "
                                  f"Waiting {wait_time:.1f}s until {wait_until.strftime('%H:%M:%S')}…")
                time.sleep(wait_time)
                
                # Recursive retry after waiting
                return self._detect_text_with_retry(image_path)
            else:
                self.metrics.failed_requests += 1
                self.console.print(f"[bold red]LLaMA OCR error:[/] {exc}")
                # Let backoff handle the retry
                raise
                
        return ""

    def _extract_names_with_retry(self, text_data: List[Dict[str, str]]) -> Dict[str, Any]:
        """Extract names with proper rate limiting and retries"""
        # Check if we need to wait before making API call
        self.rate_limiter.wait_if_needed()
        
        prompt_text = (
            "From the OCR text snippets below, extract real people's names. "
            "Skip brands, TV shows, or non‑person names.\n\n" + json.dumps(text_data, indent=2) + "\n\n" +
            "Respond ONLY with this JSON format:\n"
            "{\n  \"names\": [\n    {\n      \"name\": \"Full Name\",\n      \"first_appearance\": \"timestamp\",\n"
            "      \"last_appearance\": \"timestamp\",\n      \"count\": number\n    }\n  ]\n}"
        )

        messages = [{"role": "user", "content": prompt_text}]

        try:
            # Record this API request
            self.rate_limiter.record_request()
            self.metrics.total_requests += 1
            
            # Make API call
            res = self.client.chat.completions.create(model=self.model, messages=messages)
            
            # Process successful result
            if res and res.choices and res.choices[0].message and res.choices[0].message.content:
                self.metrics.successful_requests += 1
                self.rate_limiter.record_success()
                
                # Update usage metrics
                if hasattr(res, 'usage'):
                    self.metrics.update_from_usage(res.usage)
                
                content = res.choices[0].message.content
                start, end = content.find("{"), content.rfind("}")
                
                if start != -1 and end != -1:
                    try:
                        payload = json.loads(content[start:end + 1])
                        return payload if isinstance(payload, dict) else {"names": []}
                    except json.JSONDecodeError:
                        self.console.print("[yellow]⚠️ Failed to parse JSON from name extraction response[/]")
                        return {"names": []}
                else:
                    self.console.print("[yellow]⚠️ No valid JSON found in name extraction response[/]")
                    return {"names": []}
            else:
                self.metrics.failed_requests += 1
                self.console.print("[yellow]⚠️ Empty response during name extraction[/]")
                return {"names": []}
                
        except Exception as exc:
            err_msg = str(exc).lower()
            
            # Handle rate limiting
            if "rate limit" in err_msg or (hasattr(exc, "status_code") and exc.status_code == 429):
                self.metrics.rate_limited_requests += 1
                
                # Try to parse rate-limit reset time
                reset_ts = None
                if hasattr(exc, "response") and hasattr(exc.response, "headers"):
                    headers = exc.response.headers
                    if "X-RateLimit-Reset" in headers:
                        try:
                            reset_ts = int(headers["X-RateLimit-Reset"]) / 1000  # to seconds
                        except:
                            pass
                
                # Calculate wait time
                wait_time = self.rate_limiter.record_rate_limited(reset_ts)
                wait_until = datetime.fromtimestamp(time.time() + wait_time)
                
                self.console.print(f"[bold yellow]🛑 Rate limit exceeded during name extraction![/] "
                                  f"Waiting {wait_time:.1f}s until {wait_until.strftime('%H:%M:%S')}…")
                time.sleep(wait_time)
                
                # Recursive retry after waiting
                return self._extract_names_with_retry(text_data)
            else:
                self.metrics.failed_requests += 1
                self.console.print(f"[bold red]Name extraction error:[/] {exc}")
                return {"names": []}
                
        return {"names": []}

    # ── Public API ───────────────────────────────────────────────────────

    def process_video(self) -> List[Dict[str, Any]]:
        """Run the end‑to‑end pipeline. Returns the raw OCR results list."""
        t_start = time.time()

        self.console.print(Panel.fit(f"[bold blue]Processing:[/] {os.path.basename(self.video_path)}", title="Video OCR"))
        self.console.print(
            f"[bold]Properties:[/] {self.width}×{self.height} @ {self.fps:.2f} fps • {self.frame_count} frames"
        )

        # ── 1. Sample *distinct* frames ─────────────────────────────────
        distinct: List[Tuple[np.ndarray, int]] = []
        last_hash: Optional[np.ndarray] = None
        frame_no = 0
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as prog:
            task = prog.add_task("Sampling frames…", total=self.frame_count)
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                if frame_no % self.sample_rate == 0:
                    h = self._compute_frame_hash(frame)
                    if last_hash is None or self._frame_similarity(h, last_hash) < self.similarity_threshold:
                        distinct.append((frame, frame_no))
                        last_hash = h
                frame_no += 1
                prog.update(task, advance=1)
        self.cap.release()
        self.console.print(f"[green]Found {len(distinct)} distinct frames from {self.frame_count} total frames[/]")

        # ── 2. OCR each distinct frame ──────────────────────────────────
        processed_frames = []
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as prog:
            task = prog.add_task("OCR processing…", total=len(distinct))
            
            # Use thread pool for concurrent processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as exe:
                # Submit all jobs
                future_to_frame = {
                    exe.submit(self._process_single_frame, frame, num): (frame, num) 
                    for frame, num in distinct
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_frame):
                    result = future.result()
                    processed_frames.append(result)
                    
                    # Update progress
                    prog.update(task, advance=1)
                    
                    # Display live stats periodically
                    if len(processed_frames) % 10 == 0 or len(processed_frames) == len(distinct):
                        successful = sum(1 for r in processed_frames if r.get("processed_successfully", False))
                        self.console.print(
                            f"[dim]Progress: {len(processed_frames)}/{len(distinct)} frames processed, "
                            f"{successful} with text, RPM: {self.rate_limiter.current_rpm:.1f}[/]"
                        )

        # ── 2b. Retry frames with empty responses ─────────────────────────
        failed_frames = [f for f in processed_frames if not f.get("processed_successfully", False)]
        if failed_frames:
            self.console.print(f"[yellow]⚠️ {len(failed_frames)} frames had empty responses. Retrying them now...[/]")
            
            # Prioritize frames that had empty responses
            with Progress(
                TextColumn("[bold yellow]Retry OCR…"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=self.console,
            ) as prog:
                retry_task = prog.add_task("Retrying frames…", total=len(failed_frames))
                
                # Reset rate limiter state a bit to ensure we don't immediately hit limits
                self.rate_limiter.request_times = self.rate_limiter.request_times[-10:] if len(self.rate_limiter.request_times) > 10 else []
                
                # Add a delay before retries to ensure rate limits have cooled down
                time.sleep(5)
                
                # Sequential retries (more controlled than parallel for retries)
                for idx, frame_data in enumerate(failed_frames):
                    # Check if we have the path to retry
                    if "path" in frame_data and os.path.exists(frame_data["path"]):
                        # Try OCR again with different prompt/approach
                        text = self._detect_text_with_retry(
                            frame_data["path"], 
                            alternate_prompt=True
                        )
                        
                        # Update the frame data
                        frame_data["text"] = text
                        frame_data["processed_successfully"] = bool(
                            text and len(text.strip()) >= self.min_frame_text_length
                        )
                        frame_data["was_retry"] = True
                        
                        # Update progress
                        prog.update(retry_task, advance=1)
                        
                        # Show intermittent stats
                        if (idx + 1) % 5 == 0 or idx == len(failed_frames) - 1:
                            successful_retries = sum(1 for f in failed_frames[:idx+1] 
                                                   if f.get("processed_successfully", False))
                            self.console.print(
                                f"[dim]Retry progress: {idx+1}/{len(failed_frames)}, "
                                f"{successful_retries} recovered[/]"
                            )
                    else:
                        # Can't retry without path
                        prog.update(retry_task, advance=1)
            
            # Report retry results
            successful_retries = sum(1 for f in failed_frames if f.get("processed_successfully", False))
            self.console.print(
                f"[{'green' if successful_retries else 'yellow'}]"
                f"Retries recovered {successful_retries}/{len(failed_frames)} frames[/]"
            )

        # ── 3. Filter and sort results ───────────────────────────────────
        # Only keep frames that have meaningful text (filter out empty or too-short results)
        self.results = [
            r for r in processed_frames 
            if r.get("processed_successfully", False)
        ]
        
        # Sort chronologically
        self.results.sort(key=lambda r: r["frame_num"])
        
        # Save raw results
        results_path = os.path.join(self.output_dir, "results.json")
        with open(results_path, "w", encoding="utf-8") as fp:
            json.dump(self.results, fp, indent=2)
            
        self.console.print(f"[green]Found text in {len(self.results)}/{len(distinct)} processed frames[/]")

        # ── 4. Optional: name extraction ────────────────────────────────
        names_json: Dict[str, List] = {"names": []}
        if self.do_extract_names and self.results:
            self.console.print("\n[bold]Extracting names from text…[/]")
            
            # Prepare text data for name extraction
            text_data = [
                {"timestamp": r["timestamp"], "text": r["text"]} 
                for r in self.results
            ]
            
            # Extract names with retry logic
            names_json = self._extract_names_with_retry(text_data)
            
            # Save names results
            with open(os.path.join(self.output_dir, "names.json"), "w", encoding="utf-8") as fp:
                json.dump(names_json, fp, indent=2)

        # ── 5. Comprehensive summary ─────────────────────────────────────
        elapsed = time.time() - t_start
        summary = {
            "video_info": {
                "filename": os.path.basename(self.video_path),
                "resolution": f"{self.width}×{self.height}",
                "fps": self.fps,
                "total_frames": self.frame_count,
                "duration_seconds": self.frame_count / self.fps,
                "duration_formatted": str(timedelta(seconds=int(self.frame_count / self.fps))),
            },
            "processing_stats": {
                "sample_rate": self.sample_rate,
                "similarity_threshold": self.similarity_threshold,
                "distinct_frames_processed": len(distinct),
                "frames_with_text": len(self.results),
                "processing_time_seconds": elapsed,
                "processing_time_formatted": str(timedelta(seconds=int(elapsed))),
            },
            "api_stats": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "rate_limited_requests": self.metrics.rate_limited_requests,
                "total_tokens": self.metrics.total_tokens,
                "prompt_tokens": self.metrics.prompt_tokens,
                "completion_tokens": self.metrics.completion_tokens,
                "estimated_cost_usd": self.metrics.total_cost,
            },
            "names": names_json.get("names", []),
        }
        
        with open(os.path.join(self.output_dir, "summary.json"), "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)

        # ── 6. Display results ──────────────────────────────────────────
        # Video info table
        video_tbl = Table(title="Video Information")
        video_tbl.add_column("Property", style="cyan")
        video_tbl.add_column("Value", style="green")
        for k, v in summary["video_info"].items():
            video_tbl.add_row(k.replace("_", " ").title(), str(v))
        self.console.print(video_tbl)
        
        # Processing stats table
        proc_tbl = Table(title="Processing Statistics")
        proc_tbl.add_column("Metric", style="cyan")
        proc_tbl.add_column("Value", style="green")
        for k, v in summary["processing_stats"].items():
            proc_tbl.add_row(k.replace("_", " ").title(), f"{v:.2f}" if isinstance(v, float) else str(v))
        self.console.print(proc_tbl)
        
        # API usage table
        api_tbl = Table(title="API Usage")
        api_tbl.add_column("Metric", style="cyan")
        api_tbl.add_column("Value", style="green")
        for k, v in summary["api_stats"].items():
            api_tbl.add_row(k.replace("_", " ").title(), f"${v:.4f}" if k == "estimated_cost_usd" else str(v))
        self.console.print(api_tbl)

        # Names table (if any)
        if summary["names"]:
            ntable = Table(title="Names Detected")
            ntable.add_column("Name", style="cyan")
            ntable.add_column("First Seen", style="green")
            ntable.add_column("Last Seen", style="green")
            ntable.add_column("Count", style="green")
            for n in summary["names"]:
                ntable.add_row(n["name"], n["first_appearance"], n["last_appearance"], str(n["count"]))
            self.console.print(ntable)
        else:
            self.console.print("[yellow]No names detected in the video.[/]")
        # ── 7. Final output ─────────────────────────────────────────────
        self.console.print(
            f"[green]Results saved to:[/] {os.path.relpath(self.output_dir)}\n"
            f"  - results.json: Raw OCR results\n"
            f"  - summary.json: Complete processing statistics\n" +
            (f"  - names.json: Extracted names\n" if self.do_extract_names else "") +
            (f"  - frames/: {len(distinct)} extracted video frames\n" if self.save_frames else "")
        )
        self.console.print("Process completed successfully!")
        
        return self.results


# ── CLI entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Advanced Video OCR & name extraction using LLaMA 4 Maverick via OpenRouter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video_path", help="Input video file (mp4, mov …)")
    parser.add_argument("output_dir", help="Directory where results are written")
    parser.add_argument("api_key_file", help="JSON with { 'openrouter_api_key': '…' }")
    parser.add_argument("--sample-rate", type=int, default=5, help="Sample every Nth frame")
    parser.add_argument("--similarity-threshold", type=float, default=0.9, help="Duplicate‑frame threshold [0‑1]")
    parser.add_argument("--workers", dest="max_workers", type=int, default=5, help="Thread workers for OCR calls")
    parser.add_argument("--no-save-frames", dest="save_frames", action="store_false", help="Do not write frame images")
    parser.add_argument(
        "--skip-names",
        dest="extract_names",
        action="store_false",
        help="Skip the (slower) name‑extraction step",
    )
    parser.add_argument(
        "--min-text-length", 
        type=int, 
        default=3, 
        help="Minimum text length to keep a frame in results"
    )
    parser.add_argument(
        "--initial-rpm", 
        type=int, 
        default=30, 
        help="Initial requests per minute limit"
    )
    parser.add_argument(
        "--max-rpm",
        type=int,
        default=50,
        help="Maximum requests per minute limit"
    )
    parser.add_argument(
        "--min-rpm",
        type=int,
        default=5,
        help="Minimum requests per minute limit"
    )
    parser.add_argument(
        "--backoff-factor",
        type=float,
        default=0.8,
        help="Factor to reduce RPM by when rate limited (0-1)"
    )
    parser.add_argument(
        "--recovery-factor",
        type=float,
        default=1.05,
        help="Factor to increase RPM by after success streak"
    )
    args = parser.parse_args()

    # Display header
    console = Console()
    console.print(
        Panel.fit(
            "[bold blue]Video OCR & Name Extraction[/]\n"
            "[cyan]Using LLaMA 4 Maverick via OpenRouter with adaptive rate limiting[/]",
            title="🎬 VideoProcessor"
        )
    )

    try:
        vp = VideoProcessor(
            args.video_path,
            args.output_dir,
            args.api_key_file,
            sample_rate=args.sample_rate,
            similarity_threshold=args.similarity_threshold,
            max_workers=args.max_workers,
            save_frames=args.save_frames,
            extract_names=args.extract_names,
            min_frame_text_length=args.min_text_length,
        )
        
        # Configure rate limiter based on CLI args
        vp.rate_limiter.initial_requests_per_minute = args.initial_rpm
        vp.rate_limiter.current_rpm = float(args.initial_rpm)
        vp.rate_limiter.min_requests_per_minute = args.min_rpm
        vp.rate_limiter.max_requests_per_minute = args.max_rpm
        vp.rate_limiter.backoff_factor = args.backoff_factor
        vp.rate_limiter.recovery_factor = args.recovery_factor
        
        # Process the video
        vp.process_video()
        
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)