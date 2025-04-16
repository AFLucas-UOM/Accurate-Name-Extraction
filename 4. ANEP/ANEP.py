import cv2
import pytesseract
from pytesseract import Output
import spacy
import sys
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import numpy as np
from datetime import datetime
import torch
from ultralytics import YOLO
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.console import Console
from rich.logging import RichHandler
from transformers import pipeline
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any, Union
import argparse
from pathlib import Path
from gliner_spacy.pipeline import GlinerSpacy

# Attempt to import the GliNER spaCy component (assuming it is available as gliner_spacy)
try:
    from gliner_spacy.pipeline import GlinerSpacy
except ImportError:
    GlinerSpacy = None  # If not available, the code will log a warning later

class NewsGraphicsNameDetector:
    """Main class for the News Graphics Name Detection Pipeline using GliNER for zero-shot NER."""
    
    @dataclass
    class DetectionResult:
        """Data class for storing detection results."""
        timestamp: float
        frame_idx: int
        class_name: str
        roi_path: str
        text: str
        names: List[Dict[str, Any]]
        
        def to_dict(self):
            return asdict(self)
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the pipeline with optional configuration."""
        # Set up default configuration
        self.config = {
            "tesseract_cmd": '/opt/homebrew/bin/tesseract',
            "confidence_threshold": 0.6,
            "iou_threshold": 0.5,
            "ocr_conf_threshold": 40,
            "transformer_conf_threshold": 0.85,
            "yolo_model_path": "best.pt",
            "spacy_model": "en_core_web_md",
            "transformer_model": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "valid_classes": {
                "Breaking News Graphic",
                "Digital On-Screen Graphic",
                "Lower Third Graphic",
                "News Headline",
                "News Ticker",
                "Other News Graphic"
            }
        }
        
        # Override with user-provided configuration (if any)
        if config:
            self.config.update(config)
        
        # Setup console and logging
        self.setup_console_and_logging()
        
        # Create directories for output
        self.create_directories()
        
        # Initialize models and pipelines
        self.initialize_models()
    
    def setup_console_and_logging(self):
        """Set up the console and logging handlers."""
        self.console = Console()
        FORMAT = "%(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=FORMAT,
            datefmt="[%X]",
            handlers=[RichHandler(console=self.console, rich_tracebacks=True)]
        )
        self.timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        LOG_DIR = Path("logs")
        LOG_DIR.mkdir(exist_ok=True)
        log_filename = LOG_DIR / f"extraction_{self.timestamp_str}.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger = logging.getLogger()
        self.logger.addHandler(file_handler)
        self.log_filename = log_filename
        self.console.print("[bold green]Starting Accurate Name Extraction Pipeline (ANEP)[bold green]")
        self.logger.info("Starting Accurate Name Extraction Pipeline (ANEP)[")
    
    def create_directories(self):
        """Create necessary directories for outputs."""
        self.ROI_DIR = Path("regions_of_interest") / self.timestamp_str
        self.OCR_TEXT_DIR = Path("ocr_text") / self.timestamp_str
        self.NAMES_DIR = Path("detected_names") / self.timestamp_str
        self.RESULTS_DIR = Path("results") / self.timestamp_str
        for directory in [self.ROI_DIR, self.OCR_TEXT_DIR, self.NAMES_DIR, self.RESULTS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
    
    def initialize_models(self):
        """Initialize all ML models used in the pipeline."""
        # Determine device (CUDA, MPS, or CPU)
        if torch.cuda.is_available():
            self.device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            self.logger.info(f"Using GPU: {device_name}")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.logger.info("Using Apple MPS")
        else:
            self.device = "cpu"
            self.logger.info("Using CPU")
        
        self.ner_device = 0 if torch.cuda.is_available() else -1
        
        # Configure Tesseract
        try:
            pytesseract.pytesseract.tesseract_cmd = self.config["tesseract_cmd"]
            self.logger.info(f"Tesseract configured at {pytesseract.pytesseract.tesseract_cmd}")
        except Exception as e:
            self.logger.error(f"Failed to configure Tesseract: {e}")
            raise RuntimeError(f"Tesseract configuration failed: {e}")
        
        # Load and enhance spaCy model with GliNER (if available)
        try:
            self.console.print("[yellow]Loading SpaCy model...[/yellow]")
            self.nlp = spacy.load(self.config["spacy_model"])
            self.logger.info("SpaCy model loaded successfully")
            if GlinerSpacy:
                # Add GliNER as a custom component configured to extract only person entities
                self.nlp.add_pipe("gliner_spacy", config={"labels": ["person"]})
                self.logger.info("GliNER component added to the spaCy pipeline for zero-shot person NER")
            else:
                self.logger.warning("GliNER not found; using default spaCy NER")
        except Exception as e:
            self.logger.error(f"Failed to load SpaCy model: {e}")
            raise RuntimeError(f"SpaCy model loading failed: {e}")
        
        # Load YOLO model
        try:
            self.console.print("[yellow]Loading YOLO model...[/yellow]")
            self.yolo_model = YOLO(self.config["yolo_model_path"])
            self.logger.info("YOLO model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"YOLO model loading failed: {e}")
        
        # Load transformer NER model (fallback / ensemble method)
        try:
            self.console.print("[yellow]Loading Transformer NER model...[/yellow]")
            self.transformer_pipeline = pipeline(
                "ner", 
                model=self.config["transformer_model"], 
                device=self.ner_device,
                aggregation_strategy="simple"
            )
            self.logger.info("Transformer model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load transformer model: {e}")
            raise RuntimeError(f"Transformer model loading failed: {e}")
    
    def calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1, w2, h2 = box2
        x1_2, y1_2 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x2_1 + w2, y2_1 + h2
        x_left = max(x1_1, x2_1)
        y_top = max(y1_1, y2_1)
        x_right = min(x1_2, x2_2)
        y_bottom = min(y1_2, y2_2)
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def non_max_suppression(self, boxes: List[Tuple[int, int, int, int]], 
                            scores: List[float], 
                            iou_threshold: float = 0.5) -> List[int]:
        """Apply Non-Maximum Suppression to remove overlapping boxes."""
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        keep_indices = []
        while indices:
            current = indices[0]
            keep_indices.append(current)
            indices = [i for i in indices[1:] if self.calculate_iou(boxes[current], boxes[i]) < iou_threshold]
        return keep_indices
    
    def extract_regions_of_interest(self, frame: np.ndarray, 
                                    frame_idx: Optional[int] = None) -> List[Dict[str, Any]]:
        """Extract regions of interest (ROIs) using YOLO detection and non-max suppression."""
        self.logger.debug(f"Extracting ROIs from frame {frame_idx if frame_idx is not None else ''}")
        results = self.yolo_model.predict(
            frame, 
            conf=self.config["confidence_threshold"], 
            iou=0.4, 
            verbose=False
        )[0]
        
        boxes, scores, class_names = [], [], []
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.yolo_model.names[class_id]
            if class_name in self.config["valid_classes"] and confidence >= self.config["confidence_threshold"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                boxes.append((x1, y1, w, h))
                scores.append(confidence)
                class_names.append(class_name)
        
        if boxes:
            keep_indices = self.non_max_suppression(
                boxes, 
                scores, 
                iou_threshold=self.config["iou_threshold"]
            )
            filtered_detections = []
            for i in keep_indices:
                x, y, w, h = boxes[i]
                cropped = frame[y:y+h, x:x+w]
                filtered_detections.append({
                    "class_name": class_names[i],
                    "confidence": scores[i],
                    "bbox": boxes[i],
                    "image": cropped
                })
            self.logger.debug(f"Found {len(filtered_detections)} ROIs after non-max suppression")
            return filtered_detections
        
        return []
    
    def save_roi_image(self, image: np.ndarray, 
                       bbox: Tuple[int, int, int, int], 
                       frame_idx: int, 
                       timestamp: float, 
                       class_name: str, 
                       confidence: float) -> Optional[str]:
        """Save the cropped ROI image to disk."""
        x, y, w, h = bbox
        filename = self.ROI_DIR / f"frame_{frame_idx}_roi_{x}_{y}_{timestamp:.2f}_{class_name}_{confidence:.2f}.png"
        try:
            cv2.imwrite(str(filename), image)
            self.logger.debug(f"Saved ROI image: {filename}")
            return str(filename)
        except Exception as e:
            self.logger.error(f"Error saving ROI image: {e}")
            return None
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Advanced preprocessing to improve OCR quality."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if w < 300 or h < 300:
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        combined = cv2.bitwise_or(adaptive_thresh, otsu_thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        return opened
    
    def extract_text_from_roi(self, image: np.ndarray, 
                              roi_path: str, 
                              frame_idx: int) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract text from the ROI using Tesseract OCR."""
        preprocessed = self.preprocess_for_ocr(image)
        try:
            ocr_data = pytesseract.image_to_data(
                preprocessed, 
                config='--oem 3 --psm 6 -l eng', 
                output_type=Output.DICT
            )
        except Exception as e:
            self.logger.error(f"OCR error: {e}")
            return "", []
        
        extracted_text = []
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            if text:
                try:
                    conf = float(ocr_data['conf'][i])
                    if conf > self.config["ocr_conf_threshold"]:
                        extracted_text.append({
                            "text": text,
                            "confidence": conf,
                            "bbox": (
                                ocr_data['left'][i], 
                                ocr_data['top'][i], 
                                ocr_data['width'][i], 
                                ocr_data['height'][i]
                            ),
                            "block_num": ocr_data['block_num'][i],
                            "line_num": ocr_data['line_num'][i],
                            "word_num": ocr_data['word_num'][i]
                        })
                except ValueError:
                    continue
        
        lines = {}
        for word in extracted_text:
            key = (word['block_num'], word['line_num'])
            lines.setdefault(key, []).append(word)
        full_text = []
        for key, words in sorted(lines.items()):
            words_sorted = sorted(words, key=lambda w: w['bbox'][0])
            line_text = " ".join(word["text"] for word in words_sorted)
            full_text.append(line_text)
        complete_text = "\n".join(full_text)
        
        base_name = os.path.basename(roi_path).split('.')[0]
        text_file_path = self.OCR_TEXT_DIR / f"{base_name}_text.txt"
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(complete_text)
        
        self.logger.debug(f"Extracted {len(full_text)} lines of text from ROI and saved to {text_file_path}")
        self.console.print(f"[cyan]Text from frame {frame_idx}:[/cyan]")
        self.console.print(complete_text)
        return complete_text, extracted_text

    def normalize_name(self, name: str) -> str:
        """Normalize a detected name (e.g. convert all-uppercase names to title case)."""
        return " ".join(word.capitalize() for word in name.split())

    def is_valid_name(self, text: str) -> bool:
        """Perform checks to filter out invalid person names.
        
        Now also requiring at least two words, so that spurious OCR detections are reduced.
        """
        text = text.strip()
        if len(text.split()) < 2:
            return False
        if len(text) < 2 or len(text) > 40:
            return False
        if re.search(r'\d', text):
            return False
        if re.search(r'[^\w\s\'\.\-]', text):
            return False
        if not text[0].isupper():
            return False
        common_words = {
            "the", "and", "news", "breaking", "update", "live", "report", 
            "today", "exclusive", "just", "in", "now", "watch"
        }
        if text.lower() in common_words:
            return False
        return True
    
    def is_really_person(self, name: str) -> bool:
        """Validate a candidate name using the transformer NER pipeline."""
        if not isinstance(name, str) or len(name.strip()) == 0:
            return False
        try:
            entities = self.transformer_pipeline(name)
            return any(entity.get('entity_group', None) == 'PER' for entity in entities)
        except Exception as e:
            self.logger.warning(f"Error validating name '{name}': {e}")
            return False
    
    def extract_names_with_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Extract person names using the spaCy pipeline (now enhanced with GliNER)."""
        if not text.strip():
            return []
        doc = self.nlp(text)
        names = []
        # Assuming GliNER outputs labels as "PERSON" or "person", filter accordingly
        for ent in doc.ents:
            if (ent.label_.lower() == "person") and self.is_valid_name(ent.text):
                normalized_name = self.normalize_name(ent.text)
                names.append({
                    "name": normalized_name,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "method": "gliner-spacy"
                })
        return names
    
    def extract_names_with_transformer(self, text: str) -> List[Dict[str, Any]]:
        """Extract person names using a transformer-based NER model."""
        if not text.strip():
            return []
        try:
            results = self.transformer_pipeline(text)
            names = []
            for result in results:
                score = float(result.get('score', 0))
                if result.get('entity_group') == 'PER' and score > self.config["transformer_conf_threshold"]:
                    name = self.normalize_name(result['word'])
                    if self.is_valid_name(name):
                        names.append({
                            "name": name,
                            "score": score,
                            "start": int(result['start']),
                            "end": int(result['end']),
                            "method": "transformer"
                        })
            return names
        except Exception as e:
            self.logger.warning(f"Transformer NER error: {e}")
            return []
    
    def combine_extracted_names(self, spacy_names: List[Dict[str, Any]], transformer_names: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine and aggregate names detected by both the GliNER-enhanced spaCy pipeline and the transformer."""
        combined = {}
        for name_obj in spacy_names:
            norm = name_obj["name"]
            if norm in combined:
                combined[norm]["methods"].add("gliner-spacy")
            else:
                combined[norm] = {"name": norm, "methods": {"gliner-spacy"}, "score": 1.0}
        for name_obj in transformer_names:
            norm = name_obj["name"]
            score = name_obj.get("score", 1.0)
            if norm in combined:
                combined[norm]["methods"].add("transformer")
                combined[norm]["score"] = max(combined[norm]["score"], score)
            else:
                combined[norm] = {"name": norm, "methods": {"transformer"}, "score": score}
        validated = []
        for candidate in combined.values():
            candidate["methods"] = list(candidate["methods"])
            if self.is_valid_name(candidate["name"]) and self.is_really_person(candidate["name"]):
                validated.append(candidate)
        return validated
    
    def process_roi(self, roi_obj: Dict[str, Any], 
                    frame_idx: int, 
                    timestamp: float) -> Optional[DetectionResult]:
        """Process a single ROI: save image, extract text, and detect person names."""
        image = roi_obj["image"]
        class_name = roi_obj["class_name"]
        confidence = roi_obj["confidence"]
        bbox = roi_obj["bbox"]
        
        roi_path = self.save_roi_image(image, bbox, frame_idx, timestamp, class_name, confidence)
        if not roi_path:
            return None
        
        text, ocr_results = self.extract_text_from_roi(image, roi_path, frame_idx)
        if not text:
            return None
        
        spacy_names = self.extract_names_with_spacy(text)
        transformer_names = self.extract_names_with_transformer(text)
        validated_names = self.combine_extracted_names(spacy_names, transformer_names)
        
        if validated_names:
            self.logger.info(f"Detected {len(validated_names)} valid names in ROI from frame {frame_idx}")
            base_name = os.path.basename(roi_path).split('.')[0]
            names_file_path = self.NAMES_DIR / f"{base_name}_names.json"
            with open(names_file_path, 'w', encoding='utf-8') as f:
                json.dump(validated_names, f, indent=2)
            self.console.print(f"[green]Names detected in frame {frame_idx}:[/green]")
            for name_obj in validated_names:
                method = name_obj.get("methods", ["unknown"])
                conf_val = name_obj.get("score", 1.0)
                self.console.print(f"  - {name_obj['name']} (Methods: {', '.join(method)}, Confidence: {conf_val:.2f})")
            return self.DetectionResult(
                timestamp=timestamp,
                frame_idx=frame_idx,
                class_name=class_name,
                roi_path=roi_path,
                text=text,
                names=validated_names
            )
        return None
    
    def process_video(self, video_path: str, 
                      sampling_rate: float = 1.0, 
                      max_frames: Optional[int] = None) -> List[DetectionResult]:
        """Process the video: sample frames, detect ROIs, and extract person names."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        self.logger.info(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
        self.console.print(f"[bold]Video info:[/bold] {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
        
        interval = int(fps * sampling_rate)
        frames_to_process = total_frames // interval + (1 if total_frames % interval else 0)
        if max_frames is not None and max_frames < frames_to_process:
            frames_to_process = max_frames
            self.logger.info(f"Limited to processing {max_frames} frames")
        
        self.logger.info(f"Will process {frames_to_process} frames at {sampling_rate}s intervals")
        self.console.print(f"Will process [bold]{frames_to_process}[/bold] frames at {sampling_rate}s intervals")
        
        results = []
        frame_idx = 0
        processed_count = 0
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Processing video...", total=frames_to_process)
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = []
                while cap.isOpened() and (max_frames is None or processed_count < max_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % interval == 0:
                        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        self.logger.debug(f"Processing frame {frame_idx}/{total_frames} at {timestamp:.2f}s")
                        roi_regions = self.extract_regions_of_interest(frame, frame_idx)
                        if roi_regions:
                            self.logger.info(f"Found {len(roi_regions)} ROIs in frame {frame_idx}")
                            for roi in roi_regions:
                                futures.append(
                                    executor.submit(self.process_roi, roi, frame_idx, timestamp)
                                )
                        processed_count += 1
                        progress.update(task, advance=1)
                    frame_idx += 1
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        self.logger.error(f"Error processing ROI: {e}")
        cap.release()
        results_file = self.RESULTS_DIR / f"name_detections_{self.timestamp_str}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        self.logger.info(f"Processed {processed_count} frames, saved results to {results_file}")
        self.console.print(f"[bold green]Processed {processed_count} frames, detected {len(results)} name instances[/bold green]")
        return results
    
    def generate_summary(self, results: List[DetectionResult]) -> Dict[str, Any]:
        """Generate a summary of detected names from all processed ROIs."""
        if not results:
            self.logger.warning("No results to summarize")
            return {"unique_names": 0, "total_instances": 0, "names": {}}
        
        all_names = []
        for result in results:
            for name_obj in result.names:
                all_names.append(name_obj.get("name"))
        name_counts = {}
        for name in all_names:
            name_counts[name] = name_counts.get(name, 0) + 1
        sorted_names = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)
        self.console.print("\n[bold]Name Detection Summary:[/bold]")
        self.console.print(f"Total detected name instances: {len(all_names)}")
        self.console.print(f"Unique names detected: {len(name_counts)}")
        self.console.print("\n[bold]Top detected names:[/bold]")
        for name, count in sorted_names[:10]:
            self.console.print(f"  {name}: {count} instances")
        return {
            "unique_names": len(name_counts),
            "total_instances": len(all_names),
            "names": dict(sorted_names)
        }
    
    def run(self, video_path: str, sampling_rate: float = 1.0, max_frames: Optional[int] = None) -> Dict[str, Any]:
        """Run the complete pipeline on a video file."""
        self.console.print(f"[bold green]Processing video:[/bold green] {video_path}")
        results = self.process_video(video_path, sampling_rate, max_frames)
        summary = self.generate_summary(results)
        self.console.print("\n[bold]Output directories:[/bold]")
        self.console.print(f"Regions of interest: {self.ROI_DIR}")
        self.console.print(f"OCR text files: {self.OCR_TEXT_DIR}")
        self.console.print(f"Detected names: {self.NAMES_DIR}")
        self.console.print(f"Results JSON: {self.RESULTS_DIR}")
        self.console.print(f"Log file: {self.log_filename}")
        return {
            "results": [r.to_dict() for r in results],
            "summary": summary,
            "dirs": {
                "roi": str(self.ROI_DIR),
                "ocr": str(self.OCR_TEXT_DIR),
                "names": str(self.NAMES_DIR),
                "results": str(self.RESULTS_DIR),
                "log": str(self.log_filename)
            }
        }

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='News Graphics Name Detection Pipeline')
    parser.add_argument('--video', '-v', type=str, default='Video_T4.mp4',
                        help='Path to the video file to process')
    parser.add_argument('--sampling_rate', '-s', type=float, default=1.0,
                        help='Sampling rate in seconds (default: 1.0)')
    parser.add_argument('--max_frames', '-m', type=int, default=None,
                        help='Maximum number of frames to process (default: all)')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to JSON configuration file')
    return parser.parse_args()


def main():
    """Main function to run the pipeline."""
    args = parse_arguments()
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            sys.exit(1)
    video_path = args.video
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    try:
        detector = NewsGraphicsNameDetector(config)
        detector.run(video_path, args.sampling_rate, args.max_frames)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
