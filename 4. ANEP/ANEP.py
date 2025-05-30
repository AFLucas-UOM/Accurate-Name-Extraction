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
import xxhash
from typing import List, Dict, Tuple, Optional, Any, Union, Set
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.console import Console
from rich.logging import RichHandler
from transformers import pipeline
from ultralytics import YOLO
from rapidfuzz import fuzz
from typing import Dict, Any

# LRU Cache for similar frames
from functools import lru_cache

# Try to import GliNER spaCy component
try:
    from gliner_spacy.pipeline import GlinerSpacy
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False

# Configure LRU cache size - can be adjusted based on memory constraints
FRAME_CACHE_SIZE = 100
SIMILARITY_THRESHOLD = 0.95  # Threshold for frame similarity (0.0-1.0)
CONTIGUOUS_SKIP_THRESHOLD = 3  # Skip processing after this many similar frames in a row

# Disable parallelism for tokenizers to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

class NewsGraphicsNameDetector:
    """Enhanced News Graphics Name Detection Pipeline with deduplication and improved accuracy."""
    
    @dataclass
    class DetectionResult:
        """Data class for storing detection results."""
        timestamp: float
        frame_idx: int
        class_name: str
        roi_path: str
        text: str
        names: List[Dict[str, Any]]
        confidence: float
        hash_value: str = ""
        
        def to_dict(self):
            return asdict(self)
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the pipeline with optional configuration."""
        # Set up default configuration
        self.project_root = Path.cwd()
        self.config = {
            "tesseract_cmd": '/opt/homebrew/bin/tesseract',
            "confidence_threshold": 0.6,
            "iou_threshold": 0.5,
            "ocr_conf_threshold": 40,
            "transformer_conf_threshold": 0.85,
            "yolo_model_path": str(self.project_root / "3. Yolo Training/runs/detect/NGD-Yolov12_v5/weights/best.pt"),
            "spacy_model": "en_core_web_md",
            "transformer_model": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "contiguous_skip_threshold": CONTIGUOUS_SKIP_THRESHOLD,
            "roi_padding": 5,  # Added padding to ROIs for better context
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
        
        # Initialize frame caching system
        self.frame_hashes = {}
        self.duplicate_frames_skipped = 0
        self.contiguous_similar_frames = defaultdict(int)
        self.last_processed_hashes = {}
        
        # Initialize tracking for unique names
        self.unique_names = set()
        self.name_instances = defaultdict(int)
        self.name_timestamps = defaultdict(list)
        self.total_detections = 0
    
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
        LOG_DIR = Path.cwd() / "4. ANEP/logs"
        LOG_DIR.mkdir(exist_ok=True)
        log_filename = LOG_DIR / f"extraction_{self.timestamp_str}.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger = logging.getLogger()
        self.logger.addHandler(file_handler)
        self.log_filename = log_filename
        self.console.print("[bold green]Starting Enhanced Name Extraction Pipeline (ANEP)[bold green]")
        self.logger.info("Starting Enhanced Name Extraction Pipeline (ANEP)")
    
    def create_directories(self):
        """Create necessary directories for outputs."""
        self.ROI_DIR = Path.cwd() / "4. ANEP/regions_of_interest" / self.timestamp_str
        self.OCR_TEXT_DIR = Path.cwd() / "4. ANEP/ocr_text" / self.timestamp_str
        self.NAMES_DIR = Path.cwd() / "4. ANEP/detected_names" / self.timestamp_str
        self.RESULTS_DIR = Path.cwd() / "4. ANEP/results" / self.timestamp_str
        self.DEDUPLICATED_DIR = Path.cwd() / "4. ANEP/deduplicated_frames" / self.timestamp_str
        for directory in [self.ROI_DIR, self.OCR_TEXT_DIR, self.NAMES_DIR, self.RESULTS_DIR, self.DEDUPLICATED_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
    
    def initialize_models(self):
        """Initialize all ML models used in the pipeline."""
        # Determine device (CUDA, MPS, or CPU)
        if torch.cuda.is_available():
            self.device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            self.logger.info(f"Using GPU: {device_name}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
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
            if GLINER_AVAILABLE:
                # Add GliNER as a custom component configured to extract only person entities
                self.nlp.add_pipe("gliner_spacy", config={"labels": ["person"]})
                self.logger.info("GliNER component added to the spaCy pipeline for zero-shot PERSON NER")
            else:
                self.logger.warning("GliNER not found; using default spaCy NER")
        except Exception as e:
            self.logger.error(f"Failed to load SpaCy model: {e}")
            raise RuntimeError(f"SpaCy model loading failed: {e}")
        
        # Load YOLO model
        try:
            self.console.print("[yellow]Loading YOLO model...[/yellow]")
            self.yolo_model = YOLO(self.config["yolo_model_path"])
            self.logger.info("YOLOv12 model loaded successfully")
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
    
    def compute_frame_hash(self, frame: np.ndarray) -> str:
        """Compute a hash for a frame using xxhash for fast comparison."""
        return xxhash.xxh64(frame.tobytes()).hexdigest()
    
    def compute_perceptual_hash(self, image: np.ndarray) -> str:
        """Compute a perceptual hash for an image that's robust to minor changes."""
        # Convert to grayscale and resize to a small size for faster hashing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        resized = cv2.resize(gray, (32, 32))
        # Apply DCT (Discrete Cosine Transform)
        dct = cv2.dct(np.float32(resized))
        # Keep only the top-left 8x8 portion which contains most of the image's energy
        dct_low = dct[:8, :8]
        # Compute median value
        median = np.median(dct_low)
        # Create binary hash (1 if pixel value > median, else 0)
        hash_bits = (dct_low > median).flatten()
        # Convert to hexadecimal string
        hash_hex = "".join(['1' if b else '0' for b in hash_bits])
        return hash_hex
    
    def are_frames_similar(self, hash1: str, hash2: str) -> bool:
        """Compare two frame hashes to determine if frames are similar."""
        if not hash1 or not hash2:
            return False
        # For perceptual hash, count bit differences
        differing_bits = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        similarity = 1 - differing_bits / len(hash1)
        return similarity >= self.config["similarity_threshold"]
    
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
    
    def add_padding_to_roi(self, x: int, y: int, w: int, h: int, frame_height: int, frame_width: int) -> Tuple[int, int, int, int]:
        """Add padding to ROI for better context."""
        padding = self.config["roi_padding"]
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame_width - x, w + 2 * padding)
        h = min(frame_height - y, h + 2 * padding)
        return x, y, w, h
    
    def extract_regions_of_interest(self, frame: np.ndarray, 
                                    frame_idx: Optional[int] = None) -> List[Dict[str, Any]]:
        """Extract regions of interest (ROIs) using YOLO detection and non-max suppression."""
        self.logger.debug(f"Extracting ROIs from frame {frame_idx if frame_idx is not None else ''}")
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
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
                # Add padding to ROI for better context
                padded_x, padded_y, padded_w, padded_h = self.add_padding_to_roi(x, y, w, h, frame_height, frame_width)
                cropped = frame[padded_y:padded_y+padded_h, padded_x:padded_x+padded_w]
                # Compute hash for the cropped ROI
                roi_hash = self.compute_perceptual_hash(cropped)
                
                # Check if we have seen a similar ROI in the last N frames
                roi_key = f"{class_names[i]}_{padded_x}_{padded_y}_{padded_w}_{padded_h}"
                
                # Keep track of contiguous similar frames for this ROI
                if roi_key in self.last_processed_hashes:
                    last_hash = self.last_processed_hashes[roi_key]
                    if self.are_frames_similar(roi_hash, last_hash):
                        self.contiguous_similar_frames[roi_key] += 1
                    else:
                        self.contiguous_similar_frames[roi_key] = 0
                        self.last_processed_hashes[roi_key] = roi_hash
                else:
                    self.contiguous_similar_frames[roi_key] = 0
                    self.last_processed_hashes[roi_key] = roi_hash
                
                # Skip processing if we've seen too many similar frames in a row
                if self.contiguous_similar_frames[roi_key] >= self.config["contiguous_skip_threshold"]:
                    self.duplicate_frames_skipped += 1
                    self.logger.debug(f"Skipping similar ROI in frame {frame_idx}, ROI key: {roi_key}")
                    continue
                
                filtered_detections.append({
                    "class_name": class_names[i],
                    "confidence": scores[i],
                    "bbox": (padded_x, padded_y, padded_w, padded_h),
                    "original_bbox": boxes[i],
                    "image": cropped,
                    "hash": roi_hash
                })
            self.logger.debug(f"Found {len(filtered_detections)} ROIs after non-max suppression and duplication check")
            return filtered_detections
        
        return []
    
    def should_skip_frame(self, frame: np.ndarray, frame_idx: int) -> bool:
        """Determine if a frame should be skipped based on similarity to previous frames."""
        # Compute hash for current frame
        frame_hash = self.compute_frame_hash(frame)
        
        # Check if frame is similar to any previously processed frame
        for prev_idx, prev_hash in list(self.frame_hashes.items()):
            if abs(prev_idx - frame_idx) <= 30:  # Only check nearby frames
                if frame_hash == prev_hash:
                    self.duplicate_frames_skipped += 1
                    self.logger.debug(f"Skipping duplicate frame {frame_idx} (identical to frame {prev_idx})")
                    return True
        
        # Store hash for future comparisons
        self.frame_hashes[frame_idx] = frame_hash
        return False
    
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
    
    def preprocess_for_ocr(self, image: np.ndarray) -> List[np.ndarray]:
        """Advanced preprocessing to improve OCR quality by trying multiple techniques."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        if w < 300 or h < 300:
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # Apply multiple preprocessing techniques
        preprocessed_images = []
        
        # Method 1: CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        preprocessed_images.append(adaptive_thresh)
        
        # Method 2: Otsu thresholding
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(otsu_thresh)
        
        # Method 3: Local thresholding
        local_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5
        )
        preprocessed_images.append(local_thresh)
        
        # Method 4: Original grayscale
        preprocessed_images.append(gray)
        
        # Method 5: Combined approach
        combined = cv2.bitwise_or(adaptive_thresh, otsu_thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        preprocessed_images.append(opened)
        
        return preprocessed_images
    
    def extract_text_from_roi(self, image: np.ndarray, 
                              roi_path: str, 
                              frame_idx: int) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract text from the ROI using Tesseract OCR with multiple preprocessing techniques."""
        preprocessed_images = self.preprocess_for_ocr(image)
        
        best_text = ""
        best_words = []
        best_confidence = 0
        
        for idx, preprocessed in enumerate(preprocessed_images):
            try:
                ocr_data = pytesseract.image_to_data(
                    preprocessed, 
                    config='--oem 3 --psm 6 -l eng', 
                    output_type=Output.DICT
                )
                
                extracted_text = []
                avg_conf = 0
                conf_count = 0
                
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
                                avg_conf += conf
                                conf_count += 1
                        except ValueError:
                            continue
                
                # Calculate average confidence
                avg_conf = avg_conf / conf_count if conf_count > 0 else 0
                
                if conf_count > 0 and (avg_conf > best_confidence or len(extracted_text) > len(best_words)):
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
                    
                    # Choose best text based on length and confidence
                    quality_score = len(complete_text) * avg_conf
                    current_best_score = len(best_text) * best_confidence if best_confidence > 0 else 0
                    
                    if quality_score > current_best_score:
                        best_text = complete_text
                        best_words = extracted_text
                        best_confidence = avg_conf
                
            except Exception as e:
                self.logger.error(f"OCR error with preprocessing method {idx}: {e}")
        
        if best_text:
            base_name = os.path.basename(roi_path).split('.')[0]
            text_file_path = self.OCR_TEXT_DIR / f"{base_name}_text.txt"
            with open(text_file_path, 'w', encoding='utf-8') as f:
                f.write(best_text)
            
            self.logger.debug(f"Extracted text from ROI and saved to {text_file_path}")
            # self.console.print(f"[cyan]Text from frame {frame_idx}:[/cyan]")
            # self.console.print(best_text)
            
            return best_text, best_words
        
        return "", []

    def normalize_name(self, name: str) -> str:
        """Normalize a detected name with improved handling of prefixes, suffixes, and formatting."""
        # Handle all uppercase names
        if name.isupper():
            name = " ".join(word.capitalize() for word in name.split())
        
        # Handle name prefixes and suffixes
        prefixes = [
            # Common Titles
            "Mr.", "Mrs.", "Ms.", "Miss", "Mx.",

            # Academic Titles
            "Dr.", "Prof.", "Dean",

            # Religious Titles
            "Rev.", "Fr.", "Br.", "Sr.", "Pr.", "Pope", "Rabbi", "Imam", "Sheikh", "Cardinal", "Archbishop",

            # Honorifics & Nobility
            "Hon.", "Sir", "Dame", "Lord", "Lady", "Baron", "Baroness", "Count", "Countess", "Viscount", "Marquess", "Duke", "Duchess",

            # Military & Government
            "Capt.", "Maj.", "Col.", "Gen.", "Lt.", "Sgt.", "Adm.", "Cmdr.", "Chief",
            "Judge", "Justice", "Pres.", "Gov.", "Amb.", "Sec.",

            # Other Professional or Formal
            "Engr.", "Arch.", "Atty.", "Supt.", "Chancellor", "Constable", "Inspector",
        ]

        suffixes = [
            "Jr.", "Sr.", "II", "III", "IV", "PhD", "MD", "Esq.", "DDS", "DVM", 
            "MBA", "CPA", "RN", "DO", "OD", "JD", "EdD", "LLD", "ThD", "PE",
            "Esquire", "Ret.", "CFA", "CM", "KC", "QC", "MP", "MLA", "MPP"
        ]
        
        # Ensure prefixes and suffixes are properly capitalized
        words = name.split()
        for i, word in enumerate(words):
            if word.lower() in (p.lower() for p in prefixes):
                words[i] = next(p for p in prefixes if p.lower() == word.lower())
            elif word.lower() in (s.lower() for s in suffixes):
                words[i] = next(s for s in suffixes if s.lower() == word.lower())
            elif any(word.lower().endswith(s.lower()) for s in [","]):
                words[i] = word[:-1]  # Remove trailing commas
        
        return " ".join(words)

    def is_valid_name(self, text: str) -> bool:
        """Perform enhanced checks to filter out invalid person names."""
        text = text.strip()
        
        # Basic validation checks
        if len(text.split()) < 2:
            return False
        if len(text) < 2 or len(text) > 40:
            return False
        if re.search(r'\d', text):
            return False
        
        # Allow appropriate punctuation in names (e.g., O'Reilly, Smith-Jones)
        if re.search(r'[^\w\s\'\.\-]', text):
            return False
        
        # First character should be uppercase
        if not text[0].isupper():
            return False
        
        # Filter out common words and phrases that are not names
        common_words = {
            "the", "and", "news", "breaking", "update", "live", "report", 
            "today", "exclusive", "just", "in", "now", "watch", "story",
            "latest", "headlines", "special", "report", "coverage", "top",
            "stories", "developing", "story", "information", "bulletin",
            "channel", "station", "network", "broadcast", "program", "show"
        }
        
        # Check if the name consists only of common words
        all_words = text.lower().split()
        if all(word in common_words for word in all_words):
            return False
        
        # Check for keywords that suggest the text is not a name
        news_phrases = [
            "breaking news", "live update", "special report", "news alert",
            "top stories", "latest news", "news channel", "news network",
            "developing story", "news update", "news briefing", "news conference"
        ]
        
        if any(phrase in text.lower() for phrase in news_phrases):
            return False
        
        return True
    
    def is_really_person(self, name: str) -> bool:
        """Enhanced validation for candidate names using multiple NER methods."""
        if not isinstance(name, str) or len(name.strip()) == 0:
            return False
        
        # Check with transformer NER pipeline
        try:
            transformer_result = False
            entities = self.transformer_pipeline(name)
            transformer_result = any(entity.get('entity_group', None) == 'PER' for entity in entities)
            
            # Also check with spaCy if available
            spacy_result = False
            doc = self.nlp(name)
            spacy_result = any(ent.label_ == "PERSON" for ent in doc.ents)
            
            # Consider it a person if either method identifies it as one
            return transformer_result or spacy_result
            
        except Exception as e:
            self.logger.warning(f"Error validating name '{name}': {e}")
            return False
    
    def extract_names_with_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Extract person names using the spaCy pipeline (enhanced with GliNER if available)."""
        if not text.strip():
            return []
        doc = self.nlp(text)
        names = []
        
        for ent in doc.ents:
            if (ent.label_.lower() == "person" or ent.label_ == "PERSON") and self.is_valid_name(ent.text):
                normalized_name = self.normalize_name(ent.text)
                names.append({
                    "name": normalized_name,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "method": "gliner-spacy" if GLINER_AVAILABLE else "spacy"
                })
        return names
    
    def extract_names_with_transformer(self, text: str) -> List[Dict[str, Any]]:
        """Extract person names using a transformer-based NER model with improved multi-token handling."""
        if not text.strip():
            return []
        try:
            results = self.transformer_pipeline(text)
            names = []
            
            # Group tokens to handle multi-token names
            current_name_parts = []
            current_score = 0
            current_start = 0
            current_end = 0
            
            for result in results:
                score = float(result.get('score', 0))
                entity_group = result.get('entity_group')
                
                if entity_group == 'PER' and score > self.config["transformer_conf_threshold"]:
                    word = result['word']
                    start = int(result['start'])
                    end = int(result['end'])
                    
                    # Check if this entity is part of a multi-token name (within 5 characters)
                    if current_name_parts and start <= current_end + 5:
                        # Add to current name
                        current_name_parts.append(word)
                        current_score = (current_score * len(current_name_parts) + score) / (len(current_name_parts) + 1)
                        current_end = end
                    else:
                        # If we have a current name being built, finalize it first
                        if current_name_parts:
                            full_name = " ".join(current_name_parts)
                            if self.is_valid_name(full_name):
                                normalized_name = self.normalize_name(full_name)
                                names.append({
                                    "name": normalized_name,
                                    "score": current_score,
                                    "start": current_start,
                                    "end": current_end,
                                    "method": "transformer"
                                })
                        
                        # Start a new name
                        current_name_parts = [word]
                        current_score = score
                        current_start = start
                        current_end = end
            
            # Add the last name if there's one being processed
            if current_name_parts:
                full_name = " ".join(current_name_parts)
                if self.is_valid_name(full_name):
                    normalized_name = self.normalize_name(full_name)
                    names.append({
                        "name": normalized_name,
                        "score": current_score,
                        "start": current_start,
                        "end": current_end,
                        "method": "transformer"
                    })
            
            return names
        except Exception as e:
            self.logger.warning(f"Transformer NER error: {e}")
            return []
    
    def combine_extracted_names(self, spacy_names: List[Dict[str, Any]], transformer_names: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine and deduplicate names detected by different methods, with improved confidence scoring."""
        combined = {}
        
        # Process spaCy names
        for name_obj in spacy_names:
            norm_name = name_obj["name"]
            if norm_name in combined:
                combined[norm_name]["methods"].add("spacy")
                combined[norm_name]["confidence"] = max(combined[norm_name]["confidence"], 0.85)
            else:
                combined[norm_name] = {
                    "name": norm_name, 
                    "methods": {"spacy"}, 
                    "confidence": 0.85,
                    "start": name_obj.get("start", 0),
                    "end": name_obj.get("end", 0)
                }
        
        # Process transformer names
        for name_obj in transformer_names:
            norm_name = name_obj["name"]
            score = name_obj.get("score", 0.8)
            
            if norm_name in combined:
                combined[norm_name]["methods"].add("transformer")
                # Boost confidence if detected by multiple methods
                combined[norm_name]["confidence"] = max(
                    combined[norm_name]["confidence"],
                    score,
                    # Boost confidence if detected by both methods
                    0.92 if "spacy" in combined[norm_name]["methods"] else 0
                )
            else:
                combined[norm_name] = {
                    "name": norm_name, 
                    "methods": {"transformer"}, 
                    "confidence": score,
                    "start": name_obj.get("start", 0),
                    "end": name_obj.get("end", 0)
                }
        
        # Final validation and conversion
        validated = []
        for candidate in combined.values():
            candidate["methods"] = list(candidate["methods"])
            # Stronger validation for names detected by only one method
            if len(candidate["methods"]) == 1 and candidate["confidence"] < 0.9:
                if self.is_valid_name(candidate["name"]) and self.is_really_person(candidate["name"]):
                    validated.append(candidate)
            else:
                # For names detected by multiple methods, just do basic validation
                if self.is_valid_name(candidate["name"]):
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
        image_hash = roi_obj.get("hash", "")
        
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
            
            # Log detected names
            self.console.print(f"[green]Names detected in frame {frame_idx}:[/green]")
            for name_obj in validated_names:
                method = name_obj.get("methods", ["unknown"])
                conf_val = name_obj.get("confidence", 1.0)
                self.console.print(f"  - {name_obj['name']} (Methods: {', '.join(method)}, Confidence: {conf_val:.2f})")
                
                # Track unique names and their occurrences
                self.unique_names.add(name_obj['name'])
                self.name_instances[name_obj['name']] += 1
                self.name_timestamps[name_obj['name']].append(timestamp)
            
            return self.DetectionResult(
                timestamp=timestamp,
                frame_idx=frame_idx,
                class_name=class_name,
                roi_path=roi_path,
                text=text,
                names=validated_names,
                confidence=confidence,
                hash_value=image_hash
            )
        return None
    
    def process_video(self, video_path: str, 
                      sampling_rate: float = 1.0, 
                      max_frames: Optional[int] = None,
                      skip_duplicates: bool = True) -> List[DetectionResult]:
        """Process the video: sample frames, detect ROIs, and extract person names with duplicate skipping."""
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
        
        self.logger.info(f"Will process up to {frames_to_process} frames at {sampling_rate}s intervals")
        self.console.print(f"Will process up to [bold]{frames_to_process}[/bold] frames at {sampling_rate}s intervals")
        if skip_duplicates:
            self.console.print("[yellow]Duplicate frame detection is enabled[/yellow]")
        
        results = []
        frame_idx = 0
        processed_count = 0
        skipped_count = 0
        
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
                        
                        # Skip duplicate frames if enabled
                        if skip_duplicates and self.should_skip_frame(frame, frame_idx):
                            skipped_count += 1
                            processed_count += 1  # Count toward processed total
                            progress.update(task, advance=1)
                        else:
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
                
                # Process results as they complete
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        self.logger.error(f"Error processing ROI: {e}")
        
        cap.release()
        
        # Save results to JSON
        results_file = self.RESULTS_DIR / f"name_detections_{self.timestamp_str}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        
        # Log statistics
        self.logger.info(f"Processed {processed_count} frames, skipped {skipped_count} duplicate frames")
        self.logger.info(f"Skipped {self.duplicate_frames_skipped} ROIs due to similarity")
        self.logger.info(f"Detected {len(results)} name instances across {len(self.unique_names)} unique names")
        self.logger.info(f"Saved results to {results_file}")
        
        self.console.print(f"[bold green]Processed {processed_count} frames, skipped {skipped_count} duplicate frames[/bold green]")
        self.console.print(f"[bold green]Skipped {self.duplicate_frames_skipped} ROIs due to similarity[/bold green]")
        self.console.print(f"[bold green]Detected {len(results)} name instances across {len(self.unique_names)} unique names[/bold green]")
        
        return results
    
    def cluster_names(self) -> Dict[str, Any]:
        """
        Cluster similar names using multiple similarity metrics:
        1. Fuzzy matching (token set ratio) - robust against titles and word reordering
        2. Jaccard similarity - compares word set overlap
        3. spaCy embedding similarity - captures semantic similarity
        """
        # Stricter thresholds for different similarity metrics
        threshold_fuzzy = 95       # Increased from 90
        threshold_jaccard = 0.7    # Increased from 0.5
        threshold_embedding = 0.9  # Increased from 0.85
        
        # New parameters
        max_cluster_size = 3       # Maximum number of names to allow in a cluster before additional validation
        max_cluster_variance = 0.2 # Maximum allowed variance in similarity within a cluster
        
        # Cache for embedding calculations to avoid recomputing
        embedding_cache = {}
        
        def get_embedding(name: str) -> np.ndarray:
            """Get and cache spaCy embeddings."""
            if name not in embedding_cache:
                try:
                    embedding_cache[name] = self.nlp(name).vector
                except Exception as e:
                    self.logger.warning(f"Failed to get embedding for '{name}': {e}")
                    embedding_cache[name] = np.zeros(self.nlp.vocab.vectors.shape[1])
            return embedding_cache[name]
        
        def jaccard_similarity(name1: str, name2: str) -> float:
            """Calculate Jaccard similarity between word sets."""
            set1 = set(name1.lower().split())
            set2 = set(name2.lower().split())
            
            # Special case: names with different word counts should have higher threshold
            if abs(len(set1) - len(set2)) > 1:
                return 0.0
                
            intersection = set1.intersection(set2)
            union = set1.union(set2)
            return len(intersection) / len(union) if union else 0
        
        def embedding_similarity(name1: str, name2: str) -> float:
            """Calculate cosine similarity between word embeddings."""
            vec1 = get_embedding(name1)
            vec2 = get_embedding(name2)
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 < 1e-6 or norm2 < 1e-6:
                return 0
                
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        
        def are_names_similar(name1: str, name2: str) -> bool:
            """Determine if names are similar using multiple metrics with stricter validation."""
            # Quick exact match check
            if name1 == name2:
                return True
            
            # Check if names are dramatically different lengths (character-wise)
            if abs(len(name1) - len(name2)) > 5:
                return False
                
            # Check if word counts are very different
            words1 = name1.split()
            words2 = name2.split()
            if abs(len(words1) - len(words2)) > 1:
                return False
                
            # Check for minimum shared word requirement
            common_words = set(word.lower() for word in words1) & set(word.lower() for word in words2)
            if not common_words:
                return False
            
            # Apply all similarity metrics
            fuzzy = fuzz.token_set_ratio(name1, name2)
            
            # Only compute more expensive metrics if fuzzy score is promising
            if fuzzy >= threshold_fuzzy:
                # Additional validation: Check for common name parts
                # Detect if either name is completely contained in the other
                if name1.lower() in name2.lower() or name2.lower() in name1.lower():
                    return True
                jaccard = jaccard_similarity(name1, name2)
                if jaccard >= threshold_jaccard:
                    # Final check with embedding similarity
                    embed = embedding_similarity(name1, name2)
                    return embed >= threshold_embedding
            
            return False
        
        def validate_cluster(aliases: Set[str]) -> bool:
            """
            Check if a cluster is valid based on internal similarity metrics.
            Returns False if the cluster should be split.
            """
            # Small clusters are always valid
            if len(aliases) <= 2:
                return True
                
            # For larger clusters, check internal coherence
            if len(aliases) > max_cluster_size:
                # Calculate pairwise similarities for all aliases
                similarities = []
                alias_list = list(aliases)
                for i in range(len(alias_list)):
                    for j in range(i+1, len(alias_list)):
                        jaccard = jaccard_similarity(alias_list[i], alias_list[j])
                        fuzzy = fuzz.token_set_ratio(alias_list[i], alias_list[j]) / 100.0
                        similarities.append(max(jaccard, fuzzy))
                
                # Calculate variance of similarities
                if similarities:
                    variance = np.var(similarities)
                    mean_sim = np.mean(similarities)
                    # If variance is high or mean similarity is low, reject cluster
                    if variance > max_cluster_variance or mean_sim < 0.75:
                        return False
            return True
        
        # Track best matches to improve clustering
        name_to_cluster_idx = {}
        canonical_clusters = []
        
        # First pass: create initial clusters
        for name in self.unique_names:
            best_match_idx = -1
            best_match_score = -1
            
            # Find best matching existing cluster
            for idx, cluster in enumerate(canonical_clusters):
                # Check similarity with each alias in the cluster
                for alias in cluster["aliases"]:
                    if are_names_similar(name, alias):
                        # For tie-breaking, prefer clusters with higher mention counts
                        cluster_size = sum(self.name_instances.get(a, 0) for a in cluster["aliases"])
                        current_score = cluster_size
                        
                        if current_score > best_match_score:
                            best_match_score = current_score
                            best_match_idx = idx
            
            if best_match_idx >= 0:
                # Add to best matching cluster
                canonical_clusters[best_match_idx]["aliases"].add(name)
                name_to_cluster_idx[name] = best_match_idx
            else:
                # Create new cluster
                new_idx = len(canonical_clusters)
                canonical_clusters.append({"canonical": name, "aliases": {name}})
                name_to_cluster_idx[name] = new_idx
        
        # Add cluster validation step - split invalid clusters
        i = 0
        while i < len(canonical_clusters):
            aliases = canonical_clusters[i]["aliases"]
            if not validate_cluster(aliases):
                self.logger.info(f"Splitting invalid cluster with names: {aliases}")
                
                # Remove the current cluster
                removed_cluster = canonical_clusters.pop(i)
                
                # Create new single-name clusters for each name
                for alias in removed_cluster["aliases"]:
                    new_idx = len(canonical_clusters)
                    canonical_clusters.append({"canonical": alias, "aliases": {alias}})
                    name_to_cluster_idx[alias] = new_idx
            else:
                i += 1
        
        # Second pass: merge similar clusters with additional validation
        i = 0
        while i < len(canonical_clusters):
            j = i + 1
            while j < len(canonical_clusters):
                # Check if any aliases between clusters are similar
                should_merge = False
                similarity_scores = []
                
                for alias_i in canonical_clusters[i]["aliases"]:
                    if should_merge:
                        break
                        
                    for alias_j in canonical_clusters[j]["aliases"]:
                        if are_names_similar(alias_i, alias_j):
                            # Calculate similarity score for validation
                            jaccard = jaccard_similarity(alias_i, alias_j)
                            fuzzy = fuzz.token_set_ratio(alias_i, alias_j) / 100.0
                            similarity_scores.append(max(jaccard, fuzzy))
                            should_merge = True
                            break
                
                if should_merge:
                    # Validate potential merged cluster
                    merged_aliases = canonical_clusters[i]["aliases"].union(canonical_clusters[j]["aliases"])
                    if validate_cluster(merged_aliases):
                        # Merge cluster j into cluster i
                        canonical_clusters[i]["aliases"].update(canonical_clusters[j]["aliases"])
                        # Update mapping for all aliases from cluster j
                        for alias in canonical_clusters[j]["aliases"]:
                            name_to_cluster_idx[alias] = i
                        # Remove cluster j
                        canonical_clusters.pop(j)
                    else:
                        # Don't merge if validation fails
                        self.logger.info(f"Prevented invalid cluster merge between clusters {i} and {j}")
                        j += 1
                else:
                    j += 1
            i += 1
        
        # Third pass: select best canonical name for each cluster with improved selection logic
        for idx, cluster in enumerate(canonical_clusters):
            aliases = cluster["aliases"]
            
            # Choose canonical name based on frequency, completeness, and length
            best_canonical = cluster["canonical"]
            max_score = -1
            
            for alias in aliases:
                # Start with mentions count as base score
                mentions = self.name_instances.get(alias, 0)
                
                # Calculate name completeness factors
                word_count = len(alias.split())
                char_count = len(alias)
                
                # Base length score
                length_score = 1.0
                
                # Penalize very short or very long names
                if word_count == 1:  # Single word names are likely incomplete
                    length_score = 0.5
                elif word_count > 4:  # Very long names may include extra information
                    length_score = 0.8
                    
                # For names with equal word count (like "Shari" vs "Shariff"), 
                # add a completeness bonus based on character length
                completeness_bonus = 0
                if word_count >= 2:
                    # Add a small bonus proportional to character length
                    completeness_bonus = min(0.5, char_count / 50)
                    
                # Bonus for names that appear to be more complete
                # Check if a name appears to be a substring of another name
                for other_alias in aliases:
                    if alias != other_alias:
                        # If this name contains all words from another name plus extra characters
                        if set(other_alias.lower().split()).issubset(set(alias.lower().split())) and len(alias) > len(other_alias):
                            completeness_bonus += 0.3
                        # If this appears to be a shortened version
                        elif other_alias.lower().startswith(alias.lower()) and len(other_alias) > len(alias):
                            completeness_bonus -= 0.2
                        
                # Calculate final score
                score = (mentions * length_score) + completeness_bonus
                
                # Debug logging to understand selection process
                self.logger.debug(f"Name: {alias}, Score: {score} (Mentions: {mentions}, Length score: {length_score}, Completeness: {completeness_bonus})")
                
                if score > max_score:
                    max_score = score
                    best_canonical = alias
                # Tiebreaker: if scores are equal, prefer the longer name (likely more complete)
                elif score == max_score and len(alias) > len(best_canonical):
                    best_canonical = alias
                    
            canonical_clusters[idx]["canonical"] = best_canonical
        
        # Aggregate stats and prepare output
        grouped_summary = {}
        
        for cluster in canonical_clusters:
            canonical = cluster["canonical"]
            aliases = cluster["aliases"]
            
            # Compute aggregate statistics
            total_mentions = sum(self.name_instances.get(a, 0) for a in aliases)
            timestamps = []
            
            for alias in aliases:
                timestamps.extend(self.name_timestamps.get(alias, []))
                
            if timestamps:
                timestamps.sort()  # Ensure chronological order
                start = timestamps[0]
                end = timestamps[-1]
                duration = end - start
                
                # Calculate frequency statistics
                avg_interval = duration / max(total_mentions - 1, 1) if total_mentions > 1 else 0
                
                # Calculate activity periods
                active_periods = []
                current_period_start = start
                last_timestamp = start
                
                for ts in timestamps[1:]:
                    # If gap is more than 3 days, consider it a new period
                    if ts - last_timestamp > 3 * 86400:  # 3 days in seconds
                        active_periods.append((current_period_start, last_timestamp))
                        current_period_start = ts
                    last_timestamp = ts
                    
                # Add the last period
                active_periods.append((current_period_start, last_timestamp))
                
                # Format periods for output
                formatted_periods = [
                    {
                        "start": self.format_timestamp(period[0]),
                        "end": self.format_timestamp(period[1]),
                        "duration": period[1] - period[0]
                    }
                    for period in active_periods
                ]
            else:
                start = end = duration = avg_interval = 0
                formatted_periods = []
            
            # Create summary entry
            grouped_summary[canonical] = {
                "aliases": sorted(aliases),
                "mentions": total_mentions,
                "first_seen": self.format_timestamp(start),
                "last_seen": self.format_timestamp(end),
                "duration": duration,
                "avg_interval": avg_interval,
                "active_periods": formatted_periods,
                "metrics": {
                    "distinct_aliases": len(aliases)
                }
            }
        
        # Sort by mentions (most frequent first)
        sorted_summary = dict(sorted(
            grouped_summary.items(), 
            key=lambda item: item[1]["mentions"], 
            reverse=True
        ))
        
        # Save to disk
        cluster_path = self.RESULTS_DIR / f"name_clusters_{self.timestamp_str}.json"
        try:
            with open(cluster_path, 'w', encoding='utf-8') as f:
                json.dump(sorted_summary, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved name clustering summary to {cluster_path} with {len(sorted_summary)} clusters")
        except Exception as e:
            self.logger.error(f"Failed to save name clustering summary: {e}")
        
        # Display summary information
        self.console.print(f"\n[bold]Name Clustering Results:[/bold]")
        self.console.print(f"Found {len(sorted_summary)} name clusters from {len(self.unique_names)} unique names")
        
        # Print top clusters
        self.console.print("\n[bold]Top name clusters:[/bold]")
        for i, (canonical, details) in enumerate(list(sorted_summary.items())[:10]):
            alias_count = len(details["aliases"])
            aliases_str = ", ".join(details["aliases"][:3])
            if alias_count > 3:
                aliases_str += f" and {alias_count-3} more"
            self.console.print(f"  {canonical}: {details['mentions']} mentions, {alias_count} aliases [{aliases_str}]")
        
        return sorted_summary
    
    def generate_timeline(self, results: List[DetectionResult]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a timeline of when each name appears in the video."""
        if not self.name_timestamps:
            self.logger.warning("No name timestamps to generate timeline")
            return {}
        
        timeline = {}
        for name, timestamps in self.name_timestamps.items():
            timeline[name] = [{"timestamp": ts, "time_str": self.format_timestamp(ts)} for ts in sorted(timestamps)]
        
        # Save timeline to JSON
        timeline_file = self.RESULTS_DIR / f"name_timeline_{self.timestamp_str}.json"
        with open(timeline_file, 'w', encoding='utf-8') as f:
            json.dump(timeline, f, indent=2)
        
        self.logger.info(f"Generated timeline and saved to {timeline_file}")
        return timeline
    
    def format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    def generate_summary(self, results: List[DetectionResult], processing_time: float = None) -> Dict[str, Any]:
        """Generate a comprehensive summary of detected names from all processed ROIs."""
        if not results:
            self.logger.warning("No results to summarize")
            return {"unique_names": 0, "total_instances": 0, "names": {}}
        
        # First, cluster similar names to improve accuracy
        self.logger.info("Clustering similar names...")
        clustered_names = self.cluster_names()
        
        # Generate name occurrences based on clustered data
        sorted_names = sorted(clustered_names.items(), key=lambda x: x[1]["mentions"], reverse=True)
        
        # Generate class statistics
        class_stats = defaultdict(int)
        for result in results:
            class_stats[result.class_name] += 1
        
        # Print summary
        self.console.print("\n[bold]Name Detection Summary:[/bold]")
        self.console.print(f"Total detected name instances: {len(results)}")
        self.console.print(f"Unique names after clustering: {len(clustered_names)}")
        
        self.console.print("\n[bold]Top detected names (after clustering):[/bold]")
        for name, details in sorted_names[:10]:
            first_appearance = details.get("first_seen", "unknown")
            last_appearance = details.get("last_seen", "unknown")
            aliases = details.get("aliases", [])
            alias_count = len(aliases)
            
            self.console.print(f"  {name}: {details['mentions']} instances, {alias_count} aliases")
            self.console.print(f"     First seen: {first_appearance}, Last seen: {last_appearance}")
        
        self.console.print("\n[bold]Detection by graphic type (with Names):[/bold]")
        for class_name, count in sorted(class_stats.items(), key=lambda x: x[1], reverse=True):
            self.console.print(f"  {class_name}: {count} detections")
        
        # Create comprehensive summary
        summary = {
            "unique_names_original": len(self.unique_names),
            "unique_names_clustered": len(clustered_names),
            "total_instances": len(results),
            "clustered_names": clustered_names,
            "class_statistics": dict(class_stats),
            "efficiency": {
                "duplicate_frames_skipped": self.duplicate_frames_skipped,
                "duplicate_rois_skipped": self.duplicate_frames_skipped,
            },
            "processing_time_seconds": processing_time  # Add processing time
        }
        
        # Save summary to JSON
        summary_file = self.RESULTS_DIR / f"detection_summary_{self.timestamp_str}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Generated summary and saved to {summary_file}")
        return summary
    
    def get_model_versions(self) -> Dict[str, str]:
        """Return the versions/paths of models used for reproducibility logging."""
        versions = {
            "yolo_model": self.config.get("yolo_model_path", "unknown"),
            "spacy_model": self.config.get("spacy_model", "unknown"),
            "transformer_model": self.config.get("transformer_model", "unknown"),
            "gliner_version": "not_installed"
        }
        
        if GLINER_AVAILABLE:
            try:
                import gliner_spacy
                versions["gliner_version"] = getattr(gliner_spacy, '__version__', "unknown")
            except Exception:
                versions["gliner_version"] = "unknown"
        
        return versions

    def run(self, video_path: str, sampling_rate: float = 1.0, max_frames: Optional[int] = None, skip_duplicates: bool = True) -> Dict[str, Any]:
        """Run the complete pipeline on a video file with improved efficiency."""
        self.console.print(f"[bold green]Processing video:[/bold green] {video_path}")
        self.console.print(f"[bold]Settings:[/bold] Sampling rate: {sampling_rate}s, Skip duplicates: {skip_duplicates}")
        
        # Process video
        start_time = datetime.now()
        results = self.process_video(video_path, sampling_rate, max_frames, skip_duplicates)
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Generate outputs
        summary = self.generate_summary(results, processing_time)
        timeline = self.generate_timeline(results)
        
        # Save model version info
        model_versions = self.get_model_versions()
        versions_file = self.RESULTS_DIR / f"model_versions_{self.timestamp_str}.json"
        with open(versions_file, 'w', encoding='utf-8') as f:
            json.dump(model_versions, f, indent=2)
        self.logger.info(f"Saved model versions to {versions_file}")


        self.console.print(f"\n[bold]Processing completed in {processing_time:.2f} seconds[/bold]")
        self.console.print("\n[bold]Output directories:[/bold]")
        self.console.print(f"Regions of interest: {self.ROI_DIR}")
        self.console.print(f"OCR text files: {self.OCR_TEXT_DIR}")
        self.console.print(f"Detected names: {self.NAMES_DIR}")
        self.console.print(f"Results JSON: {self.RESULTS_DIR}")
        self.console.print(f"Log file: {self.log_filename}")
        self.console.print(f"\n[bold]Model versions used:[/bold]")
        for k, v in model_versions.items():
            self.console.print(f"  {k}: {v}")
        return {
            "results": [r.to_dict() for r in results],
            "summary": summary,
            "timeline": timeline,
            "processing_time": processing_time,
            "dirs": {
                "roi": str(self.ROI_DIR),
                "ocr": str(self.OCR_TEXT_DIR),
                "names": str(self.NAMES_DIR),
                "results": str(self.RESULTS_DIR),
                "log": str(self.log_filename)
            }
        }

def parse_arguments():
    """Parse command line arguments with enhanced options."""
    parser = argparse.ArgumentParser(description='Enhanced News Graphics Name Detection Pipeline')
    parser.add_argument('--video', '-v', type=str, default=str(Path.cwd() / "4. ANEP/Videos/Video_T4.mp4"),
                        help='Path to the video file to process')
    parser.add_argument('--sampling_rate', '-s', type=float, default=1.0,
                        help='Sampling rate in seconds (default: 1.0)')
    parser.add_argument('--max_frames', '-m', type=int, default=None,
                        help='Maximum number of frames to process (default: all)')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to JSON configuration file')
    parser.add_argument('--skip_duplicates', '-d', action='store_true', default=True,
                        help='Enable duplicate frame detection and skipping (default: True)')
    parser.add_argument('--no_skip_duplicates', '-n', action='store_false', dest='skip_duplicates',
                        help='Disable duplicate frame detection and skipping')
    parser.add_argument('--similarity_threshold', '-t', type=float, default=SIMILARITY_THRESHOLD,
                        help=f'Threshold for frame similarity (0.0-1.0, default: {SIMILARITY_THRESHOLD})')
    parser.add_argument('--contiguous_skip', '-k', type=int, default=CONTIGUOUS_SKIP_THRESHOLD,
                        help=f'Skip processing after this many similar frames in a row (default: {CONTIGUOUS_SKIP_THRESHOLD})')
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
    
    # Update config with command line parameters
    if config is None:
        config = {}
    config["similarity_threshold"] = args.similarity_threshold
    config["contiguous_skip_threshold"] = args.contiguous_skip
    
    video_path = args.video
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    try:
        detector = NewsGraphicsNameDetector(config)
        detector.run(
            video_path, 
            args.sampling_rate, 
            args.max_frames, 
            args.skip_duplicates
        )
    except KeyboardInterrupt:
        print("\nWarning: Process interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()