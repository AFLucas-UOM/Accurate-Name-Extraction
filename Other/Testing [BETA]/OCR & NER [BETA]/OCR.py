#!/usr/bin/env python3
import cv2
import pytesseract
from pytesseract import Output
import spacy
import sys
import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from tabulate import tabulate
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from transformers import pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Initialize spaCy model for Named Entity Recognition (NER)
try:
    nlp = spacy.load("en_core_web_md")
except Exception as e:
    logging.error("Failed to load spaCy model: %s", e)
    sys.exit(1)

# Set path to Tesseract OCR executable (update if necessary)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Create screenshots directory if it doesn't exist
SCREENSHOT_DIR = "screenshots"
if not os.path.exists(SCREENSHOT_DIR):
    os.makedirs(SCREENSHOT_DIR)

# Compile regex to filter out website URLs (if any)
url_pattern = re.compile(r'\b(?:https?://)?[\w-]+\.[a-z]{2,6}\b', re.IGNORECASE)

# --------------------- Device Configuration for Transformers ---------------------
# Check for GPU or Apple Silicon MPS; if none available, use CPU (-1)
if torch.cuda.is_available():
    device = 0
    logging.info("Using CUDA (GPU) for Transformers pipeline.")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
    logging.info("Using Apple Silicon MPS for Transformers pipeline.")
else:
    device = -1
    logging.info("No GPU or MPS available; using CPU for Transformers pipeline.")

# --------------------- Utility Functions ---------------------

def create_progress_bar(percentage, width=50):
    """Return a string representing the progress bar for a given percentage."""
    filled = int(width * percentage / 100)
    return "█" * filled + "░" * (width - filled)

def display_progress_bar(current, total, width=50):
    """Display the progress bar on the same line in the console."""
    percentage = (current / total) * 100 if total else 0
    progress_bar = create_progress_bar(percentage, width)
    sys.stdout.write(f"\rProcessing frames: {percentage:5.1f}% {progress_bar} {current}/{total} frames")
    sys.stdout.flush()

def preprocess_frame(frame):
    """
    Preprocess the frame to enhance OCR accuracy.
    Converts to grayscale, applies median blurring for noise reduction,
    and binary thresholding.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    return thresh

def save_cropped_image(frame, x, y, w, h, timestamp, name):
    """
    Save a cropped image of the detected name.
    The filename includes a sanitized version of the name and the timestamp.
    """
    safe_name = re.sub(r'[^\w\-_. ]', '', name)
    filename = os.path.join(SCREENSHOT_DIR, f"{safe_name}_{timestamp:07.2f}.png")
    cropped_image = frame[y:y+h, x:x+w]
    cv2.imwrite(filename, cropped_image)

def is_valid_name(name):
    """
    Validate that the detected name is plausible.
    Allows only alphabetic characters, spaces, hyphens, apostrophes, and periods.
    Ensures the name starts and ends with a letter.
    """
    name = name.strip()
    return re.match(r"^[A-Za-z][A-Za-z\s\.\'\-]*[A-Za-z]$", name) is not None

# --------------------- OCR and NER Processing Functions ---------------------

def group_ocr_by_line(ocr_data, conf_threshold=70):
    """
    Group OCR results by (block_num, par_num, line_num).
    Filters out words with low confidence.
    Returns a dictionary where each key represents a line and the value is a list of words.
    """
    lines = {}
    n = len(ocr_data['text'])
    for i in range(n):
        word_text = ocr_data['text'][i].strip()
        try:
            conf = float(ocr_data['conf'][i])
        except ValueError:
            conf = 0
        if word_text == "" or conf < conf_threshold:
            continue
        key = (ocr_data['block_num'][i], ocr_data['par_num'][i], ocr_data['line_num'][i])
        if key not in lines:
            lines[key] = []
        lines[key].append({
            "text": word_text,
            "left": ocr_data['left'][i],
            "top": ocr_data['top'][i],
            "width": ocr_data['width'][i],
            "height": ocr_data['height'][i],
            "conf": conf
        })
    return lines

def process_line(words_in_line, nlp, url_pattern):
    """
    Process a single line of OCR words.
    Reconstructs the line text and computes character offsets.
    Runs NER on the line and returns a list of detections with bounding boxes.
    """
    line_text = " ".join(word["text"] for word in words_in_line)
    words_with_offsets = []
    current_offset = 0
    for word in words_in_line:
        word_text = word["text"]
        start = current_offset
        end = start + len(word_text)
        words_with_offsets.append({
            "word": word_text,
            "left": word["left"],
            "top": word["top"],
            "width": word["width"],
            "height": word["height"],
            "start": start,
            "end": end
        })
        current_offset = end + 1  # account for the space
    detections = []
    doc = nlp(line_text)
    for ent in doc.ents:
        # Only consider PERSON entities that pass our URL and valid name checks
        if ent.label_ == "PERSON" and not url_pattern.search(ent.text) and is_valid_name(ent.text):
            ent_start = ent.start_char
            ent_end = ent.end_char
            matching_words = [w for w in words_with_offsets if not (w['end'] <= ent_start or w['start'] >= ent_end)]
            if matching_words:
                x_min = min(w['left'] for w in matching_words)
                y_min = min(w['top'] for w in matching_words)
                x_max = max(w['left'] + w['width'] for w in matching_words)
                y_max = max(w['top'] + w['height'] for w in matching_words)
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                detections.append({"name": ent.text, "bbox": bbox})
    return detections

def process_frame(frame, timestamp, nlp, url_pattern):
    """
    Process a single video frame:
      - Preprocess the frame.
      - Perform OCR (with a custom configuration).
      - Group OCR results by line and run NER on each line.
      - For each PERSON entity detected, compute the bounding box and save a cropped screenshot.
    Returns a list of detections for this frame.
    """
    processed_frame = preprocess_frame(frame)
    custom_config = r'--oem 3 --psm 6'
    ocr_data = pytesseract.image_to_data(processed_frame, config=custom_config, output_type=Output.DICT)
    lines = group_ocr_by_line(ocr_data)
    frame_detections = []
    for words in lines.values():
        detections = process_line(words, nlp, url_pattern)
        if detections:
            for det in detections:
                x, y, w, h = det["bbox"]
                frame_detections.append({
                    "timestamp": timestamp,
                    "name": det["name"],
                    "bbox": (x, y, w, h)
                })
                save_cropped_image(frame, x, y, w, h, timestamp, det["name"])
    return frame_detections

def process_video(video_path, frame_interval=20, nlp=None, url_pattern=None):
    """
    Process the video file, extracting PERSON entities from every Nth frame.
    Returns a list of detections with timestamps, names, and positions.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Error: Could not open video '%s'.", video_path)
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []
    futures = []
    frame_count = 0
    logging.info("Starting video processing...")
    with ThreadPoolExecutor() as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                futures.append(executor.submit(process_frame, frame.copy(), timestamp, nlp, url_pattern))
            frame_count += 1
            display_progress_bar(frame_count, total_frames)
    cap.release()
    for future in as_completed(futures):
        try:
            results.extend(future.result())
        except Exception as e:
            logging.error("Error processing a frame: %s", e)
    print()  # New line after progress bar
    return results

# --------------------- Results Output Functions ---------------------

def save_results_to_csv(results, filename="DetectedNames.csv"):
    """
    Save the detected names, timestamps, and positions to a CSV file.
    """
    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp (s)", "Detected Name", "Position (x, y, width, height)"])
            for r in results:
                x, y, w, h = r["bbox"]
                writer.writerow([f"{r['timestamp']:07.2f}", r["name"], f"(x={x}, y={y}, width={w}, height={h})"])
        logging.info("Results saved to %s", filename)
    except Exception as e:
        logging.error("Failed to save CSV: %s", e)

def display_results(results):
    """
    Display the results in a table format and save them to CSV.
    """
    if results:
        results.sort(key=lambda r: r["timestamp"])
        table_data = [
            [f"{r['timestamp']:07.2f}", r["name"], f"(x={r['bbox'][0]}, y={r['bbox'][1]}, width={r['bbox'][2]}, height={r['bbox'][3]})"]
            for r in results
        ]
        headers = ["Timestamp (s)", "Detected Name", "Position (x, y, width, height)"]
        print("\n" + "=" * 70)
        print("Detected Names with Timestamps and Positions from Video")
        print("=" * 70)
        print(tabulate(table_data, headers=headers, tablefmt="pretty"))
        save_results_to_csv(results)
    else:
        logging.info("No names were detected in the video.")

def generate_statistics(results):
    """
    Generate and display statistics:
      - Improved bar chart for unique name counts.
      - Enhanced heat map for bounding box centers.
    """
    name_counts = {}
    centers = []  # To store bounding box centers (x, y)
    
    for r in results:
        name = r["name"].strip().lower()
        name_counts[name] = name_counts.get(name, 0) + 1
        x, y, w, h = r["bbox"]
        center_x = x + w / 2
        center_y = y + h / 2
        centers.append((center_x, center_y))
    
    # Use seaborn theme for better visuals
    sns.set_theme(style="whitegrid")
    
    # Improved Bar Chart of Unique Names
    df_counts = pd.DataFrame({
        "Name": list(name_counts.keys()),
        "Count": list(name_counts.values())
    })
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_counts, x="Name", y="Count", palette="viridis")
    ax.set_title("Frequency of Detected Names", fontsize=16)
    ax.set_xlabel("Name", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Enhanced Heat Map of Detection Locations
    if centers:
        centers = np.array(centers)
        x_vals = centers[:, 0]
        y_vals = centers[:, 1]
        heatmap, _, _ = np.histogram2d(x_vals, y_vals, bins=50)
        heatmap = np.rot90(heatmap)
        heatmap = np.flipud(heatmap)
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap, cmap="mako", cbar_kws={'label': 'Frequency'})
        plt.title("Heat Map of Name Detection Locations", fontsize=16)
        plt.xlabel("X Coordinate Bins", fontsize=14)
        plt.ylabel("Y Coordinate Bins", fontsize=14)
        plt.tight_layout()
        plt.show()
    else:
        print("No detection centers to plot.")

# --------------------- Transformers-based Filtering ---------------------
def filter_detected_names_csv(input_csv="DetectedNames.csv", output_csv="FilteredDetectedNames.csv"):
    """
    Load the CSV of detected names, then use a Hugging Face Transformers NER pipeline
    to further validate that each name is an actual person.
    The pipeline is configured to use the available device (GPU, MPS, or CPU).
    Saves the filtered data to a new CSV file.
    """
    df = pd.read_csv(input_csv)
    ner = pipeline("ner", 
                   model="dbmdz/bert-large-cased-finetuned-conll03-english", 
                   device=device)
    
    def is_valid_name_transformers(name):
        entities = ner(name)
        return any(entity['entity'] in ['B-PER', 'I-PER'] for entity in entities)
    
    filtered_df = df[df['Detected Name'].apply(is_valid_name_transformers)]
    filtered_df.to_csv(output_csv, index=False)
    logging.info("Filtered CSV saved to %s", output_csv)

# --------------------- Main Entry Point ---------------------
def main():
    video_path = 'video2.mp4'  # Update with your video file path
    results = process_video(video_path, frame_interval=20, nlp=nlp, url_pattern=url_pattern)
    display_results(results)
    generate_statistics(results)
    
    # After saving the CSV of detected names, filter it using Transformers-based NER.
    filter_detected_names_csv("DetectedNames.csv", "FilteredDetectedNames.csv")

if __name__ == "__main__":
    main()