import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import subprocess

# ---------- Helper Functions (kept for compatibility) ----------
def contains_graphics(frame, roi_coords, edge_threshold=50):
    # (Unused in this simplified version)
    return True

def is_static(frame, next_frame, static_threshold=30):
    # (Unused in this simplified version)
    return True

def compute_frame_difference(frame1, frame2):
    # (Unused in this simplified version)
    return 0

# ---------- New Simple Extraction Function ----------
def extract_random_frame(video_path):
    """
    Opens the video and selects a random frame from the first 2 seconds.
    Returns a dictionary with keys: 'frame_index', 'frame', and 'replace_count'.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame = int(fps * 2)
    if total_frames < max_frame:
        max_frame = total_frames
    idx = np.random.randint(0, max_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return {
        "frame_index": idx,
        "frame": frame,
        "replace_count": 0
    }

# ---------- Simple Tooltip Class ----------
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 20
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()

# ---------- Simplified GUI Application ----------
class VideoFrameExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Frame Extractor")
        self.root.configure(bg="white")
        self.root.geometry("1100x800")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Rounded.TButton",
                        font=("Segoe UI", 10, "bold"),
                        foreground="#ffffff",
                        background="#007ACC",
                        relief="flat",
                        padding=6)
        style.map("Rounded.TButton",
                  background=[('active', '#005F99'), ('disabled', '#d9d9d9')])
        
        # Set application icon
        try:
            current_dir = os.getcwd()
            image_path = os.path.join(current_dir, "1. Dataset Downloader", "4. Frame Extraction", "FE.icns")
            icon = ImageTk.PhotoImage(file=image_path)
            self.root.iconphoto(True, icon)
        except Exception as e:
            print("Icon not found or failed to load:", e)

        # Menu bar
        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load Video Directory", command=self.load_directory, accelerator="Command+L")
        filemenu.add_separator()
        filemenu.add_command(label="Open Output Directory", command=self.open_output_dir)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=root.quit, accelerator="Control+Q")
        menubar.add_cascade(label="File", menu=filemenu)
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=helpmenu)
        root.config(menu=menubar)
        
        # Bind keys (macOS style)
        root.bind("<Command-l>", lambda event: self.load_directory())
        root.bind("<Command-n>", lambda event: self.process_next_video())
        root.bind("<Command-s>", lambda event: self.save_frames())
        root.bind("<Control-s>", lambda event: self.open_settings())
        root.bind("<Control-q>", lambda event: root.quit())
        
        # Variables
        self.video_files = []
        self.current_video_index = 0
        self.current_video_path = None
        self.candidate_frame = None  # Only one random frame is used
        self.output_dir = "/Volumes/Filis SSD/FYP/Dataset_Frames"
        os.makedirs(self.output_dir, exist_ok=True)

        # Header frame
        header_frame = tk.Frame(root, bg="white")
        header_frame.pack(pady=10, fill=tk.X)
        title_label = tk.Label(header_frame, text="Video Frame Extractor", font=("Helvetica", 22, "bold"), bg="white")
        title_label.pack()
        
        # Control frame (centered)
        control_frame = tk.Frame(root, bg="white")
        control_frame.pack(pady=5, fill=tk.X)
        self.load_dir_btn = ttk.Button(control_frame, text="Load Video Directory", command=self.load_directory, style="Rounded.TButton")
        self.load_dir_btn.grid(row=0, column=0, padx=10)
        self.info_label = tk.Label(control_frame, text="Select a directory containing videos.", font=("Arial", 14), bg="white")
        self.info_label.grid(row=0, column=1, padx=20)
        self.settings_btn = ttk.Button(control_frame, text="Settings", command=self.open_settings, style="Rounded.TButton")
        self.settings_btn.grid(row=0, column=2, padx=10)
        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_columnconfigure(1, weight=1)
        control_frame.grid_columnconfigure(2, weight=1)
        
        self.video_name_label = tk.Label(root, text="", font=("Arial", 18), bg="white")
        self.video_name_label.pack(pady=5)
        
        # Candidate frame display (centered in a frame with 3 columns)
        self.preview_frame = tk.Frame(root, bg="white")
        self.preview_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Bottom frame (centered buttons)
        bottom_frame = tk.Frame(root, bg="white")
        bottom_frame.pack(pady=10, fill=tk.X)
        button_frame = tk.Frame(bottom_frame, bg="white")
        button_frame.pack(expand=True)
        self.next_btn = ttk.Button(button_frame, text="Next Video", command=self.process_next_video, state=tk.DISABLED, style="Rounded.TButton")
        self.next_btn.pack(side=tk.LEFT, padx=10)
        self.save_btn = ttk.Button(button_frame, text="Save Frames", command=self.save_frames, style="Rounded.TButton")
        self.save_btn.pack(side=tk.LEFT, padx=10)
        
        self.status_label = tk.Label(root, text="Status: Ready", font=("Arial", 12), bg="white", anchor="w")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        self.root.update_idletasks()

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")
        self.status_label.update_idletasks()

    def show_about(self):
        messagebox.showinfo("About", "Video Frame Extractor\nVersion 1.0\nDeveloped by Your Name")

    def open_output_dir(self):
        try:
            if os.name == 'nt':
                os.startfile(self.output_dir)
            elif os.uname().sysname == 'Darwin':
                subprocess.Popen(["open", self.output_dir])
            else:
                subprocess.Popen(["xdg-open", self.output_dir])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open directory: {e}")

    def open_settings(self):
        settings_win = tk.Toplevel(self.root)
        settings_win.title("Settings")
        settings_win.geometry("340x330")
        settings_win.configure(bg="white")
        # Bind Control+W to close settings window
        settings_win.bind("<Control-w>", lambda event: settings_win.destroy())
        # For simplicity, we won't include extra settings here.
        tk.Label(settings_win, text="No extra settings in this simplified version.", bg="white", font=("Arial", 12)).pack(pady=20)
        ttk.Button(settings_win, text="Close", command=settings_win.destroy, style="Rounded.TButton").pack(pady=10)

    def load_directory(self):
        directory = filedialog.askdirectory()
        if not directory:
            return
        self.video_files = [os.path.join(directory, f) for f in os.listdir(directory)
                            if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        if not self.video_files:
            messagebox.showerror("Error", "No video files found in the selected directory.")
            return
        self.current_video_index = 0
        self.info_label.config(text=f"Found {len(self.video_files)} video(s).")
        self.next_btn.config(state=tk.NORMAL)
        self.update_status("Directory loaded. Processing first video...")
        self.process_video(self.video_files[self.current_video_index])

    def process_video(self, video_path):
        self.current_video_path = video_path
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.video_name_label.config(text=f"Processing video: {video_name}")
        self.update_status(f"Extracting a random frame from the first 2 seconds of {video_name}...")
        # Run frame extraction in a background thread
        threading.Thread(target=self.extract_frame_thread, args=(video_path,), daemon=True).start()

    def extract_frame_thread(self, video_path):
        candidate = extract_random_frame(video_path)
        self.root.after(0, self.on_frame_extracted, candidate)

    def on_frame_extracted(self, candidate):
        if candidate is None:
            self.update_status("Failed to extract a frame.")
            return
        self.candidate_frame = candidate
        self.display_candidate()
        self.update_status("Frame loaded.")

    def display_candidate(self):
        # Clear previous display
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        if not self.candidate_frame:
            return
        # Display candidate frame in one cell (centered in 3 columns)
        thumb_frame = tk.Frame(self.preview_frame, bd=2, relief=tk.GROOVE, bg="white")
        frame_rgb = cv2.cvtColor(self.candidate_frame["frame"], cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(frame_rgb)
        im_thumb = im_pil.resize((300, 200))
        photo = ImageTk.PhotoImage(im_thumb)
        self.candidate_frame["photo"] = photo
        img_label = tk.Label(thumb_frame, image=photo, bg="white")
        img_label.pack(padx=5, pady=5)
        img_label.bind("<Double-Button-1>", lambda e, img=im_pil: self.preview_full_image(img))
        idx_label = tk.Label(thumb_frame, text=f"Frame {self.candidate_frame.get('frame_index')}", bg="white", font=("Arial", 10))
        idx_label.pack(pady=2)
        replace_btn = ttk.Button(thumb_frame, text="Replace", command=self.replace_candidate, style="Rounded.TButton")
        replace_btn.pack(pady=5, padx=5)
        # Center the candidate in a grid of 3 columns (empty cells on either side)
        thumb_frame.grid(row=0, column=1, padx=15, pady=10, sticky="nsew")

    def preview_full_image(self, im_pil):
        preview_win = tk.Toplevel(self.root)
        preview_win.title("Full-Size Preview")
        preview_win.configure(bg="white")
        screen_width = self.root.winfo_screenwidth() - 100
        screen_height = self.root.winfo_screenheight() - 100
        img_width, img_height = im_pil.size
        scale = min(screen_width/img_width, screen_height/img_height, 1)
        new_size = (int(img_width*scale), int(img_height*scale))
        im_resized = im_pil.resize(new_size)
        photo = ImageTk.PhotoImage(im_resized)
        label = tk.Label(preview_win, image=photo, bg="white")
        label.image = photo
        label.pack(padx=10, pady=10)

    def replace_candidate(self):
        # Simply pick a new random frame from the first 2 seconds
        candidate = extract_random_frame(self.current_video_path)
        if candidate is None:
            messagebox.showinfo("Info", "Unable to extract a new frame.")
            return
        self.candidate_frame = candidate
        self.display_candidate()
        self.update_status("Frame replaced.")

    def save_frames(self):
        if not self.candidate_frame:
            return
        video_name = os.path.splitext(os.path.basename(self.current_video_path))[0]
        idx = self.candidate_frame.get("frame_index")
        frame = self.candidate_frame.get("frame")
        if frame is not None:
            output_path = os.path.join(self.output_dir, f"{video_name}_frame{idx}.png")
            cv2.imwrite(output_path, frame)
            self.update_status(f"Frame saved for {video_name}.")
        else:
            self.update_status("No frame to save.")

    def process_next_video(self):
        self.save_frames()
        self.current_video_index += 1
        if self.current_video_index >= len(self.video_files):
            messagebox.showinfo("Info", "All videos processed.")
            self.next_btn.config(state=tk.DISABLED)
            self.video_name_label.config(text="All videos processed.")
            for widget in self.preview_frame.winfo_children():
                widget.destroy()
            self.update_status("All videos processed.")
            return
        self.process_video(self.video_files[self.current_video_index])

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoFrameExtractorGUI(root)
    root.mainloop()
