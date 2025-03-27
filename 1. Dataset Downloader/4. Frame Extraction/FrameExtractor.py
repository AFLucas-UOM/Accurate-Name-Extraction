import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import subprocess

# ---------- Helper Functions (unchanged) ----------
def contains_graphics(frame, roi_coords, edge_threshold=50):
    x, y, w, h = roi_coords
    frame_h, frame_w = frame.shape[:2]
    if y + h > frame_h or x + w > frame_w:
        new_y = int(frame_h * 0.8)
        roi = frame[new_y:frame_h, 0:frame_w]
    else:
        roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        roi = frame
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_roi, 50, 150)
    edge_density = cv2.countNonZero(edges) / (roi.shape[1] * roi.shape[0])
    return edge_density > (edge_threshold / 1000.0)

def is_static(frame, next_frame, static_threshold=30):
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    return np.mean(diff) < static_threshold  

def compute_frame_difference(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    return np.mean(diff)

def get_random_valid_frame(video_path, segment_start, segment_end, roi_coords,
                           static_threshold=30, attempts=10, existing_frames=[]):
    cap = cv2.VideoCapture(video_path)
    candidate = None
    for _ in range(attempts):
        if segment_end <= segment_start:
            idx = segment_start
        else:
            idx = np.random.randint(segment_start, segment_end)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx+1)
        ret, next_frame = cap.read()
        if ret and next_frame is not None:
            if not is_static(frame, next_frame, static_threshold):
                continue
        if not contains_graphics(frame, roi_coords):
            continue
        if any(compute_frame_difference(frame, ex) < 30 for ex in existing_frames):
            continue
        candidate = (idx, frame)
        break
    cap.release()
    return candidate

def fallback_random_frame(video_path, excluded_indices, min_separation=300,
                          roi_coords=None, static_threshold=30, attempts=20):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    candidate = None
    for _ in range(attempts):
        idx = int(np.random.randint(0, total_frames))
        if any(abs(idx - ex) < min_separation for ex in excluded_indices):
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        if roi_coords is not None:
            if not contains_graphics(frame, roi_coords):
                continue
        candidate = (idx, frame)
        break
    cap.release()
    return candidate

def extract_candidate_frames(video_path, num_frames, roi_coords, static_threshold=30):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    segment_size = total_frames // num_frames
    candidate_list = []
    for i in range(num_frames):
        segment_start = i * segment_size
        segment_end = (i+1) * segment_size if i < num_frames - 1 else total_frames
        candidate = get_random_valid_frame(
            video_path, segment_start, segment_end,
            roi_coords, static_threshold=static_threshold, attempts=10,
            existing_frames=[cand['frame'] for cand in candidate_list if cand.get('frame') is not None]
        )
        if candidate is None:
            idx = np.random.randint(segment_start, segment_end) if segment_end > segment_start else segment_start
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            candidate = (idx, frame) if ret else (idx, None)
        candidate_list.append({
            "segment_start": segment_start,
            "segment_end": segment_end,
            "frame_index": candidate[0],
            "frame": candidate[1],
            "replace_count": 0
        })
    cap.release()
    return candidate_list

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

# ---------- Improved GUI Application ----------
class VideoFrameExtractorGUI:
    MIN_SEPARATION = 300

    def __init__(self, root):
        self.root = root
        self.root.title("Video Frame Extractor")
        self.root.configure(bg="white")
        self.root.geometry("1100x800")
        # Use a modern ttk theme ("clam")
        style = ttk.Style()
        style.theme_use("clam")
        # Define a custom style for modern, rounded buttons
        style.configure("Rounded.TButton",
                        font=("Segoe UI", 10, "bold"),
                        foreground="#ffffff",
                        background="#007ACC",
                        relief="flat",
                        padding=6)
        style.map("Rounded.TButton",
                  background=[('active', '#005F99'), ('disabled', '#d9d9d9')])

        # Set the application icon (change the path as needed)
        try:
            current_dir = os.getcwd()
            image_path = os.path.join(current_dir, "4. Frame Extraction", "FE.icns")
            icon = ImageTk.PhotoImage(file=image_path)
            self.root.iconphoto(True, icon)
        except Exception as e:
            print("Icon not found or failed to load:", e)

        # Create a menu bar
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

        # Bind new keys:
        root.bind("<Command-l>", lambda event: self.load_directory())
        root.bind("<Command-n>", lambda event: self.process_next_video())
        root.bind("<Command-s>", lambda event: self.save_frames())
        root.bind("<Control-s>", lambda event: self.open_settings())
        root.bind("<Control-q>", lambda event: root.quit())

        # Variables and defaults
        self.video_files = []
        self.current_video_index = 0
        self.current_video_path = None
        self.roi_coords = (0, 450, 640, 90)  # Default ROI (x, y, width, height)
        self.static_threshold = 30  # Default threshold
        self.num_candidates = 5     # Default number of candidate frames
        self.candidate_frames = []
        self.candidate_widgets = []

        # Set output directory
        self.output_dir = "/Volumes/Fili's SSD/FYP/Dataset_Frames"
        os.makedirs(self.output_dir, exist_ok=True)

        # Header frame with title
        header_frame = tk.Frame(root, bg="white")
        header_frame.pack(pady=10, fill=tk.X)
        title_label = tk.Label(header_frame, text="Video Frame Extractor", font=("Helvetica", 22, "bold"), bg="white")
        title_label.pack()

        # Control frame with buttons and info (centered)
        control_frame = tk.Frame(root, bg="white")
        control_frame.pack(pady=5, fill=tk.X)
        self.load_dir_btn = ttk.Button(control_frame, text="Load Video Directory", command=self.load_directory, style="Rounded.TButton")
        self.load_dir_btn.grid(row=0, column=0, padx=10)
        self.info_label = tk.Label(control_frame, text="Select a directory containing videos.", font=("Arial", 14), bg="white")
        self.info_label.grid(row=0, column=1, padx=20)
        self.settings_btn = ttk.Button(control_frame, text="Settings", command=self.open_settings, style="Rounded.TButton")
        self.settings_btn.grid(row=0, column=2, padx=10)
        # Configure grid to center the columns
        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_columnconfigure(1, weight=1)
        control_frame.grid_columnconfigure(2, weight=1)

        self.video_name_label = tk.Label(root, text="", font=("Arial", 18), bg="white")
        self.video_name_label.pack(pady=5)

        # Progress bar (hidden until needed)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(root, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(pady=5, fill=tk.X, padx=20)
        self.progress_bar.pack_forget()  # hide initially

        # Scrollable frame for candidate previews (grid layout)
        self.preview_container = tk.Frame(root, bg="white")
        self.preview_container.pack(pady=10, fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(self.preview_container, bg="white")
        self.preview_frame = tk.Frame(canvas, bg="white")
        vsb = ttk.Scrollbar(self.preview_container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas.create_window((0,0), window=self.preview_frame, anchor="nw")
        self.preview_frame.bind("<Configure>", lambda event, canvas=canvas: canvas.configure(scrollregion=canvas.bbox("all")))

        # Bottom frame with navigation buttons and status bar (centered)
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
        
        # Force an update so buttons are active immediately
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
        # Increase window height so all content is visible immediately
        settings_win.geometry("340x330")
        settings_win.configure(bg="white")
        # Bind Control+W to close the settings window
        settings_win.bind("<Control-w>", lambda event: settings_win.destroy())
        
        # Labels for ROI and threshold settings
        tk.Label(settings_win, text="ROI X:", bg="white", font=("Arial", 10)).grid(row=0, column=0, padx=10, pady=8, sticky="e")
        tk.Label(settings_win, text="ROI Y:", bg="white", font=("Arial", 10)).grid(row=1, column=0, padx=10, pady=8, sticky="e")
        tk.Label(settings_win, text="ROI Width:", bg="white", font=("Arial", 10)).grid(row=2, column=0, padx=10, pady=8, sticky="e")
        tk.Label(settings_win, text="ROI Height:", bg="white", font=("Arial", 10)).grid(row=3, column=0, padx=10, pady=8, sticky="e")
        tk.Label(settings_win, text="Static Threshold:", bg="white", font=("Arial", 10)).grid(row=4, column=0, padx=10, pady=8, sticky="e")
        tk.Label(settings_win, text="Number of Frames:", bg="white", font=("Arial", 10)).grid(row=5, column=0, padx=10, pady=8, sticky="e")
        
        # Variables for settings
        roi_x = tk.IntVar(value=self.roi_coords[0])
        roi_y = tk.IntVar(value=self.roi_coords[1])
        roi_w = tk.IntVar(value=self.roi_coords[2])
        roi_h = tk.IntVar(value=self.roi_coords[3])
        static_thresh = tk.IntVar(value=self.static_threshold)
        num_frames = tk.IntVar(value=self.num_candidates)
        
        # Entry fields for settings
        tk.Entry(settings_win, textvariable=roi_x).grid(row=0, column=1, padx=10, pady=8)
        tk.Entry(settings_win, textvariable=roi_y).grid(row=1, column=1, padx=10, pady=8)
        tk.Entry(settings_win, textvariable=roi_w).grid(row=2, column=1, padx=10, pady=8)
        tk.Entry(settings_win, textvariable=roi_h).grid(row=3, column=1, padx=10, pady=8)
        tk.Entry(settings_win, textvariable=static_thresh).grid(row=4, column=1, padx=10, pady=8)
        tk.Entry(settings_win, textvariable=num_frames).grid(row=5, column=1, padx=10, pady=8)
        
        def apply_settings():
            self.roi_coords = (roi_x.get(), roi_y.get(), roi_w.get(), roi_h.get())
            self.static_threshold = static_thresh.get()
            self.num_candidates = num_frames.get()
            settings_win.destroy()
            self.update_status("Settings updated.")
            
        ttk.Button(settings_win, text="Apply", command=apply_settings, style="Rounded.TButton").grid(row=6, column=0, columnspan=2, pady=20)
        
    def load_directory(self):
        directory = filedialog.askdirectory()
        if not directory:
            return
        self.video_files = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.lower().endswith(('.mp4', '.avi', '.mov'))
        ]
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
        existing_files = [f for f in os.listdir(self.output_dir) if f.startswith(video_name)]
        if len(existing_files) >= self.num_candidates:
            messagebox.showinfo("Info", f"{video_name} already processed. Skipping.")
            self.process_next_video()
            return
        self.video_name_label.config(text=f"Processing video: {video_name}")
        self.update_status(f"Extracting candidate frames from {video_name}...")
        self.progress_bar.pack(pady=5, fill=tk.X, padx=20)
        self.progress_var.set(0)
        self.root.update_idletasks()
        
        # Run frame extraction in a background thread
        threading.Thread(target=self.extract_frames_thread, args=(video_path,), daemon=True).start()

    def extract_frames_thread(self, video_path):
        self.progress_var.set(20)
        self.root.after(100, lambda: self.progress_var.set(40))
        candidate_frames = extract_candidate_frames(video_path, self.num_candidates, self.roi_coords, static_threshold=self.static_threshold)
        self.progress_var.set(80)
        self.root.after(0, self.on_frames_extracted, candidate_frames)

    def on_frames_extracted(self, candidate_frames):
        self.candidate_frames = candidate_frames
        self.display_candidates()
        self.progress_var.set(100)
        self.root.update_idletasks()
        self.progress_bar.pack_forget()
        self.update_status("Candidate frames loaded.")

    def display_candidates(self):
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        self.candidate_widgets = []
        cols = 3  # Changed to 3 columns
        for i, cand in enumerate(self.candidate_frames):
            frame_data = cand.get("frame")
            if frame_data is None:
                continue
            thumb_frame = tk.Frame(self.preview_frame, bd=2, relief=tk.GROOVE, bg="white")
            frame_rgb = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(frame_rgb)
            im_thumb = im_pil.resize((300, 200))
            photo = ImageTk.PhotoImage(im_thumb)
            cand["photo"] = photo

            img_label = tk.Label(thumb_frame, image=photo, bg="white")
            img_label.pack(padx=5, pady=5)
            img_label.bind("<Double-Button-1>", lambda e, img=im_pil: self.preview_full_image(img))
            
            idx_label = tk.Label(thumb_frame, text=f"Frame {cand.get('frame_index')}", bg="white", font=("Arial", 10))
            idx_label.pack(pady=2)
            replace_btn = ttk.Button(thumb_frame, text="Replace", command=lambda idx=i: self.replace_candidate(idx), style="Rounded.TButton")
            replace_btn.pack(pady=5, padx=5)
            row = i // cols
            col = i % cols
            thumb_frame.grid(row=row, column=col, padx=15, pady=10, sticky="nsew")
            self.candidate_widgets.append(thumb_frame)

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

    def replace_candidate(self, idx):
        cand = self.candidate_frames[idx]
        cand["replace_count"] += 1
        other_indices = [c["frame_index"] for i, c in enumerate(self.candidate_frames)
                         if i != idx and c.get("frame") is not None]

        if cand["replace_count"] > 10:
            cap = cv2.VideoCapture(self.current_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            new_candidate = fallback_random_frame(
                self.current_video_path,
                other_indices,
                min_separation=self.MIN_SEPARATION,
                roi_coords=self.roi_coords,
                static_threshold=self.static_threshold,
                attempts=20
            )
            if new_candidate is None:
                messagebox.showinfo("Info", "Unable to find a replacement candidate from a different segment.")
                return
            new_idx, new_frame = new_candidate
            new_segment_start = max(0, new_idx - 50)
            new_segment_end = min(total_frames, new_idx + 50)
            cand["segment_start"] = new_segment_start
            cand["segment_end"] = new_segment_end
        else:
            seg_start = cand["segment_start"]
            seg_end = cand["segment_end"]
            new_candidate = get_random_valid_frame(
                self.current_video_path,
                seg_start, seg_end,
                self.roi_coords,
                static_threshold=self.static_threshold,
                attempts=10,
                existing_frames=[c["frame"] for i, c in enumerate(self.candidate_frames)
                                 if i != idx and c.get("frame") is not None]
            )
            if new_candidate is None:
                cap = cv2.VideoCapture(self.current_video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                new_candidate = fallback_random_frame(
                    self.current_video_path,
                    other_indices,
                    min_separation=self.MIN_SEPARATION,
                    roi_coords=self.roi_coords,
                    static_threshold=self.static_threshold,
                    attempts=20
                )
                if new_candidate is None:
                    messagebox.showinfo("Info", "Unable to find a replacement candidate.")
                    return

        new_idx, new_frame = new_candidate
        cand["frame_index"] = new_idx
        cand["frame"] = new_frame
        frame_rgb = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(frame_rgb).resize((300, 200))
        photo = ImageTk.PhotoImage(im_pil)
        cand["photo"] = photo
        widget = self.candidate_widgets[idx]
        for child in widget.winfo_children():
            if isinstance(child, tk.Label) and child.cget("image"):
                child.config(image=photo)
                break

    def save_frames(self):
        video_name = os.path.splitext(os.path.basename(self.current_video_path))[0]
        for cand in self.candidate_frames:
            idx = cand.get("frame_index")
            frame = cand.get("frame")
            if frame is not None:
                output_path = os.path.join(self.output_dir, f"{video_name}_frame{idx}.png")
                cv2.imwrite(output_path, frame)
        self.update_status(f"Frames saved for {video_name}.")

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
