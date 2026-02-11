import os
import sys
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import math
import csv

missing_deps = []
try:
    import numpy as np
except ImportError:
    missing_deps.append("numpy")

try:
    from PIL import Image, ImageTk
except ImportError:
    missing_deps.append("pillow")

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    matplotlib.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
except ImportError:
    missing_deps.append("matplotlib")


if missing_deps:
    deps = " ".join(missing_deps)
    sys.exit(f"Please install dependencies first: pip install {deps}")

# Default pixel-to-nanometer ratio (nm/px)
PIXEL_TO_NM_DEFAULT = 108.33
# Default sampling step (nm)
STEP_NM_DEFAULT = 50


def bresenham_line(x0: int, y0: int, x1: int, y1: int):
    """Generate pixel coordinates for a line between two points."""
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points


class LineProfileApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EasyColoc — A design by ChenLab_Jian Lin")

        self.image_paths = [None, None, None]
        self.images = [None, None, None]
        self.canvas_image = None
        self.line_coords_display = None  # (x0,y0,x1,y1) on canvas
        self.dragging_anchor = None  # 0=start, 1=end
        # Line thickness in nanometers
        self.thickness_var = tk.IntVar(value=500)
        self.pixel_size_var = tk.DoubleVar(value=PIXEL_TO_NM_DEFAULT)
        self.step_nm_var = tk.IntVar(value=STEP_NM_DEFAULT)
        self.use_ch3_var = tk.BooleanVar(value=False)
        self.output_dir_var = tk.StringVar(value=os.getcwd())
        self.save_raw_var = tk.BooleanVar(value=True)
        self.save_cal_var = tk.BooleanVar(value=True)
        self.save_csv_var = tk.BooleanVar(value=True)
        self.raw_fig = None
        self.cal_fig = None
        self.raw_ax = None
        self.cal_ax = None
        self.raw_canvas_widget = None
        self.cal_canvas_widget = None
        self.status_var = tk.StringVar(value="Ready")

        # Canvas scaling
        self.canvas_width = 640
        self.canvas_height = 640
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(top)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="Select Channel 1 Image", command=lambda: self.load_image(0)).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(btn_frame, text="Select Channel 2 Image", command=lambda: self.load_image(1)).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Checkbutton(
            btn_frame, text="Enable Channel 3", variable=self.use_ch3_var, command=self.update_preview
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Select Channel 3 Image", command=lambda: self.load_image(2)).pack(
            side=tk.LEFT, padx=5
        )
        self.analyze_btn = ttk.Button(btn_frame, text="Start Analysis", command=self.analyze)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        self.analyze_btn.state(["disabled"])

        thickness_frame = ttk.Frame(top)
        thickness_frame.pack(fill=tk.X, pady=2)
        ttk.Label(thickness_frame, text="Line width (nm, drag):").pack(side=tk.LEFT)
        self.thickness_scale = ttk.Scale(
            thickness_frame,
            from_=100,
            to=2000,
            orient=tk.HORIZONTAL,
            variable=self.thickness_var,
            command=lambda _: self._on_thickness_change(),
        )
        self.thickness_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 8))
        ttk.Label(thickness_frame, text="Current").pack(side=tk.LEFT, padx=(0, 2))
        self.thickness_entry = ttk.Entry(
            thickness_frame, width=6, textvariable=self.thickness_var
        )
        self.thickness_entry.pack(side=tk.LEFT, padx=(0, 6))
        self.thickness_entry.bind("<Return>", lambda _: self._on_thickness_change())
        self.thickness_entry.bind("<FocusOut>", lambda _: self._on_thickness_change())
        ttk.Label(thickness_frame, text="Default 500 nm; thickness bounds shown in preview").pack(side=tk.LEFT)

        pixel_frame = ttk.Frame(top)
        pixel_frame.pack(fill=tk.X, pady=2)
        ttk.Label(pixel_frame, text="Pixel size (nm/px):").pack(side=tk.LEFT)
        pixel_entry = ttk.Entry(pixel_frame, width=8, textvariable=self.pixel_size_var)
        pixel_entry.pack(side=tk.LEFT, padx=(6, 8))
        pixel_entry.bind("<Return>", lambda _: self._on_pixel_size_change())
        pixel_entry.bind("<FocusOut>", lambda _: self._on_pixel_size_change())
        ttk.Label(pixel_frame, text=f"Default {PIXEL_TO_NM_DEFAULT}").pack(side=tk.LEFT)

        step_frame = ttk.Frame(top)
        step_frame.pack(fill=tk.X, pady=2)
        ttk.Label(step_frame, text="Interpolation step (nm):").pack(side=tk.LEFT)
        step_entry = ttk.Entry(step_frame, width=6, textvariable=self.step_nm_var)
        step_entry.pack(side=tk.LEFT, padx=(6, 8))
        step_entry.bind("<Return>", lambda _: self._on_step_change())
        step_entry.bind("<FocusOut>", lambda _: self._on_step_change())
        ttk.Label(step_frame, text=f"Default {STEP_NM_DEFAULT}").pack(side=tk.LEFT)

        out_frame = ttk.Frame(top)
        out_frame.pack(fill=tk.X, pady=4)
        ttk.Label(out_frame, text="Output folder:").pack(side=tk.LEFT)
        out_entry = ttk.Entry(out_frame, textvariable=self.output_dir_var)
        out_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
        ttk.Button(out_frame, text="Browse…", command=self._choose_output_dir).pack(
            side=tk.LEFT
        )
        ttk.Label(out_frame, text="Default is current working directory").pack(side=tk.LEFT, padx=(6, 0))

        output_opts = ttk.Frame(top)
        output_opts.pack(fill=tk.X, pady=2)
        ttk.Label(output_opts, text="Outputs:").pack(side=tk.LEFT)
        ttk.Checkbutton(output_opts, text="Raw plots", variable=self.save_raw_var).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Checkbutton(output_opts, text="Calibrated plots", variable=self.save_cal_var).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Checkbutton(output_opts, text="CSV", variable=self.save_csv_var).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        self.path_label = ttk.Label(
            top, text="No images selected", foreground="#555", anchor="w", padding=(0, 5)
        )
        self.path_label.pack(fill=tk.X)

        content = ttk.Frame(top)
        content.pack(fill=tk.BOTH, expand=True)

        canvas_frame = ttk.Frame(content, borderwidth=1, relief=tk.SOLID)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#f5f5f5",
            cursor="cross",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        plot_frame = ttk.Frame(content, borderwidth=1, relief=tk.SOLID, width=360)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(6, 0))
        plot_frame.pack_propagate(False)
        self._init_plot_canvases(plot_frame)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        tip = (
            "Steps: 1) Select two images; 2) Draw a line on the left preview;"
            " 3) Click Start Analysis to generate raw/calibrated plots and CSV."
        )
        ttk.Label(top, text=tip, foreground="#333", padding=(0, 8)).pack(fill=tk.X)

        status_bar = ttk.Frame(self.root, relief=tk.SUNKEN, padding=(6, 2))
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Label(status_bar, textvariable=self.status_var, anchor="w").pack(
            fill=tk.X
        )

    def load_image(self, idx: int):
        path = filedialog.askopenfilename(
            title=f"Select Channel {idx + 1} Image",
            filetypes=[
                ("Image files", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            img = Image.open(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Failed to read image: {exc}")
            return

        # Convert to grayscale for processing
        if img.mode not in ("L", "I;16", "I"):
            img = img.convert("L")

        self.images[idx] = img
        self.image_paths[idx] = path

        info = [
            f"Ch1: {os.path.basename(self.image_paths[0]) if self.image_paths[0] else 'None'}",
            f"Ch2: {os.path.basename(self.image_paths[1]) if self.image_paths[1] else 'None'}",
            f"Ch3: {os.path.basename(self.image_paths[2]) if self.image_paths[2] else 'None'}",
        ]
        self.path_label.config(text=" | ".join(info))
        self._update_status("Ready")

        # Preview composite based on Channel 1
        if self.images[0] is not None:
            self.update_preview()

        if self.images[0] is not None and self.images[1] is not None:
            self.analyze_btn.state(["!disabled"])

    def update_preview(self):
        """Compose channels into preview and store scaling parameters."""
        if self.images[0] is None:
            return

        img1 = self.images[0]
        img2 = self.images[1]
        img3 = self.images[2] if self.use_ch3_var.get() else None

        # Match sizes
        if img2 is not None and img2.size != img1.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
        if img3 is not None and img3.size != img1.size:
            img3 = img3.resize(img1.size, Image.Resampling.LANCZOS)

        arr1 = np.asarray(img1, dtype=float)
        arr1 = arr1 - arr1.min()
        denom1 = arr1.max()
        arr1 = arr1 / denom1 * 255 if denom1 > 0 else np.zeros_like(arr1)
        g1 = np.clip(arr1, 0, 255)

        if img2 is not None:
            arr2 = np.asarray(img2, dtype=float)
            arr2 = arr2 - arr2.min()
            denom2 = arr2.max()
            arr2 = arr2 / denom2 * 255 if denom2 > 0 else np.zeros_like(arr2)
            c2 = np.clip(arr2, 0, 255)
        else:
            c2 = np.zeros_like(g1)

        if img3 is not None:
            arr3 = np.asarray(img3, dtype=float)
            arr3 = arr3 - arr3.min()
            denom3 = arr3.max()
            arr3 = arr3 / denom3 * 255 if denom3 > 0 else np.zeros_like(arr3)
            c3 = np.clip(arr3, 0, 255)
        else:
            c3 = np.zeros_like(g1)

        # Color mapping: Ch1=Green, Ch2=Magenta (R+B), Ch3=Yellow (R+G)
        r = np.clip(c2 + c3, 0, 255).astype(np.uint8)
        g = np.clip(g1 + c3, 0, 255).astype(np.uint8)
        b = np.clip(c2, 0, 255).astype(np.uint8)

        rgb_arr = np.stack([r, g, b], axis=2)
        img = Image.fromarray(rgb_arr)

        scale = min(
            self.canvas_width / img.width,
            self.canvas_height / img.height,
        )
        scale = max(scale, 1e-6)
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        bg = Image.new("RGB", (self.canvas_width, self.canvas_height), "#f5f5f5")
        self.offset_x = (self.canvas_width - new_w) // 2
        self.offset_y = (self.canvas_height - new_h) // 2
        bg.paste(resized.convert("RGB"), (self.offset_x, self.offset_y))

        self.scale = scale
        self.canvas_image = ImageTk.PhotoImage(bg)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_image)
        if self.line_coords_display:
            self._draw_line_on_canvas(*self.line_coords_display)

    def _on_press(self, event):
        hit = self._hit_anchor(event.x, event.y)
        if hit is not None and self.line_coords_display:
            x0, y0, x1, y1 = self.line_coords_display
            if hit == 0:
                self.line_coords_display = (event.x, event.y, x1, y1)
            else:
                self.line_coords_display = (x0, y0, event.x, event.y)
            self.dragging_anchor = hit
        else:
            self.dragging_anchor = None
            self.line_coords_display = (event.x, event.y, event.x, event.y)
        self._redraw_canvas()

    def _on_drag(self, event):
        if not self.line_coords_display:
            return
        x0, y0, x1, y1 = self.line_coords_display
        if self.dragging_anchor == 0:
            self.line_coords_display = (event.x, event.y, x1, y1)
        elif self.dragging_anchor == 1:
            self.line_coords_display = (x0, y0, event.x, event.y)
        else:
            self.line_coords_display = (x0, y0, event.x, event.y)
        self._redraw_canvas()

    def _on_release(self, event):
        if not self.line_coords_display:
            return
        x0, y0, x1, y1 = self.line_coords_display
        if self.dragging_anchor == 0:
            self.line_coords_display = (event.x, event.y, x1, y1)
        elif self.dragging_anchor == 1:
            self.line_coords_display = (x0, y0, event.x, event.y)
        else:
            self.line_coords_display = (x0, y0, event.x, event.y)
        self.dragging_anchor = None
        self._redraw_canvas()

    def _redraw_canvas(self):
        if self.canvas_image:
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_image)
        if self.line_coords_display:
            self._draw_line_on_canvas(*self.line_coords_display)

    def _draw_line_on_canvas(self, x0, y0, x1, y1):
        thickness_nm = max(1, self.thickness_var.get())
        thickness_px = max(1.0, thickness_nm / self._get_pixel_size())
        # Center line
        self.canvas.create_line(x0, y0, x1, y1, fill="red", width=2)

        if thickness_px <= 1:
            return

        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length == 0:
            return
        # Unit vector perpendicular to the line
        perp_x = -dy / length
        perp_y = dx / length

        offset = ((thickness_px - 1.0) / 2.0) * self.scale
        offset = max(offset, 1.0)

        # Boundary lines
        ux0 = x0 + perp_x * offset
        uy0 = y0 + perp_y * offset
        ux1 = x1 + perp_x * offset
        uy1 = y1 + perp_y * offset

        lx0 = x0 - perp_x * offset
        ly0 = y0 - perp_y * offset
        lx1 = x1 - perp_x * offset
        ly1 = y1 - perp_y * offset

        self.canvas.create_line(ux0, uy0, ux1, uy1, fill="#1f77b4", width=1, dash=(4, 2))
        self.canvas.create_line(lx0, ly0, lx1, ly1, fill="#1f77b4", width=1, dash=(4, 2))

        # End-point anchors for dragging
        r = 6
        self.canvas.create_oval(x0 - r, y0 - r, x0 + r, y0 + r, fill="white", outline="red", width=2)
        self.canvas.create_oval(x1 - r, y1 - r, x1 + r, y1 + r, fill="white", outline="red", width=2)

    def _hit_anchor(self, x, y, radius=8):
        """Check if mouse hits endpoint anchor; return 0/1 or None."""
        if not self.line_coords_display:
            return None
        x0, y0, x1, y1 = self.line_coords_display
        if (x - x0) ** 2 + (y - y0) ** 2 <= radius ** 2:
            return 0
        if (x - x1) ** 2 + (y - y1) ** 2 <= radius ** 2:
            return 1
        return None

    def _display_to_original(self, x: float, y: float):
        """Map canvas coordinates back to original image coordinates."""
        ox = (x - self.offset_x) / self.scale
        oy = (y - self.offset_y) / self.scale
        ox = int(np.clip(ox, 0, self.images[0].width - 1))
        oy = int(np.clip(oy, 0, self.images[0].height - 1))
        return ox, oy

    def _on_thickness_change(self):
        # Normalize input
        try:
            val = int(self.thickness_var.get())
        except Exception:
            val = 100
        val = max(100, min(val, 2000))
        self.thickness_var.set(val)
        if self.line_coords_display:
            self._redraw_canvas()

    def _on_pixel_size_change(self):
        try:
            val = float(self.pixel_size_var.get())
        except Exception:
            val = PIXEL_TO_NM_DEFAULT
        val = max(1e-6, val)
        self.pixel_size_var.set(val)
        if self.line_coords_display:
            self._redraw_canvas()

    def _on_step_change(self):
        try:
            val = int(self.step_nm_var.get())
        except Exception:
            val = STEP_NM_DEFAULT
        val = max(1, val)
        self.step_nm_var.set(val)

    def _extract_profile(self, arr: np.ndarray, start, end, thickness: int):
        x0, y0 = start
        x1, y1 = end
        if x0 == x1 and y0 == y1:
            return np.array([], dtype=float)
        coords = bresenham_line(x0, y0, x1, y1)
        values = []
        h, w = arr.shape
        dx = x1 - x0
        dy = y1 - y0
        length = float(np.hypot(dx, dy))
        if length == 0:
            return np.array([], dtype=float)
        # Unit vector perpendicular to the line
        perp = (-dy / length, dx / length)
        thick = max(1, thickness)
        half = (thick - 1) / 2.0

        for x, y in coords:
            if thick == 1:
                if 0 <= x < w and 0 <= y < h:
                    values.append(float(arr[y, x]))
                continue
            acc = []
            for offset in np.linspace(-half, half, thick):
                ox = int(round(x + perp[0] * offset))
                oy = int(round(y + perp[1] * offset))
                if 0 <= ox < w and 0 <= oy < h:
                    acc.append(float(arr[oy, ox]))
            if acc:
                values.append(float(np.mean(acc)))
        return np.array(values, dtype=float)

    def analyze(self):
        if self.images[0] is None or self.images[1] is None:
            messagebox.showwarning("Notice", "Please select Channel 1 and Channel 2 images first.")
            return
        if self.use_ch3_var.get() and self.images[2] is None:
            messagebox.showwarning("Notice", "Channel 3 is enabled. Please select a Channel 3 image.")
            return
        if not self.line_coords_display:
            messagebox.showwarning("Notice", "Please draw a line on the preview.")
            return

        x0, y0, x1, y1 = self.line_coords_display
        start = self._display_to_original(x0, y0)
        end = self._display_to_original(x1, y1)
        thickness_nm = max(1, self.thickness_var.get())
        pixel_size = self._get_pixel_size()
        step_nm = self._get_step_nm()
        thickness_px = max(1, int(round(thickness_nm / pixel_size)))

        try:
            arr1 = np.asarray(self.images[0], dtype=float)
            arr2 = np.asarray(self.images[1], dtype=float)
            arr3 = (
                np.asarray(self.images[2], dtype=float) if self.use_ch3_var.get() and self.images[2] else None
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Image conversion failed: {exc}")
            return

        raw1 = self._extract_profile(arr1, start, end, thickness_px)
        raw2 = self._extract_profile(arr2, start, end, thickness_px)
        raw3 = self._extract_profile(arr3, start, end, thickness_px) if arr3 is not None else None

        if raw1.size == 0 or raw2.size == 0:
            messagebox.showerror("Error", "The line is out of bounds; no data extracted.")
            return
        if self.use_ch3_var.get() and (raw3 is None or raw3.size == 0):
            messagebox.showerror("Error", "Channel 3 line is out of bounds; no data extracted.")
            return

        # Build distance axis (nm)
        base_distance_nm = np.arange(len(raw1)) * pixel_size

        # If only one point, skip interpolation
        if base_distance_nm.size < 2:
            interp_distance_nm = np.round(base_distance_nm).astype(int)
            raw1_interp = raw1
            raw2_interp = raw2
            raw3_interp = raw3 if raw3 is not None else None
        else:
            # Resample using custom step
            max_dist = float(base_distance_nm[-1])
            interp_distance_nm = np.arange(0, int(max_dist) + 1, step_nm, dtype=float)
            raw1_interp = np.interp(interp_distance_nm, base_distance_nm, raw1)
            raw2_interp = np.interp(interp_distance_nm, base_distance_nm, raw2)
            raw3_interp = (
                np.interp(interp_distance_nm, base_distance_nm, raw3) if raw3 is not None else None
            )
            interp_distance_nm = np.round(interp_distance_nm).astype(int)

        cal1 = self._calibrate(raw1_interp)
        cal2 = self._calibrate(raw2_interp)
        cal3 = self._calibrate(raw3_interp) if raw3_interp is not None else None
        self._save_outputs(interp_distance_nm, raw1_interp, raw2_interp, raw3_interp, cal1, cal2, cal3)
        self._update_plot_canvases(interp_distance_nm, raw1_interp, raw2_interp, raw3_interp, cal1, cal2, cal3)
        self._update_status("Analysis complete")

    def _choose_output_dir(self):
        """Choose output directory."""
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_dir_var.set(path)

    @staticmethod
    def _calibrate(arr: np.ndarray):
        if arr.size == 0:
            return np.zeros_like(arr)
        min_val = float(arr.min())
        max_val = float(arr.max())
        span = max_val - min_val
        if span <= 0:
            return np.zeros_like(arr)
        scaled = (arr - min_val) / span * 255.0
        return np.clip(scaled, 0, 255)

    def _init_plot_canvases(self, parent: ttk.Frame):
        """Initialize raw/calibrated plot preview."""
        raw_frame = ttk.LabelFrame(parent, text="Raw plot preview")
        raw_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 3))
        self.raw_fig = Figure(figsize=(3.6, 2.5), dpi=100)
        self.raw_ax = self.raw_fig.add_subplot(111)
        self.raw_ax.text(0.5, 0.5, "Waiting", ha="center", va="center")
        canvas_raw = FigureCanvasTkAgg(self.raw_fig, master=raw_frame)
        self.raw_canvas_widget = canvas_raw.get_tk_widget()
        self.raw_canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas_raw.draw()

        cal_frame = ttk.LabelFrame(parent, text="Calibrated plot preview")
        cal_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(3, 6))
        self.cal_fig = Figure(figsize=(3.6, 2.5), dpi=100)
        self.cal_ax = self.cal_fig.add_subplot(111)
        self.cal_ax.text(0.5, 0.5, "Waiting", ha="center", va="center")
        canvas_cal = FigureCanvasTkAgg(self.cal_fig, master=cal_frame)
        self.cal_canvas_widget = canvas_cal.get_tk_widget()
        self.cal_canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas_cal.draw()

        self.raw_canvas = canvas_raw
        self.cal_canvas = canvas_cal

    def _update_plot_canvases(self, x, raw1, raw2, raw3, cal1, cal2, cal3):
        """Update right-side preview plots."""
        if self.raw_ax is None or self.cal_ax is None:
            return
        # Raw
        self.raw_ax.clear()
        # Channel colors: Channel1 green, Channel2 magenta
        self.raw_ax.plot(x, raw1, label="Channel 1", color="green")
        self.raw_ax.plot(x, raw2, label="Channel 2", color="magenta")
        if raw3 is not None:
            self.raw_ax.plot(x, raw3, label="Channel 3", color="yellow")
        self.raw_ax.set_xlabel("Distance (nm)")
        self.raw_ax.set_ylabel("Intensity")
        self.raw_ax.legend()
        self.raw_fig.tight_layout()
        self.raw_canvas.draw()

        # Calibrated
        self.cal_ax.clear()
        self.cal_ax.plot(x, cal1, label="Channel 1 (cal)", color="green")
        self.cal_ax.plot(x, cal2, label="Channel 2 (cal)", color="magenta")
        if cal3 is not None:
            self.cal_ax.plot(x, cal3, label="Channel 3 (cal)", color="yellow")
        self.cal_ax.set_xlabel("Distance (nm)")
        self.cal_ax.set_ylabel("Intensity (scaled to 255)")
        self.cal_ax.legend()
        self.cal_fig.tight_layout()
        self.cal_canvas.draw()

    def _save_outputs(self, distance_nm, raw1, raw2, raw3, cal1, cal2, cal3):
        if not any(
            [self.save_raw_var.get(), self.save_cal_var.get(), self.save_csv_var.get()]
        ):
            messagebox.showwarning("Notice", "Please select at least one output type.")
            return

        base_dir = self.output_dir_var.get().strip() or os.getcwd()
        base_dir = os.path.abspath(base_dir)
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save failed", f"Cannot create output directory: {exc}")
            return

        ch1_name = os.path.basename(self.image_paths[0]) if self.image_paths[0] else "channel1"
        stem, _ = os.path.splitext(ch1_name)
        saved_paths = []

        if self.save_raw_var.get():
            plt.figure(figsize=(8, 4))
            plt.plot(distance_nm, raw1, label="Channel 1", color="green")
            plt.plot(distance_nm, raw2, label="Channel 2", color="magenta")
            if raw3 is not None:
                plt.plot(distance_nm, raw3, label="Channel 3", color="yellow")
            plt.xlabel("Distance (nm)")
            plt.ylabel("Intensity")
            plt.title("Raw Plot Profile")
            plt.legend()
            plt.tight_layout()
            raw_plot = os.path.join(base_dir, f"raw_{stem}.png")
            plt.savefig(raw_plot, dpi=150)
            plt.close()
            saved_paths.append(raw_plot)

        if self.save_cal_var.get():
            plt.figure(figsize=(8, 4))
            plt.plot(distance_nm, cal1, label="Channel 1 (cal)", color="green")
            plt.plot(distance_nm, cal2, label="Channel 2 (cal)", color="magenta")
            if cal3 is not None:
                plt.plot(distance_nm, cal3, label="Channel 3 (cal)", color="yellow")
            plt.xlabel("Distance (nm)")
            plt.ylabel("Intensity (scaled to 255)")
            plt.title("Calibrated Plot Profile")
            plt.legend()
            plt.tight_layout()
            cal_plot = os.path.join(base_dir, f"cal_{stem}.png")
            plt.savefig(cal_plot, dpi=150)
            plt.close()
            saved_paths.append(cal_plot)

        if self.save_csv_var.get():
            csv_path = os.path.join(base_dir, f"profile_{stem}.csv")
            header = [
                "distance_nm",
                "channel1_calibrated",
                "channel2_calibrated",
                "channel3_calibrated" if cal3 is not None else None,
                "channel1_raw",
                "channel2_raw",
                "channel3_raw" if raw3 is not None else None,
            ]
            header = [h for h in header if h is not None]
            cols = [distance_nm, cal1, cal2, cal3, raw1, raw2, raw3]
            cols = [c for c in cols if c is not None]
            rows = zip(*cols)
            try:
                with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(rows)
                saved_paths.append(csv_path)
            except Exception as e:
                messagebox.showerror("Save failed", str(e))
                return

        self._update_status("Analysis complete")

    def _update_status(self, text: str):
        self.status_var.set(text)

    def _get_pixel_size(self) -> float:
        try:
            val = float(self.pixel_size_var.get())
        except Exception:
            val = PIXEL_TO_NM_DEFAULT
        return max(1e-6, val)

    def _get_step_nm(self) -> int:
        try:
            val = int(self.step_nm_var.get())
        except Exception:
            val = STEP_NM_DEFAULT
        return max(1, val)


def main():
    root = tk.Tk()
    app = LineProfileApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

