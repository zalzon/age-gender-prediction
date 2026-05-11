from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk, ImageOps

from src.services.keras_image_predictor import KerasImagePredictionService
from src.services.mock_predictor import MockPredictionService


class AgeGenderPredictionApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Age & Gender Prediction Dashboard")
        self.root.geometry("1180x720")
        self.root.minsize(1000, 640)

        self.image_predictor = KerasImagePredictionService()
        self.predictor = MockPredictionService()
        self.image_path: Path | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.live_capture: cv2.VideoCapture | None = None
        self.live_job: str | None = None
        self.live_frame_count = 0

        self.analysis_mode = tk.StringVar(value="upload")
        self.selected_model = tk.StringVar(value="MobileNetV2")
        self.status_text = tk.StringVar(value="Upload an image to begin.")
        self.gender_text = tk.StringVar(value="-")
        self.age_group_text = tk.StringVar(value="-")
        self.confidence_text = tk.StringVar(value="-")
        self.model_text = tk.StringVar(value="-")

        self._configure_style()
        self._build_layout()
        self._refresh_mode_ui()

    def _configure_style(self) -> None:
        self.root.configure(background="#f3f6fb")

        style = ttk.Style()
        style.theme_use("clam")

        style.configure("App.TFrame", background="#f3f6fb")
        style.configure("Card.TFrame", background="#ffffff", relief="flat")
        style.configure("Section.TLabel", background="#ffffff", foreground="#16324a", font=("Segoe UI", 14, "bold"))
        style.configure("Body.TLabel", background="#ffffff", foreground="#3f4b5a", font=("Segoe UI", 10))
        style.configure("Status.TLabel", background="#f3f6fb", foreground="#5b6574", font=("Segoe UI", 10))
        style.configure("ResultValue.TLabel", background="#ffffff", foreground="#0f172a", font=("Segoe UI", 16, "bold"))
        style.configure("ResultName.TLabel", background="#ffffff", foreground="#6b7280", font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(18, 11))
        style.map("Accent.TButton", foreground=[("!disabled", "white")], background=[("!disabled", "#1f5fbf"), ("active", "#184ea3")])
        style.configure("Ghost.TRadiobutton", background="#ffffff", foreground="#334155", font=("Segoe UI", 10))

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, style="App.TFrame", padding=18)
        container.pack(fill="both", expand=True)

        body = ttk.Frame(container, style="App.TFrame")
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=5)
        body.columnconfigure(1, weight=4)
        body.rowconfigure(0, weight=1)

        left_card = ttk.Frame(body, style="Card.TFrame", padding=20)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right_card = ttk.Frame(body, style="Card.TFrame", padding=20)
        right_card.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        self._build_upload_section(left_card)
        self._build_results_section(right_card)

        footer = ttk.Label(
            container,
            textvariable=self.status_text,
            style="Status.TLabel",
            anchor="w",
        )
        footer.pack(fill="x", pady=(10, 0))

    def _build_upload_section(self, parent: ttk.Frame) -> None:
        # Use grid so the preview is constrained and controls remain visible
        parent.columnconfigure(0, weight=1)

        ttk.Label(parent, text="Image Input", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            parent,
            text="Switch between uploaded image analysis and live webcam analysis.",
            style="Body.TLabel",
            wraplength=470,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(6, 12))

        # Mode selector
        mode_row = ttk.Frame(parent, style="Card.TFrame")
        mode_row.grid(row=2, column=0, sticky="w", pady=(0, 12))
        ttk.Label(mode_row, text="Analysis Mode", style="Section.TLabel").pack(anchor="w")
        mode_buttons = ttk.Frame(mode_row, style="Card.TFrame")
        mode_buttons.pack(anchor="w", pady=(4, 0))
        ttk.Radiobutton(
            mode_buttons,
            text="Upload Image",
            value="upload",
            variable=self.analysis_mode,
            style="Ghost.TRadiobutton",
            command=self._refresh_mode_ui,
        ).pack(side="left", padx=(0, 18))
        ttk.Radiobutton(
            mode_buttons,
            text="Live Camera",
            value="live",
            variable=self.analysis_mode,
            style="Ghost.TRadiobutton",
            command=self._refresh_mode_ui,
        ).pack(side="left")

        # Controls rows (fixed size)
        self.upload_controls_frame = ttk.Frame(parent, style="Card.TFrame")
        self.upload_controls_frame.grid(row=3, column=0, sticky="w")
        ttk.Button(self.upload_controls_frame, text="Upload Image", style="Accent.TButton", command=self._upload_image).pack(side="left")
        ttk.Label(self.upload_controls_frame, text="PNG, JPG, JPEG, BMP, GIF, WEBP", style="Body.TLabel").pack(side="left", padx=12)

        self.live_controls_frame = ttk.Frame(parent, style="Card.TFrame")
        # live controls are created but hidden until needed
        ttk.Button(self.live_controls_frame, text="Start Live Analysis", style="Accent.TButton", command=self._start_live_analysis).pack(side="left")
        ttk.Button(self.live_controls_frame, text="Stop", style="Accent.TButton", command=self._stop_live_analysis).pack(side="left", padx=(10, 0))

        selector_frame = ttk.Frame(parent, style="Card.TFrame")
        selector_frame.grid(row=4, column=0, sticky="we", pady=(8, 8))
        selector_frame.columnconfigure(0, weight=1)
        ttk.Label(selector_frame, text="Choose Model", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            selector_frame,
            text="Pick one model for image analysis.",
            style="Body.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 10))

        radio_row = ttk.Frame(selector_frame, style="Card.TFrame")
        radio_row.grid(row=2, column=0, sticky="w")
        ttk.Radiobutton(
            radio_row,
            text="MobileNetV2",
            value="MobileNetV2",
            variable=self.selected_model,
            style="Ghost.TRadiobutton",
        ).pack(side="left", padx=(0, 18))
        ttk.Radiobutton(
            radio_row,
            text="NASNetMobile",
            value="NASNetMobile",
            variable=self.selected_model,
            style="Ghost.TRadiobutton",
        ).pack(side="left")

        action_row = ttk.Frame(parent, style="Card.TFrame")
        action_row.grid(row=5, column=0, sticky="we", pady=(6, 0))
        action_row.columnconfigure(0, weight=1)
        ttk.Button(action_row, text="Predict", style="Accent.TButton", command=self._predict).grid(row=0, column=0, sticky="we")

        # Preview gets the flexible remaining space so it cannot push controls away
        preview_frame = ttk.Frame(parent, style="Card.TFrame")
        preview_frame.grid(row=6, column=0, sticky="nsew", pady=(12, 0))
        parent.rowconfigure(6, weight=1)
        preview_frame.configure(height=300)
        preview_frame.grid_propagate(False)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)

        ttk.Label(preview_frame, text="Preview", style="Section.TLabel").grid(row=0, column=0, sticky="w")

        preview_border = tk.Frame(preview_frame, background="#ffffff", highlightbackground="#dbe3ef", highlightthickness=1)
        preview_border.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        preview_border.configure(height=220)
        preview_border.grid_propagate(False)
        preview_border.rowconfigure(0, weight=1)
        preview_border.columnconfigure(0, weight=1)

        self.preview_label = ttk.Label(
            preview_border,
            text="No image selected",
            style="Body.TLabel",
            anchor="center",
            justify="center",
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")

    def _build_results_section(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Results", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text="Mock output updates immediately for testing.",
            style="Body.TLabel",
            wraplength=360,
            justify="left",
        ).pack(anchor="w", pady=(6, 14))

        result_card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        result_card.pack(fill="both", expand=True)

        self._add_result_row(result_card, "Gender", self.gender_text)
        self._add_result_row(result_card, "Age Group", self.age_group_text)
        self._add_result_row(result_card, "Confidence", self.confidence_text)
        self._add_result_row(result_card, "Model", self.model_text)

        ttk.Separator(result_card).pack(fill="x", pady=16)

        note = ttk.Label(
            result_card,
            text="Upload-image predictions use the selected model.",
            style="Body.TLabel",
            wraplength=340,
            justify="left",
        )
        note.pack(anchor="w")

        self.live_status_text = tk.StringVar(value="Live analysis is off.")
        ttk.Label(
            result_card,
            textvariable=self.live_status_text,
            style="Body.TLabel",
            wraplength=340,
            justify="left",
        ).pack(anchor="w", pady=(12, 0))

    def _add_result_row(self, parent: ttk.Frame, label: str, variable: tk.StringVar) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=8)
        row.columnconfigure(0, weight=1)
        row.columnconfigure(1, weight=0)
        ttk.Label(row, text=label, style="ResultName.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(row, textvariable=variable, style="ResultValue.TLabel").grid(row=0, column=1, sticky="e")

    def _upload_image(self) -> None:
        if self.analysis_mode.get() == "live":
            self.analysis_mode.set("upload")
            self._refresh_mode_ui()

        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        self.image_path = Path(file_path)
        try:
            image = Image.open(self.image_path).convert("RGB")
            preview = ImageOps.contain(image, (380, 260))
            self.preview_photo = ImageTk.PhotoImage(preview)
            self.preview_label.configure(image=self.preview_photo, text="")
            self.status_text.set(f"Loaded image: {self.image_path.name}")
        except Exception as exc:  # pragma: no cover - UI error path
            self.image_path = None
            self.preview_photo = None
            self.preview_label.configure(image="", text="No image selected")
            messagebox.showerror("Image Error", f"Could not open the selected image.\n\n{exc}")
            self.status_text.set("Image loading failed.")

    def _predict(self) -> None:
        if self.image_path is None:
            messagebox.showwarning("Missing Image", "Please upload an image before running prediction.")
            return

        try:
            result = self.image_predictor.predict(self.image_path, self.selected_model.get())
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            messagebox.showerror("Model Error", str(exc))
            self.status_text.set("Image prediction failed.")
            return

        self.gender_text.set(result.predicted_gender)
        self.age_group_text.set(result.predicted_age_group)
        self.confidence_text.set(f"{result.confidence_score:.2f}%")
        self.model_text.set(result.selected_model)
        self.status_text.set(f"Predicted using {result.selected_model} for {self.image_path.name}.")

    def _refresh_mode_ui(self) -> None:
        if self.analysis_mode.get() == "live":
            try:
                self.upload_controls_frame.grid_remove()
            except Exception:
                pass
            # show live controls
            self.live_controls_frame.grid(row=3, column=0, sticky="w")
            self.preview_label.configure(text="Live camera is stopped.", image="")
            self.live_status_text.set("Live analysis is off.")
        else:
            try:
                self.live_controls_frame.grid_remove()
            except Exception:
                pass
            self.upload_controls_frame.grid(row=3, column=0, sticky="w")
            if self.live_capture is not None:
                self._stop_live_analysis()

    def _start_live_analysis(self) -> None:
        if self.live_capture is not None:
            return

        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            messagebox.showerror("Camera Error", "Could not open the webcam.")
            capture.release()
            self.live_status_text.set("Live analysis could not start.")
            return

        self.live_capture = capture
        self.live_frame_count = 0
        self.status_text.set("Live analysis started.")
        self.live_status_text.set("Live analysis running.")
        self._update_live_frame()

    def _update_live_frame(self) -> None:
        if self.live_capture is None:
            return

        success, frame = self.live_capture.read()
        if not success:
            self.live_status_text.set("Live camera stopped sending frames.")
            self._stop_live_analysis()
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        preview = ImageOps.contain(pil_image, (380, 260))
        self.preview_photo = ImageTk.PhotoImage(preview)
        self.preview_label.configure(image=self.preview_photo, text="")

        self.live_frame_count += 1
        if self.live_frame_count % 2 == 0:
            result = self.predictor.predict(self.selected_model.get())
            self.gender_text.set(result.predicted_gender)
            self.age_group_text.set(result.predicted_age_group)
            self.confidence_text.set(f"{result.confidence_score:.2f}%")
            self.model_text.set(result.selected_model)

        self.live_status_text.set("Live analysis running.")
        self.status_text.set("Live camera frame updated.")
        self.live_job = self.root.after(500, self._update_live_frame)

    def _stop_live_analysis(self) -> None:
        if self.live_job is not None:
            try:
                self.root.after_cancel(self.live_job)
            except tk.TclError:
                pass
            self.live_job = None

        if self.live_capture is not None:
            self.live_capture.release()
            self.live_capture = None

        try:
            self.live_status_text.set("Live analysis is off.")
            if self.analysis_mode.get() == "live":
                self.preview_label.configure(text="Live camera is stopped.", image="")
        except tk.TclError:
            pass

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            self._stop_live_analysis()
