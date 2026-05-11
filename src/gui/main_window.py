from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import threading
from PIL import Image, ImageTk, ImageOps

from src.services.keras_image_predictor import KerasImagePredictionService
# Mock predictor removed; live-mode inference is disabled until implemented


class AgeGenderPredictionApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Age & Gender Prediction Dashboard")
        # Start fullscreen (user can still resize)
        self.root.state('zoomed')
        self.root.minsize(1000, 640)

        self.image_predictor = KerasImagePredictionService()
        self.image_path: Path | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        # live-camera removed; no live capture state
        self.selected_model = tk.StringVar(value="MobileNetV2")
        self.status_text = tk.StringVar(value="Upload an image to begin.")
        self.gender_text = tk.StringVar(value="-")
        self.gender_confidence_text = tk.StringVar(value="-")
        self.age_group_text = tk.StringVar(value="-")
        self.confidence_text = tk.StringVar(value="-")
        self.model_text = tk.StringVar(value="-")
        self._age_label_display = {
            "child": "Child",
            "teen": "Teen",
            "young_adult": "Young adult",
            "adult": "Adult",
            "senior": "Senior",
        }

        self._configure_style()
        self._build_layout()

    def _configure_style(self) -> None:
        self.root.configure(background="#f3f6fb")

        style = ttk.Style()
        style.theme_use("clam")

        style.configure("App.TFrame", background="#f3f6fb")
        style.configure("Card.TFrame", background="#ffffff", relief="flat")
        style.configure("Section.TLabel", background="#ffffff", foreground="#16324a", font=("Segoe UI", 15, "bold"))
        style.configure("ResultsTitle.TLabel", background="#ffffff", foreground="#16324a", font=("Segoe UI", 18, "bold"))
        style.configure("Body.TLabel", background="#ffffff", foreground="#3f4b5a", font=("Segoe UI", 11))
        style.configure("Status.TLabel", background="#f3f6fb", foreground="#5b6574", font=("Segoe UI", 10))
        style.configure("ResultValue.TLabel", background="#ffffff", foreground="#0f172a", font=("Segoe UI", 15, "bold"))
        style.configure("ResultName.TLabel", background="#ffffff", foreground="#334155", font=("Segoe UI", 11, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(18, 11))
        style.map(
            "Accent.TButton",
            foreground=[("disabled", "#6b7280"), ("!disabled", "white")],
            background=[("disabled", "#d1d5db"), ("!disabled", "#1f5fbf"), ("active", "#184ea3")],
        )
        style.configure("Ghost.TRadiobutton", background="#ffffff", foreground="#334155", font=("Segoe UI", 10))

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, style="App.TFrame", padding=12)
        container.pack(fill="both", expand=True)

        body = ttk.Frame(container, style="App.TFrame")
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left_card = ttk.Frame(body, style="Card.TFrame", padding=16)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right_card = ttk.Frame(body, style="Card.TFrame", padding=16)
        right_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

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
            text="Upload an image for analysis.",
            style="Body.TLabel",
            wraplength=430,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(6, 12))

        # Controls rows (fixed size)
        self.upload_controls_frame = ttk.Frame(parent, style="Card.TFrame")
        self.upload_controls_frame.grid(row=3, column=0, sticky="w")
        ttk.Button(self.upload_controls_frame, text="Upload Image", style="Accent.TButton", command=self._upload_image).pack(side="left")
        ttk.Label(self.upload_controls_frame, text="PNG, JPG, JPEG, BMP, GIF, WEBP", style="Body.TLabel").pack(side="left", padx=12)

        # no live controls; upload-only UI

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
        self.predict_button = ttk.Button(action_row, text="Predict", style="Accent.TButton", command=self._predict)
        self.predict_button.grid(row=0, column=0, sticky="we")

        # Preview gets the flexible remaining space so it cannot push controls away
        preview_frame = ttk.Frame(parent, style="Card.TFrame")
        preview_frame.grid(row=6, column=0, sticky="nsew", pady=(12, 0))
        parent.rowconfigure(6, weight=1)
        preview_frame.configure(height=320)
        preview_frame.grid_propagate(False)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)

        ttk.Label(preview_frame, text="Preview", style="Section.TLabel").grid(row=0, column=0, sticky="w")

        preview_border = tk.Frame(preview_frame, background="#ffffff", highlightbackground="#dbe3ef", highlightthickness=1)
        preview_border.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        preview_border.configure(height=240)
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
        ttk.Label(parent, text="Results", style="ResultsTitle.TLabel").pack(anchor="w", pady=(0, 10), padx=(2, 0))

        result_card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        result_card.pack(fill="both", expand=True)
        result_card.columnconfigure(0, weight=1)

        # Friendly, non-technical result rows
        result_card.rowconfigure(0, weight=1)
        result_card.rowconfigure(1, weight=1)
        result_card.rowconfigure(2, weight=1)
        result_card.rowconfigure(3, weight=1)
        result_card.rowconfigure(4, weight=1)
        self._add_result_row(result_card, 0, "Predicted Gender", self.gender_text)
        self._add_result_row(result_card, 1, "Gender Confidence", self.gender_confidence_text)
        self._add_result_row(result_card, 2, "Predicted Age Range", self.age_group_text)
        self._add_result_row(result_card, 3, "Age Confidence", self.confidence_text)
        self._add_result_row(result_card, 4, "Model", self.model_text)

    def _add_result_row(self, parent: ttk.Frame, row_index: int, label: str, variable: tk.StringVar) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.grid(row=row_index, column=0, sticky="ew", pady=(5, 5))
        row.columnconfigure(0, weight=1)
        row.columnconfigure(1, weight=0)
        ttk.Label(row, text=label, style="ResultName.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            row,
            textvariable=variable,
            style="ResultValue.TLabel",
            width=12,
            anchor="e",
        ).grid(row=0, column=1, sticky="e", padx=(12, 0))

    def _upload_image(self) -> None:
        # Upload-only flow

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
        # keep layout stable: only disable the button while analyzing
        self.predict_button.config(state="disabled")
        self.status_text.set("Running prediction...")

        def worker():
            try:
                result = self.image_predictor.predict(self.image_path, self.selected_model.get())
                gender_prob = getattr(self.image_predictor, 'last_gender_prob', None)
                age_probs = getattr(self.image_predictor, 'last_age_probs', None)
                self.root.after(0, lambda: self._on_prediction_done(result, gender_prob, age_probs, None))
            except Exception as exc:
                self.root.after(0, lambda: self._on_prediction_done(None, None, None, exc))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _on_prediction_done(self, result, gender_prob, age_probs, error):
        self.predict_button.config(state="normal")

        if error is not None:
            messagebox.showerror("Model Error", str(error))
            self.status_text.set("Image prediction failed.")
            return

        self.gender_text.set(result.predicted_gender)
        if gender_prob is not None:
            female = gender_prob
            male = 1.0 - female
            gender_confidence = female if result.predicted_gender == "Female" else male
            self.gender_confidence_text.set(f"{gender_confidence * 100:.0f}%")
        else:
            self.gender_confidence_text.set("-")

        # Age group is returned like 'young_adult (20-39)'; convert label to nicer casing
        age_text = result.predicted_age_group
        for key, pretty in self._age_label_display.items():
            if age_text.startswith(key):
                age_text = age_text.replace(key, pretty, 1)
                break

        self.age_group_text.set(age_text)
        self.confidence_text.set(f"{result.confidence_score:.0f}%")
        self.model_text.set(result.selected_model)

        self.status_text.set(f"Predicted using {result.selected_model} for {self.image_path.name}.")

    def _refresh_mode_ui(self) -> None:
        # Upload-only UI: ensure upload controls visible
        try:
            self.upload_controls_frame.grid(row=3, column=0, sticky="w")
        except Exception:
            pass

    # Live camera analysis removed — upload-image only

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            # nothing special to stop in upload-only mode
            pass
