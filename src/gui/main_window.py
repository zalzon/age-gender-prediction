from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk, ImageOps

from src.services.mock_predictor import MockPredictionService


class AgeGenderPredictionApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Age & Gender Prediction Dashboard")
        self.root.geometry("1180x720")
        self.root.minsize(1000, 640)

        self.predictor = MockPredictionService()
        self.image_path: Path | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None

        self.selected_model = tk.StringVar(value="MobileNetV2")
        self.status_text = tk.StringVar(value="Upload an image to begin.")
        self.gender_text = tk.StringVar(value="-")
        self.age_group_text = tk.StringVar(value="-")
        self.confidence_text = tk.StringVar(value="-")
        self.model_text = tk.StringVar(value="-")

        self.canvas: tk.Canvas | None = None
        self.scrollable_body: ttk.Frame | None = None
        self._body_window_id: int | None = None

        self._configure_style()
        self._build_layout()

    def _configure_style(self) -> None:
        self.root.configure(background="#eef2f7")

        style = ttk.Style()
        style.theme_use("clam")

        style.configure("App.TFrame", background="#eef2f7")
        style.configure("Hero.TFrame", background="#17324d")
        style.configure("Card.TFrame", background="#ffffff", relief="flat")
        style.configure("Section.TLabel", background="#ffffff", foreground="#17324d", font=("Segoe UI", 13, "bold"))
        style.configure("Body.TLabel", background="#ffffff", foreground="#334155", font=("Segoe UI", 10))
        style.configure("HeroTitle.TLabel", background="#17324d", foreground="#ffffff", font=("Segoe UI", 22, "bold"))
        style.configure("HeroSub.TLabel", background="#17324d", foreground="#dbeafe", font=("Segoe UI", 10))
        style.configure("Status.TLabel", background="#eef2f7", foreground="#475569", font=("Segoe UI", 10))
        style.configure("ResultValue.TLabel", background="#ffffff", foreground="#0f172a", font=("Segoe UI", 15, "bold"))
        style.configure("ResultName.TLabel", background="#ffffff", foreground="#64748b", font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(16, 10))
        style.map("Accent.TButton", foreground=[("!disabled", "white")], background=[("!disabled", "#2563eb"), ("active", "#1d4ed8")])
        style.configure("Ghost.TRadiobutton", background="#ffffff", foreground="#334155", font=("Segoe UI", 10))

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, style="App.TFrame", padding=18)
        container.pack(fill="both", expand=True)

        hero = ttk.Frame(container, style="Hero.TFrame", padding=24)
        hero.pack(fill="x")

        ttk.Label(
            hero,
            text="Comparative Analysis Dashboard",
            style="HeroTitle.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            hero,
            text="MobileNetV2 vs NASNetMobile for multi-task age group and gender prediction",
            style="HeroSub.TLabel",
        ).pack(anchor="w", pady=(6, 0))

        body = ttk.Frame(container, style="App.TFrame")
        body.pack(fill="both", expand=True, pady=(18, 0))

        self.canvas = tk.Canvas(body, background="#eef2f7", highlightthickness=0)
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollable_body = ttk.Frame(self.canvas, style="App.TFrame")
        self._body_window_id = self.canvas.create_window((0, 0), window=self.scrollable_body, anchor="nw")

        self.scrollable_body.columnconfigure(0, weight=5)
        self.scrollable_body.columnconfigure(1, weight=4)
        self.scrollable_body.rowconfigure(0, weight=1)

        self.scrollable_body.bind("<Configure>", self._on_body_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        left_card = ttk.Frame(self.scrollable_body, style="Card.TFrame", padding=20)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right_card = ttk.Frame(self.scrollable_body, style="Card.TFrame", padding=20)
        right_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        self._build_upload_section(left_card)
        self._build_results_section(right_card)

        footer = ttk.Label(
            container,
            textvariable=self.status_text,
            style="Status.TLabel",
            anchor="w",
        )
        footer.pack(fill="x", pady=(14, 0))

    def _on_body_configure(self, event: tk.Event) -> None:
        if self.canvas is None:
            return
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        if self.canvas is None or self._body_window_id is None:
            return
        self.canvas.itemconfigure(self._body_window_id, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        if self.canvas is None:
            return
        if event.delta:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _build_upload_section(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Image Input", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text="Choose a face image. The app currently uses mock prediction data only.",
            style="Body.TLabel",
            wraplength=470,
            justify="left",
        ).pack(anchor="w", pady=(6, 14))

        upload_row = ttk.Frame(parent, style="Card.TFrame")
        upload_row.pack(fill="x")

        ttk.Button(upload_row, text="Upload Image", style="Accent.TButton", command=self._upload_image).pack(side="left")
        ttk.Label(upload_row, text="Supported preview: common image formats", style="Body.TLabel").pack(side="left", padx=12)

        preview_frame = ttk.Frame(parent, style="Card.TFrame")
        preview_frame.pack(fill="both", expand=True, pady=(18, 18))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)

        ttk.Label(preview_frame, text="Preview", style="Section.TLabel").grid(row=0, column=0, sticky="w")

        self.preview_label = ttk.Label(
            preview_frame,
            text="No image selected",
            style="Body.TLabel",
            anchor="center",
            relief="solid",
            padding=14,
        )
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(12, 0))

        selector_frame = ttk.Frame(parent, style="Card.TFrame")
        selector_frame.pack(fill="x")

        ttk.Label(selector_frame, text="Model Selector", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            selector_frame,
            text="Choose the model you want to compare in the interface.",
            style="Body.TLabel",
        ).pack(anchor="w", pady=(4, 10))

        radio_row = ttk.Frame(selector_frame, style="Card.TFrame")
        radio_row.pack(anchor="w")
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

        ttk.Button(parent, text="Predict", style="Accent.TButton", command=self._predict).pack(anchor="e", pady=(20, 0))

    def _build_results_section(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Prediction Results", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text="Mock output updates immediately so the UI can be tested before ML integration.",
            style="Body.TLabel",
            wraplength=360,
            justify="left",
        ).pack(anchor="w", pady=(6, 16))

        result_card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        result_card.pack(fill="both", expand=True)

        self._add_result_row(result_card, "Predicted Gender", self.gender_text)
        self._add_result_row(result_card, "Predicted Age Group", self.age_group_text)
        self._add_result_row(result_card, "Confidence Score", self.confidence_text)
        self._add_result_row(result_card, "Selected Model", self.model_text)

        ttk.Separator(result_card).pack(fill="x", pady=16)

        note = ttk.Label(
            result_card,
            text="This front-end is intentionally isolated from TensorFlow so your teammate can connect the model layer later without changing the UI structure.",
            style="Body.TLabel",
            wraplength=340,
            justify="left",
        )
        note.pack(anchor="w")

    def _add_result_row(self, parent: ttk.Frame, label: str, variable: tk.StringVar) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=10)
        ttk.Label(row, text=label, style="ResultName.TLabel").pack(anchor="w")
        ttk.Label(row, textvariable=variable, style="ResultValue.TLabel").pack(anchor="w", pady=(2, 0))

    def _upload_image(self) -> None:
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
            preview = ImageOps.contain(image, (420, 320))
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

        result = self.predictor.predict(self.selected_model.get())
        self.gender_text.set(result.predicted_gender)
        self.age_group_text.set(result.predicted_age_group)
        self.confidence_text.set(f"{result.confidence_score:.2f}%")
        self.model_text.set(result.selected_model)
        self.status_text.set(f"Mock prediction generated for {self.image_path.name}.")

    def run(self) -> None:
        self.root.mainloop()
