import ast
import importlib.util
import os
import threading
import time
import tkinter as tk
from tkinter import ttk


BASE_DIR = os.path.dirname(__file__)


def translate_and_tts(text, language):
    print(f"Translating to {language} and generating TTS...")


def _has_function(module_path, function_name):
    """Check source for function existence before attempting import."""
    if not os.path.exists(module_path):
        return False

    try:
        with open(module_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=module_path)
        return any(
            isinstance(node, ast.FunctionDef) and node.name == function_name
            for node in tree.body
        )
    except Exception:
        return False


def _load_function(module_filename, function_name):
    """Safely try to load a function; return None if unavailable."""
    module_path = os.path.join(BASE_DIR, module_filename)
    if not _has_function(module_path, function_name):
        return None

    try:
        spec = importlib.util.spec_from_file_location(module_filename[:-3], module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, function_name, None)
    except Exception:
        return None


class GestureUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture-to-Speech UI")
        self.root.geometry("700x500")

        self.selected_language = "English"
        self.buffer_text = ""

        # Optional integrations if your modules expose these functions.
        # Expected signatures:
        # - collect_data() -> str
        # - process_data(buffer_text: str, language: str) -> any
        self.collect_data_fn = _load_function("collect.py", "collect_data")
        self.process_data_fn = _load_function("main.py", "process_data")

        self._build_ui()
        self._render_buffer()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Stored Buffer", font=("TkDefaultFont", 11, "bold")).pack(
            anchor="w"
        )

        self.buffer_box = tk.Text(main, height=12, wrap="word")
        self.buffer_box.pack(fill="both", expand=True, pady=(6, 8))

        controls = ttk.Frame(main)
        controls.pack(fill="x", pady=(0, 8))

        ttk.Button(controls, text="Refresh Buffer", command=self.refresh_buffer).pack(
            side="left"
        )

        self.lang_label = ttk.Label(
            controls, text=f"Selected language: {self.selected_language}"
        )
        self.lang_label.pack(side="right")

        lang_frame = ttk.LabelFrame(main, text="Language Selection", padding=10)
        lang_frame.pack(fill="x", pady=(0, 10))

        languages = ["English", "Telugu", "Hindi", "French", "Spanish"]
        for idx, lang in enumerate(languages):
            btn = ttk.Button(
                lang_frame,
                text=lang,
                command=lambda l=lang: self.select_language(l),
                width=14,
            )
            row = 0 if idx < 3 else 1
            col = idx if idx < 3 else idx - 3
            btn.grid(row=row, column=col, padx=6, pady=6, sticky="w")

        action_frame = ttk.Frame(main)
        action_frame.pack(fill="x")

        ttk.Button(action_frame, text="Collect", command=self.on_collect).pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(action_frame, text="Run / Process", command=self.on_process).pack(
            side="left"
        )

        self.status_var = tk.StringVar(value="Status: Idle")
        ttk.Label(main, textvariable=self.status_var).pack(anchor="w", pady=(10, 0))

    def _set_status(self, text):
        self.status_var.set(f"Status: {text}")

    def _render_buffer(self):
        self.buffer_box.delete("1.0", tk.END)
        self.buffer_box.insert("1.0", self.buffer_text)

    def refresh_buffer(self):
        # Pull current text from buffer variable into the visible box.
        self._render_buffer()

    def select_language(self, language):
        self.selected_language = language
        self.lang_label.config(text=f"Selected language: {language}")
        print(f"Selected language: {language}")

    def _placeholder_collect_data(self):
        # Placeholder behavior when collect_data() is not available.
        stamp = time.strftime("%H:%M:%S")
        new_text = f"[Collected at {stamp}]"
        print("collect.collect_data not found. Using placeholder collect.")
        return new_text

    def _placeholder_process_data(self, text, language):
        print("main.process_data not found. Using placeholder process.")
        print(f"Buffer: {text}")
        print(f"Language: {language}")

    def on_collect(self):
        def worker():
            self.root.after(0, lambda: self._set_status("Collecting"))

            try:
                if self.collect_data_fn is not None:
                    result = self.collect_data_fn()
                else:
                    result = self._placeholder_collect_data()

                if isinstance(result, str) and result.strip():
                    if self.buffer_text.strip():
                        self.buffer_text += "\n"
                    self.buffer_text += result.strip()

                self.root.after(0, self._render_buffer)
            finally:
                self.root.after(0, lambda: self._set_status("Idle"))

        threading.Thread(target=worker, daemon=True).start()

    def on_process(self):
        def worker():
            self.root.after(0, lambda: self._set_status("Processing"))

            try:
                # Capture current text from box in case user edited directly.
                current_text = self.buffer_box.get("1.0", tk.END).strip()
                self.buffer_text = current_text

                if self.process_data_fn is not None:
                    self.process_data_fn(current_text, self.selected_language)
                else:
                    self._placeholder_process_data(current_text, self.selected_language)

                translate_and_tts(current_text, self.selected_language)
            finally:
                self.root.after(0, lambda: self._set_status("Idle"))

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = GestureUI(root)
    root.mainloop()
