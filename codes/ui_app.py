import ast
import importlib.util
import os
import threading
import time
import tkinter as tk
from tkinter import ttk
import cv2


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


class OpenCVCameraUI:
    """UI-only layer for drawing and click hit-testing in OpenCV frames."""

    def __init__(self, camera_view_w=800, camera_view_h=600, ui_panel_w=260):
        self.camera_view_w = camera_view_w
        self.camera_view_h = camera_view_h
        self.ui_panel_w = ui_panel_w
        self.ui_buttons = []
        self.title_font = 0.55
        self.text_font = 0.46
        self.button_font = 0.5
        self.status_font = 0.46
        self.buffer_top = 40
        self.buffer_bottom = 170
        self.lang_title_y = 190

    def _draw_button(self, frame, rect, label, active=False):
        x1, y1, x2, y2 = rect
        color = (70, 170, 70) if active else (50, 50, 50)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 230, 230), 1)
        baseline = y1 + int((y2 - y1) * 0.68)
        cv2.putText(
            frame,
            label,
            (x1 + 6, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.button_font,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _rebuild_buttons(self):
        panel_x = self.camera_view_w
        self.ui_buttons = []

        # Scale panel elements for small displays (e.g., 480x320) and larger screens.
        h = self.camera_view_h
        self.title_font = 0.46 if h <= 360 else 0.62
        self.text_font = 0.36 if h <= 360 else 0.5
        self.button_font = 0.38 if h <= 360 else 0.52
        self.status_font = 0.36 if h <= 360 else 0.48

        self.buffer_top = 30 if h <= 360 else 32
        self.buffer_bottom = min(max(95, int(h * 0.33)) if h <= 360 else max(120, int(h * 0.48)), h - 150)
        self.lang_title_y = self.buffer_bottom + 22

        inner_left = panel_x + 10
        inner_right = panel_x + self.ui_panel_w - 10
        gap = 8 if h <= 360 else 10
        btn_w = max(44, (inner_right - inner_left - gap) // 2)
        btn_h = 20 if h <= 360 else 34
        first_row_y = self.lang_title_y + 10

        langs = ["English", "Telugu", "Hindi", "French", "Spanish"]
        for i, lang in enumerate(langs):
            col = i % 2
            row = i // 2
            x1 = inner_left + col * (btn_w + gap)
            y1 = first_row_y + row * (btn_h + gap)
            x2 = x1 + btn_w
            y2 = y1 + btn_h
            self.ui_buttons.append(
                {"label": lang, "action": "lang", "value": lang, "rect": (x1, y1, x2, y2)}
            )

        action_btn_w = max(44, (inner_right - inner_left - gap) // 2)
        action_y1 = first_row_y + 3 * (btn_h + gap) + 8
        action_y2 = action_y1 + btn_h

        self.ui_buttons.append(
            {
                "label": "Reload",
                "action": "reload",
                "value": None,
                "rect": (inner_left, action_y1, inner_left + action_btn_w, action_y2),
            }
        )
        self.ui_buttons.append(
            {
                "label": "Run",
                "action": "run",
                "value": None,
                "rect": (
                    inner_left + action_btn_w + gap,
                    action_y1,
                    inner_left + action_btn_w + gap + action_btn_w,
                    action_y2,
                ),
            }
        )
        self.ui_buttons.append(
            {
                "label": "Clear",
                "action": "clear",
                "value": None,
                "rect": (inner_left, action_y2 + 8, inner_left + action_btn_w, action_y2 + 8 + btn_h),
            }
        )
        self.ui_buttons.append(
            {
                "label": "Exit",
                "action": "exit",
                "value": None,
                "rect": (
                    inner_left + action_btn_w + gap,
                    action_y2 + 8,
                    inner_left + action_btn_w + gap + action_btn_w,
                    action_y2 + 8 + btn_h,
                ),
            }
        )

        # If controls exceed the visible panel height, shift them upward.
        max_bottom = max(btn["rect"][3] for btn in self.ui_buttons)
        safe_bottom = h - 28
        if max_bottom > safe_bottom:
            shift = max_bottom - safe_bottom
            self.lang_title_y -= shift
            for btn in self.ui_buttons:
                x1, y1, x2, y2 = btn["rect"]
                btn["rect"] = (x1, y1 - shift, x2, y2 - shift)

    def draw(self, frame, stored_buffer_lines, selected_language, ui_status):
        if not self.ui_buttons:
            self._rebuild_buttons()

        h, w = frame.shape[:2]
        panel_x = self.camera_view_w

        cv2.rectangle(frame, (panel_x, 0), (w - 1, h - 1), (30, 30, 30), -1)
        cv2.line(frame, (panel_x, 0), (panel_x, h - 1), (90, 90, 90), 1)

        cv2.putText(frame, "Stored Buffer", (panel_x + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, self.title_font, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (panel_x + 10, self.buffer_top), (w - 10, self.buffer_bottom), (60, 60, 60), 1)

        lines = stored_buffer_lines[-8:] if stored_buffer_lines else ["(empty)"]
        max_chars = max(10, (self.ui_panel_w - 40) // 10)
        y = self.buffer_top + 18
        for ln in lines:
            cv2.putText(frame, ln[:max_chars], (panel_x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, self.text_font, (230, 230, 230), 1, cv2.LINE_AA)
            y += 16
            if y > self.buffer_bottom - 4:
                break

        cv2.putText(frame, "Languages", (panel_x + 10, self.lang_title_y), cv2.FONT_HERSHEY_SIMPLEX, self.title_font, (255, 255, 255), 1, cv2.LINE_AA)

        for btn in self.ui_buttons:
            active = btn["action"] == "lang" and btn["value"] == selected_language
            self._draw_button(frame, btn["rect"], btn["label"], active=active)

        cv2.putText(frame, f"Status: {ui_status}", (panel_x + 10, h - 24), cv2.FONT_HERSHEY_SIMPLEX, self.status_font, (200, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Lang: {selected_language}", (panel_x + 10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, self.status_font, (200, 220, 255), 1, cv2.LINE_AA)

    def handle_mouse(self, event, x, y, on_action):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        for btn in self.ui_buttons:
            x1, y1, x2, y2 = btn["rect"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                on_action(btn["action"], btn.get("value"))
                break
