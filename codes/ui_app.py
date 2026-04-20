import ast
import importlib.util
import os
import threading
import time
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None


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

        languages = ["English", "Telugu", "Hindi", "French", "Spanish", "German"]
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
        self.colors = {
            "panel_bg": (18, 24, 28),
            "card_bg": (26, 33, 38),
            "card_border": (68, 84, 92),
            "header": (232, 238, 242),
            "text": (215, 224, 230),
            "muted": (150, 170, 182),
            "btn": (44, 56, 66),
            "btn_active": (46, 140, 84),
            "btn_border": (108, 124, 132),
        }
        self.logo_img = self._load_logo_image()
        self._logo_missing_logged = False

    def _load_logo_image(self):
        candidates = [
            os.path.join(BASE_DIR, "assets", "hash_logo.png"),
            os.path.join(BASE_DIR, "assets", "hash logo.png"),
            os.path.join(BASE_DIR, "assets", "hash_logo.jpg"),
            os.path.join(BASE_DIR, "assets", "hash_logo.jpeg"),
            os.path.join(BASE_DIR, "hash_logo.png"),
            os.path.join(BASE_DIR, "hash logo.png"),
            os.path.join(BASE_DIR, "hash_logo.jpg"),
            os.path.join(BASE_DIR, "logo.png"),
            os.path.join(BASE_DIR, "logo.jpg"),
        ]
        for path in candidates:
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    print(f"Loaded panel logo: {path}")
                    return img
        return None

    def _draw_logo(self, frame, panel_x, panel_w):
        logo_max = max(40, min(76, int(self.camera_view_h * 0.20)))
        box_pad = 8
        card_w = logo_max + box_pad * 2
        card_h = logo_max + box_pad * 2
        x1 = 8
        y1 = 8
        x2 = x1 + card_w
        y2 = y1 + card_h

        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors["card_bg"], -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors["card_border"], 1)

        if self.logo_img is None:
            if not self._logo_missing_logged:
                print(
                    "Logo not found. Place logo at codes/assets/hash_logo.png "
                    "(or codes/hash_logo.png)."
                )
                self._logo_missing_logged = True
            cv2.putText(
                frame,
                "#Hash",
                (x1 + 12, y1 + card_h // 2 + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                self.colors["header"],
                1,
                cv2.LINE_AA,
            )
            return

        src = self.logo_img
        h, w = src.shape[:2]
        if h <= 0 or w <= 0:
            return

        scale = min(logo_max / w, logo_max / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA)

        lx = x1 + (card_w - new_w) // 2
        ly = y1 + (card_h - new_h) // 2

        if resized.shape[2] == 4:
            alpha = resized[:, :, 3:4].astype(np.float32) / 255.0
            roi = frame[ly:ly + new_h, lx:lx + new_w].astype(np.float32)
            rgb = resized[:, :, :3].astype(np.float32)
            out = alpha * rgb + (1.0 - alpha) * roi
            frame[ly:ly + new_h, lx:lx + new_w] = out.astype(np.uint8)
        else:
            frame[ly:ly + new_h, lx:lx + new_w] = resized[:, :, :3]

    def _draw_button(self, frame, rect, label, active=False):
        x1, y1, x2, y2 = rect
        color = self.colors["btn_active"] if active else self.colors["btn"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors["btn_border"], 1)

        btn_w = x2 - x1
        btn_h = y2 - y1
        font_scale = self.button_font
        while font_scale > 0.30:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            if tw <= btn_w - 8 and th <= btn_h - 6:
                break
            font_scale -= 0.03

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        tx = x1 + max(4, (btn_w - tw) // 2)
        baseline = y1 + (btn_h + th) // 2 - 1
        cv2.putText(
            frame,
            label,
            (tx, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (245, 248, 250),
            1,
            cv2.LINE_AA,
        )

    def _rebuild_buttons(self):
        panel_x = self.camera_view_w
        self.ui_buttons = []

        # Scale panel elements for small displays (e.g., 480x320) and larger screens.
        h = self.camera_view_h
        self.title_font = 0.49 if h <= 360 else 0.64
        self.text_font = 0.40 if h <= 360 else 0.52
        self.button_font = 0.40 if h <= 360 else 0.54
        self.status_font = 0.39 if h <= 360 else 0.50

        self.buffer_top = 30 if h <= 360 else 32
        self.buffer_bottom = min(max(95, int(h * 0.34)) if h <= 360 else max(120, int(h * 0.48)), h - 150)
        self.lang_title_y = self.buffer_bottom + 22

        inner_left = panel_x + 10
        inner_right = panel_x + self.ui_panel_w - 10
        gap = 8 if h <= 360 else 10
        btn_w = max(44, (inner_right - inner_left - gap) // 2)
        btn_h = 21 if h <= 360 else 34
        first_row_y = self.lang_title_y + 10

        langs = ["English", "Telugu", "Hindi", "French", "Spanish", "German"]
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
                "label": "STT",
                "action": "stt",
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
        panel_w = w - panel_x

        cv2.rectangle(frame, (panel_x, 0), (w - 1, h - 1), self.colors["panel_bg"], -1)
        cv2.line(frame, (panel_x, 0), (panel_x, h - 1), self.colors["card_border"], 1)
        self._draw_logo(frame, panel_x, panel_w)

        cv2.putText(frame, "Stored Buffer", (panel_x + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, self.title_font, self.colors["header"], 1, cv2.LINE_AA)
        cv2.rectangle(frame, (panel_x + 10, self.buffer_top), (w - 10, self.buffer_bottom), self.colors["card_bg"], -1)
        cv2.rectangle(frame, (panel_x + 10, self.buffer_top), (w - 10, self.buffer_bottom), self.colors["card_border"], 1)

        lines = stored_buffer_lines[-8:] if stored_buffer_lines else ["(empty)"]
        max_chars = max(10, (self.ui_panel_w - 40) // 10)
        y = self.buffer_top + 18
        for ln in lines:
            cv2.putText(frame, ln[:max_chars], (panel_x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, self.text_font, self.colors["text"], 1, cv2.LINE_AA)
            y += 16
            if y > self.buffer_bottom - 4:
                break

        cv2.putText(frame, "Languages", (panel_x + 10, self.lang_title_y), cv2.FONT_HERSHEY_SIMPLEX, self.title_font, self.colors["header"], 1, cv2.LINE_AA)

        for btn in self.ui_buttons:
            active = btn["action"] == "lang" and btn["value"] == selected_language
            self._draw_button(frame, btn["rect"], btn["label"], active=active)

        cv2.putText(frame, f"Status: {ui_status}", (panel_x + 10, h - 24), cv2.FONT_HERSHEY_SIMPLEX, self.status_font, (174, 235, 190), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Lang: {selected_language}", (panel_x + 10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, self.status_font, self.colors["muted"], 1, cv2.LINE_AA)

    def handle_mouse(self, event, x, y, on_action):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        for btn in self.ui_buttons:
            x1, y1, x2, y2 = btn["rect"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                on_action(btn["action"], btn.get("value"))
                break


class OpenCVSTTUI:
    """Standalone STT screen UI shown after pressing the STT button."""

    def __init__(self, view_w=800, view_h=600):
        self.view_w = view_w
        self.view_h = view_h
        self.ui_buttons = []
        self.colors = {
            "bg_top": (22, 30, 37),
            "bg_bottom": (13, 17, 22),
            "card": (24, 34, 44),
            "card_border": (92, 114, 126),
            "text": (228, 236, 241),
            "muted": (164, 183, 194),
            "btn": (45, 61, 72),
            "btn_active": (46, 140, 84),
            "btn_border": (110, 131, 142),
        }
        self.text_card_rect = (0, 0, 0, 0)
        self.title_font = 0.88
        self.subtitle_font = 0.52
        self.text_font = 0.72
        self.status_font = 0.52
        self._font_cache = {}
        self._font_paths = {
            "English": [
                "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ],
            "French": [
                "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ],
            "Spanish": [
                "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ],
            "German": [
                "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ],
            "Hindi": [
                "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
                "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            ],
            "Telugu": [
                "/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf",
                "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            ],
        }
        if Image is None or ImageDraw is None or ImageFont is None:
            print("STT UI: Pillow unavailable, complex scripts may not render correctly.")
        else:
            print("STT UI: Pillow Unicode renderer enabled.")
        self._rebuild_buttons()

    def _draw_gradient_bg(self, frame):
        h, w = frame.shape[:2]
        top = np.array(self.colors["bg_top"], dtype=np.float32)
        bottom = np.array(self.colors["bg_bottom"], dtype=np.float32)
        for y in range(h):
            t = y / max(h - 1, 1)
            row = (1.0 - t) * top + t * bottom
            frame[y, :, :] = row.astype(np.uint8)

    def _draw_button(self, frame, rect, label, active=False):
        x1, y1, x2, y2 = rect
        color = self.colors["btn_active"] if active else self.colors["btn"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors["btn_border"], 1)

        font_scale = self.text_font
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        while tw > (x2 - x1 - 10) and font_scale > 0.38:
            font_scale -= 0.03
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        tx = x1 + max(4, (x2 - x1 - tw) // 2)
        ty = y1 + ((y2 - y1 + th) // 2)
        cv2.putText(
            frame,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (245, 248, 250),
            1,
            cv2.LINE_AA,
        )

    def _wrap_text(self, text, max_width, font_scale):
        words = text.split()
        if not words:
            return [""]

        lines = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            (cand_w, _), _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            if cand_w <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def _get_font(self, language, font_px):
        if ImageFont is None:
            return None

        key = (language, int(font_px))
        cached = self._font_cache.get(key)
        if cached is not None:
            return cached

        candidates = self._font_paths.get(language, []) + [
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, max(12, int(font_px)))
                    self._font_cache[key] = font
                    return font
                except Exception:
                    continue
        return None

    def _text_width(self, text, font_scale, language):
        if Image is not None and ImageDraw is not None and ImageFont is not None:
            font = self._get_font(language, 36 * font_scale)
            if font is not None:
                dummy = Image.new("RGB", (2, 2))
                draw = ImageDraw.Draw(dummy)
                bbox = draw.textbbox((0, 0), text, font=font)
                return max(0, bbox[2] - bbox[0])

        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        return w

    def _wrap_text_for_language(self, text, max_width, font_scale, language):
        words = text.split()
        if not words:
            return [""]

        lines = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if self._text_width(candidate, font_scale, language) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def _draw_multilingual_lines(self, frame, lines, x, y, line_height, language):
        if Image is None or ImageDraw is None or ImageFont is None:
            for idx, ln in enumerate(lines):
                cv2.putText(
                    frame,
                    ln,
                    (x, y + idx * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_font,
                    self.colors["text"],
                    1,
                    cv2.LINE_AA,
                )
            return

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = self._get_font(language, 36 * self.text_font)
        if font is None:
            for idx, ln in enumerate(lines):
                cv2.putText(
                    frame,
                    ln,
                    (x, y + idx * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_font,
                    self.colors["text"],
                    1,
                    cv2.LINE_AA,
                )
            return

        # Convert BGR -> RGB for PIL draw color.
        rgb = (self.colors["text"][2], self.colors["text"][1], self.colors["text"][0])
        for idx, ln in enumerate(lines):
            draw.text((x, y + idx * line_height), ln, font=font, fill=rgb)

        frame[:, :, :] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _rebuild_buttons(self):
        self.ui_buttons = []
        w = self.view_w
        h = self.view_h

        is_small = h <= 500 or w <= 900
        self.title_font = 0.68 if is_small else 0.90
        self.subtitle_font = 0.42 if is_small else 0.56
        self.text_font = 0.64 if is_small else 0.82
        self.status_font = 0.45 if is_small else 0.52

        languages = ["English", "Telugu", "Hindi", "French", "Spanish", "German"]
        gap = 5 if is_small else 8
        margin_x = 14 if is_small else 22
        usable_w = max(1, w - 2 * margin_x - 5 * gap)
        btn_w = max(44, usable_w // 6)
        btn_h = 24 if is_small else 32
        start_x = max(margin_x, (w - (6 * btn_w + 5 * gap)) // 2)
        start_y = 54 if is_small else 70

        for idx, lang in enumerate(languages):
            x1 = start_x + idx * (btn_w + gap)
            y1 = start_y
            x2 = x1 + btn_w
            y2 = y1 + btn_h
            self.ui_buttons.append(
                {"label": lang, "action": "lang", "value": lang, "rect": (x1, y1, x2, y2)}
            )

        action_gap = 10
        action_h = 34 if is_small else 40
        back_w = 104 if is_small else 128
        mic_w = 130 if is_small else 160
        y2 = h - 20
        y1 = y2 - action_h

        x2 = w - 22
        x1 = x2 - back_w
        mic_x2 = x1 - action_gap
        mic_x1 = mic_x2 - mic_w
        if mic_x1 < margin_x:
            mic_x1 = margin_x
            mic_x2 = min(x1 - action_gap, mic_x1 + mic_w)

        self.ui_buttons.append(
            {"label": "Mic: Start", "action": "toggle_mic", "value": None, "rect": (mic_x1, y1, mic_x2, y2)}
        )
        self.ui_buttons.append(
            {"label": "Back", "action": "back_camera", "value": None, "rect": (x1, y1, x2, y2)}
        )

        card_x1 = margin_x
        card_x2 = w - margin_x
        card_y1 = start_y + btn_h + 18
        card_y2 = y1 - 12
        if card_y2 - card_y1 < 120:
            card_y1 = max(start_y + btn_h + 8, card_y2 - 120)
        self.text_card_rect = (card_x1, card_y1, card_x2, card_y2)

    def draw(self, frame, selected_language, stt_text, ui_status, mic_listening=False, mic_label="Mic: Auto", text_language="English"):
        h, w = frame.shape[:2]
        if h != self.view_h or w != self.view_w:
            self.view_h = h
            self.view_w = w
            self._rebuild_buttons()

        is_small = h <= 500 or w <= 900

        self._draw_gradient_bg(frame)

        cv2.putText(
            frame,
            "Speech to Text (STT)",
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.title_font,
            self.colors["text"],
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Select language and speak into the microphone",
            (24, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.subtitle_font,
            self.colors["muted"],
            1,
            cv2.LINE_AA,
        )

        for btn in self.ui_buttons:
            label = btn["label"]
            if btn["action"] == "toggle_mic":
                label = "Mic: Stop" if mic_listening else "Mic: Start"
            active = (btn["action"] == "lang" and btn["value"] == selected_language) or (
                btn["action"] == "toggle_mic" and mic_listening
            )
            self._draw_button(frame, btn["rect"], label, active=active)

        card_x1, card_y1, card_x2, card_y2 = self.text_card_rect
        cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), self.colors["card"], -1)
        cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), self.colors["card_border"], 1)

        cv2.putText(
            frame,
            "Converted Text",
            (card_x1 + 10, card_y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56 if is_small else 0.66,
            self.colors["text"],
            1,
            cv2.LINE_AA,
        )

        visible_text = stt_text.strip() if stt_text.strip() else "Waiting for speech input..."
        text_x = card_x1 + 10
        max_w = max(120, card_x2 - card_x1 - 20)
        wrapped = self._wrap_text_for_language(visible_text, max_w, self.text_font, text_language)
        line_height = 28 if is_small else 34
        total_height = len(wrapped) * line_height
        start_y = max(card_y1 + 60, card_y1 + ((card_y2 - card_y1) - total_height) // 2 + 18)
        text_y = start_y
        max_lines = max(1, (card_y2 - start_y - 10) // line_height)
        self._draw_multilingual_lines(frame, wrapped[:max_lines], text_x, text_y, line_height, text_language)

        cv2.putText(
            frame,
            f"Status: {ui_status}",
            (24, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.status_font,
            (174, 235, 190),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            mic_label,
            (24, h - 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.status_font,
            self.colors["muted"],
            1,
            cv2.LINE_AA,
        )

    def handle_mouse(self, event, x, y, on_action):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        for btn in self.ui_buttons:
            x1, y1, x2, y2 = btn["rect"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                on_action(btn["action"], btn.get("value"))
                break
