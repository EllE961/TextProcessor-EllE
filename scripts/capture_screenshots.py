"""Run app.py unchanged, feed it one paragraph in each mode, and save both window captures as docs/assets/hero.png.

Windows only. Run from the repo root after `pip install -r requirements.txt pillow`:

    python scripts/capture_screenshots.py

The script replaces Tk's mainloop with one that types the paragraph into the input box, presses Generate
through the button's invoke() (no real mouse or keyboard input), waits for the worker thread to finish,
then grabs the window. It prints the generated text and the time each mode took.
"""
import ctypes
import ctypes.wintypes
import os
import random
import sys
import time
import tkinter as tk

ctypes.windll.shcore.SetProcessDpiAwareness(2)

import torch
from PIL import Image, ImageGrab

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")
OUT = os.path.join(ROOT, "docs", "assets")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
random.seed(0)

TEXT = ("Honeybees keep their hive at about 35 degrees Celsius while the brood is developing. On cold days the "
        "workers cluster together and shiver their flight muscles to make heat, and on hot days they spread water "
        "across the comb and fan their wings so that evaporation carries the heat away. A colony that cannot hold "
        "this temperature raises weaker bees, so the whole hive treats it as a shared job rather than leaving it "
        "to a few individuals.")
MODES = ["Paraphrase", "Summarize"]
app = {"__name__": "__main__", "__file__": APP}
original_mainloop = tk.Tk.mainloop


def grab(win):
    """Capture the visible window frame (title bar included) in physical pixels."""
    win.update_idletasks()
    win.update()
    hwnd = int(win.wm_frame(), 16)
    rect = ctypes.wintypes.RECT()
    ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect))
    return ImageGrab.grab(bbox=(rect.left, rect.top, rect.right, rect.bottom), all_screens=True)


def scripted_mainloop(win, n=0):
    scale = win.winfo_fpixels("1i") / 96.0  # app.py asks for 900x800 at 96 dpi
    win.geometry(f"{round(900 * scale)}x{round(800 * scale)}+40+20")
    win.attributes("-topmost", True)
    state = {"i": 0, "t0": 0.0, "shots": []}

    def start():
        app["operation_var"].set(MODES[state["i"]])
        app["text_entry"].delete("1.0", tk.END)
        app["text_entry"].insert("1.0", TEXT)
        app["output_text"].delete("1.0", tk.END)
        state["t0"] = time.time()
        app["generate_button"].invoke()
        win.after(400, poll)

    def poll():
        busy = str(app["generate_button"]["state"]) == "disabled"
        out = app["output_text"].get("1.0", "end-1c")
        if busy or not out.strip():
            win.after(300, poll)
            return
        mode = MODES[state["i"]]
        print(f"\n{mode}: {time.time() - state['t0']:.1f} s\n{out}", flush=True)
        win.after(600, finish)

    def finish():
        state["shots"].append(grab(win))
        state["i"] += 1
        if state["i"] < len(MODES):
            win.after(300, start)
            return
        a, b = state["shots"]
        gap = 24
        hero = Image.new("RGB", (a.width + b.width + gap, max(a.height, b.height)), (13, 12, 11))
        hero.paste(a, (0, 0))
        hero.paste(b, (a.width + gap, 0))
        hero.save(os.path.join(OUT, "hero.png"))
        print("saved docs/assets/hero.png", hero.size, flush=True)
        win.after(300, win.destroy)

    win.after(1500, start)
    original_mainloop(win, n)


tk.Tk.mainloop = scripted_mainloop
os.chdir(ROOT)
exec(compile(open(APP, encoding="utf-8").read(), APP, "exec"), app)
