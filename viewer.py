# ---
# viewer.py — minimal cross‑platform live viewer (no extra deps)
# --------------------------------------------------------------
# Usage:
#   python viewer.py --dir out --interval 500 --fullscreen 0
# Keys:
#   q = quit, f = toggle fullscreen, space = pause/resume

import argparse, glob, os, time
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk


def newest_frame(folder: Path):
    files = list(folder.glob('frame_*.png'))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def fit_image_to(size, img: Image.Image):
    W, H = size
    w, h = img.size
    scale = min(W / max(1, w), H / max(1, h))
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.LANCZOS)


def run_viewer(folder: str, interval_ms: int, fullscreen: bool):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    root = tk.Tk()
    root.title('Autogenesis — Live Viewer')
    root.configure(bg='#000')
    root.attributes('-fullscreen', bool(fullscreen))

    paused = {'v': False}
    last_path = {'v': None}
    photo_ref = {'v': None}

    canvas = tk.Canvas(root, bg='#000', highlightthickness=0, bd=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    def toggle_fullscreen(event=None):
        fs = not bool(root.attributes('-fullscreen'))
        root.attributes('-fullscreen', fs)

    def toggle_pause(event=None):
        paused['v'] = not paused['v']

    def quit_app(event=None):
        root.destroy()

    root.bind('<f>', toggle_fullscreen)
    root.bind('<space>', toggle_pause)
    root.bind('<q>', quit_app)
    root.bind('<Escape>', quit_app)

    def refresh():
        if not paused['v']:
            p = newest_frame(folder)
            if p is not None and p != last_path['v']:
                try:
                    img = Image.open(p).convert('RGB')
                except Exception:
                    root.after(interval_ms, refresh)
                    return
                # fit to window
                W = root.winfo_width() or root.winfo_screenwidth()
                H = root.winfo_height() or root.winfo_screenheight()
                img2 = fit_image_to((W, H), img)
                photo = ImageTk.PhotoImage(img2)
                photo_ref['v'] = photo  # prevent GC
                canvas.delete('all')
                # center image
                x = (W - img2.size[0]) // 2
                y = (H - img2.size[1]) // 2
                canvas.create_image(x, y, anchor='nw', image=photo)
                last_path['v'] = p
        root.after(interval_ms, refresh)

    # Kick off
    root.after(interval_ms, refresh)
    root.mainloop()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='out', help='Folder to watch for frame_*.png')
    ap.add_argument('--interval', type=int, default=500, help='Refresh interval in ms')
    ap.add_argument('--fullscreen', type=int, default=0, help='1 to start fullscreen')
    args = ap.parse_args()
    run_viewer(args.dir, args.interval, bool(args.fullscreen))