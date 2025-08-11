"""
Autogenesis_1.0 — MVP autonomous visual generator (revised for long runs)
-----------------------------------------------------------
Adds scoring squash/normalisation, durability check using SciPy Gaussian blur,
novelty archive persistence, and optional time-window durability.
"""

from __future__ import annotations
import argparse, os, time, math, random, zlib, json, hashlib, pickle
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
try:
    import torch
    TORCH = True
    CUDA = torch.cuda.is_available()
except Exception:
    TORCH = False
    CUDA = False

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

@dataclass
class LedgerEntry:
    t: float
    genome_hash: str
    genome: Dict[str, Any]
    score: Dict[str, float]
    sensor_summary: Dict[str, float]

def sensor_packet() -> Dict[str, float]:
    t = time.time()
    clk = t - int(t)
    try:
        ur = int.from_bytes(os.urandom(8), 'little') / (2**64 - 1)
    except Exception:
        ur = random.random()
    pc = (time.perf_counter_ns() % 1_000_000) / 1_000_000.0
    return {"clk": clk, "ur": ur, "pc": pc}

@dataclass
class Genome:
    op: str
    params: Dict[str, float]
    seed: int

OPS = ["flow", "rd", "spectral"]

def make_genome(rng: random.Random) -> Genome:
    op = rng.choice(OPS)
    if op == "flow":
        params = {"scale": 0.003 + rng.random()*0.01,
                  "steps": rng.randint(200, 1200),
                  "jitter": 0.2 + rng.random()*0.8}
    elif op == "rd":
        params = {"f": 0.01 + rng.random()*0.04,
                  "k": 0.045 + rng.random()*0.02,
                  "dt": 1.0,
                  "diff_u": 0.16,
                  "diff_v": 0.08,
                  "iters": rng.randint(200, 1000)}
    else:
        params = {"bands": rng.randint(3, 7),
                  "alpha": 0.6 + rng.random()*0.4,
                  "iters": rng.randint(2, 6)}
    return Genome(op=op, params=params, seed=rng.randrange(2**32))

def perlin_like_noise(h, w, scale, rng):
    gy, gx = np.mgrid[0:h, 0:w]
    angles = rng.random()*2*np.pi + np.zeros((h, w))
    g = np.dstack((np.cos(angles), np.sin(angles)))
    y = gy * scale
    x = gx * scale
    field = np.sin(2*np.pi*(x*np.cos(0.7)+y*np.sin(1.3))) + np.sin(2*np.pi*(x*0.7 - y*1.1))
    field = (field - field.min())/(np.ptp(field)+1e-8)
    return field.astype(np.float32), g.astype(np.float32)

# Assume render_flow, render_rd, render_spectral defined as before
def render(genome: Genome, h: int, w: int) -> np.ndarray:
    if genome.op == "flow":
        return render_flow(h,w,genome.params,genome.seed)
    elif genome.op == "rd":
        return render_rd(h,w,genome.params,genome.seed)
    else:
        return render_spectral(h,w,genome.params,genome.seed)

try:
    import imagehash
except Exception:
    imagehash = None

def compressibility_score(img: np.ndarray) -> float:
    raw = img.tobytes()
    comp = zlib.compress(raw, level=6)
    ratio = len(raw) / (len(comp)+1)
    return float(ratio)

_prev_img = None
_prev_prev_img = None

def surprise_score(img: np.ndarray) -> float:
    global _prev_img, _prev_prev_img
    if _prev_img is None:
        _prev_prev_img = img
        _prev_img = img
        return 0.0
    pred = _prev_img.astype(np.float32) + (_prev_img.astype(np.float32) - _prev_prev_img.astype(np.float32))
    pred = np.clip(pred,0,255)
    err = np.mean((img.astype(np.float32) - pred)**2)
    return float(err/ (255.0**2))

def homeostasis_score(img: np.ndarray) -> float:
    gray = np.mean(img, axis=-1)
    F = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(F)
    h, w = mag.shape
    Y, X = np.mgrid[-h//2:h//2, -w//2:w//2]
    R = np.sqrt(X**2 + Y**2)
    mid = (R > min(h,w)*0.05) & (R < min(h,w)*0.25)
    energy_mid = float(np.mean(mag[mid]))
    energy_total = float(np.mean(mag)) + 1e-8
    return energy_mid / energy_total

def novelty_score(img: np.ndarray, archive: Dict[str, float]) -> float:
    if imagehash is None:
        return random.random()*0.01
    im = Image.fromarray(img)
    ph = imagehash.phash(im)
    key = str(ph)
    if key in archive:
        return 0.0
    return 1.0

def joint_score(img: np.ndarray, sensors: Dict[str,float], archive: Dict[str,float]) -> Dict[str, float]:
    sc = {
        "compress": compressibility_score(img),
        "surprise_raw": surprise_score(img),
        "homeo": homeostasis_score(img),
    }
    sc["novelty"] = novelty_score(img, archive)
    mu, sigma = 0.02, 0.02
    surprise_s = float(np.exp(-((sc["surprise_raw"] - mu)**2) / (2*sigma**2)))
    compress_s = float(np.tanh(sc["compress"] / 100.0))
    homeo_s = float(np.clip(sc["homeo"] / 2.5, 0.0, 1.0))
    sc.update({"compress_s": compress_s, "homeo_s": homeo_s, "surprise": surprise_s})
    sc["total"] = (
        0.35*homeo_s +
        0.30*compress_s +
        0.25*surprise_s +
        0.10*sc["novelty"]
    )
    return sc

def genome_hash(g: Genome) -> str:
    m = hashlib.sha256()
    m.update((g.op+json.dumps(g.params, sort_keys=True)+str(g.seed)).encode())
    return m.hexdigest()[:16]

def tournament(h, w, base_rng: random.Random, k=6) -> Tuple[Genome, np.ndarray, Dict[str,float]]:
    best = None
    best_img = None
    best_score = None
    sensors = sensor_packet()
    for _ in range(k):
        g = make_genome(base_rng)
        img = render(g, h, w)
        sc = joint_score(img, sensors, archive)
        if best is None or sc["total"] > best_score["total"]:
            best, best_img, best_score = g, img, sc
    return best, best_img, best_score

def save_image(img: np.ndarray, path: str):
    Image.fromarray(img).save(path)

def durability_score(img: np.ndarray) -> float:
    img_blur = gaussian_filter(img.astype(np.float32), sigma=(0.5,0.5,0))
    img_blur = np.clip(img_blur,0,255).astype(np.uint8)
    return joint_score(img_blur, sensor_packet(), archive)["total"]

def time_window_durability(cand_img: np.ndarray, window_s=30, step_s=5) -> bool:
    start_score = joint_score(cand_img, sensor_packet(), archive)["total"]
    for _ in range(window_s // step_s):
        time.sleep(step_s)
        perturbed = gaussian_filter(cand_img.astype(np.float32), sigma=(0.5,0.5,0))
        perturbed = np.clip(perturbed, 0, 255).astype(np.uint8)
        score = joint_score(perturbed, sensor_packet(), archive)["total"]
        if score <= start_score * 0.98:
            return False
    return True

def save_archive(path="archive.pkl"):
    with open(path, "wb") as f:
        pickle.dump(archive, f)

def load_archive(path="archive.pkl"):
    global archive
    if os.path.exists(path):
        with open(path, "rb") as f:
            archive.update(pickle.load(f))

archive: Dict[str, float] = {}

def run(out: str, minutes: float, width: int, height: int, epsilon: float):
    global _prev_img, _prev_prev_img
    ensure_dir(out)
    load_archive()
    rng = random.Random()
    current = make_genome(rng)
    current_img = render(current, height, width)
    current_score = joint_score(current_img, sensor_packet(), archive)
    _prev_prev_img = _prev_img = current_img
    start = time.time()
    while time.time() - start < minutes*60:
        cand, cand_img, cand_score = tournament(height, width, rng)
        if cand_score["total"] > current_score["total"] * (1 + epsilon):
            if not time_window_durability(cand_img):
                time.sleep(0.2)
                continue
            _prev_prev_img, _prev_img = _prev_img, cand_img
            if imagehash is not None:
                key = str(imagehash.phash(Image.fromarray(cand_img)))
                archive[key] = cand_score["total"]
            current, current_img, current_score = cand, cand_img, cand_score
            tag = f"{int(time.time())}_{genome_hash(current)}"
            save_image(current_img, os.path.join(out, f"frame_{tag}.png"))
            led = LedgerEntry(
                t=time.time(),
                genome_hash=genome_hash(current),
                genome={"op": current.op, "params": current.params, "seed": current.seed},
                score=current_score,
                sensor_summary=sensor_packet()
            )
            with open(os.path.join(out, f"ledger_{tag}.json"), 'w') as f:
                json.dump(asdict(led), f)
            print("[PROMOTION]", tag, json.dumps(current_score))
        else:
            time.sleep(0.2)
    save_archive()
    tag = f"final_{int(time.time())}_{genome_hash(current)}"
    save_image(current_img, os.path.join(out, f"frame_{tag}.png"))
    print("[END] final score:", json.dumps(current_score))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='out')
    p.add_argument('--minutes', type=float, default=1.0)
    p.add_argument('--width', type=int, default=1080)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--epsilon', type=float, default=0.08, help='min fractional improvement to replace')
    args = p.parse_args()
    run(args.out, args.minutes, args.width, args.height, args.epsilon)
