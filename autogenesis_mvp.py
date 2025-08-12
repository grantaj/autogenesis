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

# --- Tunable configuration (CLI-exposed) ---
@dataclass
class Config:
    w_homeo: float
    w_compress: float
    w_surprise: float
    w_novelty: float
    surprise_mu: float
    surprise_sigma: float
    dur_window: int
    dur_step: int
    epsilon: float
    drift_tau_min: float  # 0 disables drift
    min_epsilon: float

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

# --------------------------- Renderers --------------------------------------

def render_flow(h: int, w: int, params: Dict[str, float], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise, _grad = perlin_like_noise(h, w, float(params.get("scale", 0.006)), rng)

    pos = np.dstack(np.meshgrid(
        np.linspace(0, 1, w, endpoint=False),
        np.linspace(0, 1, h, endpoint=False)
    )).astype(np.float32)

    img = np.zeros((h, w, 3), dtype=np.float32)
    steps = int(params.get("steps", 600))
    jitter = float(params.get("jitter", 0.5))

    for i in range(steps):
        vx = np.cos(noise * 2 * np.pi)
        vy = np.sin(noise * 2 * np.pi)
        pos[..., 0] = (pos[..., 0] + jitter * vx / w) % 1.0
        pos[..., 1] = (pos[..., 1] + jitter * vy / h) % 1.0

        phase = (i / max(1, steps)) * 2 * np.pi
        c = np.dstack([
            0.5 + 0.5 * np.sin(phase + 6.0 * pos[..., 0]),
            0.5 + 0.5 * np.sin(phase + 5.0 * pos[..., 1]),
            0.5 + 0.5 * np.sin(phase + 7.0 * (pos[..., 0] + pos[..., 1]))
        ])
        img = 0.99 * img + 0.01 * c

    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def render_rd(h: int, w: int, params: Dict[str, float], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    U = np.ones((h, w), dtype=np.float32)
    V = np.zeros((h, w), dtype=np.float32)

    # random seeding
    idx_y = rng.integers(0, h, size=50)
    idx_x = rng.integers(0, w, size=50)
    V[idx_y, idx_x] = 1.0

    def lap(X: np.ndarray) -> np.ndarray:
        return (-4 * X + np.roll(X, 1, 0) + np.roll(X, -1, 0)
                + np.roll(X, 1, 1) + np.roll(X, -1, 1))

    f = float(params.get("f", 0.03))
    k = float(params.get("k", 0.055))
    dt = float(params.get("dt", 1.0))
    Du = float(params.get("diff_u", 0.16))
    Dv = float(params.get("diff_v", 0.08))
    iters = int(params.get("iters", 500))

    for _ in range(iters):
        UVV = U * V * V
        U += (Du * lap(U) - UVV + f * (1 - U)) * dt
        V += (Dv * lap(V) + UVV - (f + k) * V) * dt
        U = np.clip(U, 0, 1)
        V = np.clip(V, 0, 1)

    img = np.stack([U, V, 1 - U], axis=-1)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)


def render_spectral(h: int, w: int, params: Dict[str, float], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bands = int(params.get("bands", 5))
    alpha = float(params.get("alpha", 0.8))
    _iters = int(params.get("iters", 3))

    img = np.zeros((h, w), dtype=np.float32)
    Y, X = np.mgrid[0:h, 0:w]

    for _ in range(bands):
        fx = int(rng.integers(1, max(2, min(64, w // 4))))
        fy = int(rng.integers(1, max(2, min(64, h // 4))))
        phase = float(rng.random() * 2 * np.pi)
        img += np.sin(2 * np.pi * (X * fx / w + Y * fy / h) + phase)

    img = img / max(1, bands)
    img = np.sign(img) * np.power(np.abs(img), alpha).astype(np.float32)

    rgb = np.stack([
        0.5 + 0.5 * np.sin(2.1 * img),
        0.5 + 0.5 * np.sin(1.7 * img + 1.3),
        0.5 + 0.5 * np.sin(1.3 * img + 2.1)
    ], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return (rgb * 255).astype(np.uint8)

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

def joint_score(img: np.ndarray, sensors: Dict[str,float], archive: Dict[str,float], cfg: Config) -> Dict[str, float]:
    sc = {
        "compress": compressibility_score(img),
        "surprise_raw": surprise_score(img),
        "homeo": homeostasis_score(img),
    }
    sc["novelty"] = novelty_score(img, archive)

    # shaped surprise around target
    surprise_s = float(np.exp(-((sc["surprise_raw"] - cfg.surprise_mu)**2) / (2*cfg.surprise_sigma**2)))
    # squash/normalize
    compress_s = float(np.tanh(sc["compress"] / 100.0))
    homeo_s    = float(np.clip(sc["homeo"] / 2.5, 0.0, 1.0))

    sc.update({"compress_s": compress_s, "homeo_s": homeo_s, "surprise": surprise_s})
    sc["total"] = (
        cfg.w_homeo*homeo_s +
        cfg.w_compress*compress_s +
        cfg.w_surprise*surprise_s +
        cfg.w_novelty*sc["novelty"]
    )
    return sc


def genome_hash(g: Genome) -> str:
    m = hashlib.sha256()
    m.update((g.op+json.dumps(g.params, sort_keys=True)+str(g.seed)).encode())
    return m.hexdigest()[:16]

def tournament(h, w, base_rng: random.Random, k=6, cfg: Config=None) -> Tuple[Genome, np.ndarray, Dict[str,float]]:
    best = None
    best_img = None
    best_score = None
    sensors = sensor_packet()
    for _ in range(k):
        g = make_genome(base_rng)
        img = render(g, h, w)
        sc = joint_score(img, sensors, archive, cfg)
        if best is None or sc["total"] > best_score["total"]:
            best, best_img, best_score = g, img, sc
    return best, best_img, best_score


def save_image(img: np.ndarray, path: str):
    Image.fromarray(img).save(path)

def durability_score(img: np.ndarray) -> float:
    img_blur = gaussian_filter(img.astype(np.float32), sigma=(0.5,0.5,0))
    img_blur = np.clip(img_blur,0,255).astype(np.uint8)
    return joint_score(img_blur, sensor_packet(), archive, cfg)["total"]

def time_window_durability(cand_img: np.ndarray, cfg: Config) -> bool:
    start_score = joint_score(cand_img, sensor_packet(), archive, cfg)["total"]
    steps = max(1, int(cfg.dur_window // max(1, cfg.dur_step)))
    for _ in range(steps):
        time.sleep(cfg.dur_step)
        pert = gaussian_filter(cand_img.astype(np.float32), sigma=(0.5,0.5,0))
        pert = np.clip(pert, 0, 255).astype(np.uint8)
        s = joint_score(pert, sensor_packet(), archive, cfg)["total"]
        if s <= start_score * 0.98:
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

def run(out: str, minutes: float, width: int, height: int, cfg: Config):
    global _prev_img, _prev_prev_img
    ensure_dir(out)
    load_archive()
    start = time.time()
    last_promo_t = start
    rng = random.Random()
    current = make_genome(rng)
    current_img = render(current, height, width)
    current_score = joint_score(current_img, sensor_packet(), archive, cfg)
    _prev_prev_img = _prev_img = current_img
    start = time.time()
    while time.time() - start < minutes*60:
        cand, cand_img, cand_score = tournament(height, width, rng, cfg=cfg)
        age_min = (time.time() - last_promo_t) / 60.0
        eps_eff = max(cfg.min_epsilon, cfg.epsilon * math.exp(-age_min / cfg.drift_tau_min)) if cfg.drift_tau_min > 0 else cfg.epsilon

        if cand_score["total"] > current_score["total"] * (1 + eps_eff):
            last_promo_t = time.time()
            if not time_window_durability(cand_img, cfg):
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
                sensor_summary=sensor_packet(),
                config=asdict(cfg)
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
    # scoring weights & surprise shape
    p.add_argument('--w_homeo', type=float, default=0.35)
    p.add_argument('--w_compress', type=float, default=0.30)
    p.add_argument('--w_surprise', type=float, default=0.25)
    p.add_argument('--w_novelty', type=float, default=0.10)
    p.add_argument('--surprise_mu', type=float, default=0.02)
    p.add_argument('--surprise_sigma', type=float, default=0.02)
    # durability and promotion policy
    p.add_argument('--dur_window', type=int, default=12, help='durability window (seconds)')
    p.add_argument('--dur_step', type=int, default=4, help='durability recheck step (seconds)')
    p.add_argument('--epsilon', type=float, default=0.06, help='base fractional improvement to replace')
    p.add_argument('--drift_tau', type=float, default=45.0, help='minutes; 0 disables drift')
    p.add_argument('--min_epsilon', type=float, default=0.02)
    args = p.parse_args()

    cfg = Config(
        w_homeo=args.w_homeo,
        w_compress=args.w_compress,
        w_surprise=args.w_surprise,
        w_novelty=args.w_novelty,
        surprise_mu=args.surprise_mu,
        surprise_sigma=args.surprise_sigma,
        dur_window=args.dur_window,
        dur_step=args.dur_step,
        epsilon=args.epsilon,
        drift_tau_min=args.drift_tau,
        min_epsilon=args.min_epsilon,
    )
    run(args.out, args.minutes, args.width, args.height, cfg)

