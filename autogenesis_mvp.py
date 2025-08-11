"""
Autogenesis_1.0 — MVP autonomous visual generator
--------------------------------------------------
A self-driven image generator that decides when to change state based on
machine-native criteria (compressibility vs. surprise, novelty, homeostasis).

This is a single-file prototype designed to run on CPU or CUDA (if torch+GPU
are available). No human-tuned aesthetics. Parameters are chosen by the system.

Run:
  python autogenesis_mvp.py --out out --minutes 5

It will render to an on-disk framebuffer (PNG files) and print minimal telemetry.
A separate viewer (e.g., `feh --reload 0.2 out`) can be used to live display.

Dependencies (install with pip):
  numpy, pillow, imagehash, torch (optional), tqdm
"""

from __future__ import annotations
import argparse, os, time, math, random, zlib, json, hashlib
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

# --------------------------- Utility & IO -----------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

@dataclass
class LedgerEntry:
    t: float
    genome_hash: str
    score: Dict[str, float]
    sensor_summary: Dict[str, float]
    genome: Dict[str, Any]   # <— add this line


# --------------------------- Sensors (machine-facing) -----------------------

def sensor_packet() -> Dict[str, float]:
    """Non-anthropocentric jitter sources available on most systems.
    In a full build, replace with real sensors (EM noise, network jitter, etc.)."""
    t = time.time()
    # clock jitter / monotonic drift
    clk = t - int(t)
    # /dev/urandom entropy sample (fallback to PRNG if unavailable)
    try:
        ur = int.from_bytes(os.urandom(8), 'little') / (2**64 - 1)
    except Exception:
        ur = random.random()
    # process noise: cpu time slice variations via time.perf_counter_ns
    pc = (time.perf_counter_ns() % 1_000_000) / 1_000_000.0
    return {"clk": clk, "ur": ur, "pc": pc}

# --------------------------- Operator library -------------------------------

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
    elif op == "rd":  # Gray-Scott
        params = {"f": 0.01 + rng.random()*0.04,
                  "k": 0.045 + rng.random()*0.02,
                  "dt": 1.0,
                  "diff_u": 0.16,
                  "diff_v": 0.08,
                  "iters": rng.randint(200, 1000)}
    else:  # spectral tiling
        params = {"bands": rng.randint(3, 7),
                  "alpha": 0.6 + rng.random()*0.4,
                  "iters": rng.randint(2, 6)}
    return Genome(op=op, params=params, seed=rng.randrange(2**32))

# --------------------------- Renderers --------------------------------------

def perlin_like_noise(h, w, scale, rng):
    # Cheap gradient noise (not true Perlin, but adequate and differentiable-ish)
    gy, gx = np.mgrid[0:h, 0:w]
    angles = rng.random()*2*np.pi + np.zeros((h, w))
    g = np.dstack((np.cos(angles), np.sin(angles)))
    y = gy * scale
    x = gx * scale
    # Simple sinusoidal field composition
    field = np.sin(2*np.pi*(x*np.cos(0.7)+y*np.sin(1.3))) + np.sin(2*np.pi*(x*0.7 - y*1.1))
    field = (field - field.min()) / (np.ptp(field) + 1e-8) 
    return field.astype(np.float32), g.astype(np.float32)


def render_flow(h, w, params, seed) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise, grad = perlin_like_noise(h, w, params["scale"], rng)
    pos = np.dstack(np.meshgrid(np.linspace(0,1,w,endpoint=False),
                                np.linspace(0,1,h,endpoint=False)))
    pos = pos.astype(np.float32)
    img = np.zeros((h,w,3), dtype=np.float32)
    for i in range(params["steps"]):
        # Advect positions along gradient-like flow
        vx = np.cos(noise*2*np.pi)
        vy = np.sin(noise*2*np.pi)
        pos[...,0] = (pos[...,0] + params["jitter"]*vx/w) % 1.0
        pos[...,1] = (pos[...,1] + params["jitter"]*vy/h) % 1.0
        # Sample colors from evolving phases
        phase = (i/params["steps"]) * 2*np.pi
        c = np.dstack([
            0.5+0.5*np.sin(phase + 6.0*pos[...,0]),
            0.5+0.5*np.sin(phase + 5.0*pos[...,1]),
            0.5+0.5*np.sin(phase + 7.0*(pos[...,0]+pos[...,1]))
        ])
        img = 0.99*img + 0.01*c
    img = np.clip(img, 0, 1)
    return (img*255).astype(np.uint8)


def render_rd(h, w, params, seed) -> np.ndarray:
    rng = np.random.default_rng(seed)
    U = np.ones((h, w), dtype=np.float32)
    V = np.zeros((h, w), dtype=np.float32)
    # Seed with noise
    V[rng.integers(0,h,50), rng.integers(0,w,50)] = 1.0
    def lap(X):
        return (-4*X + np.roll(X,1,0)+np.roll(X,-1,0)+np.roll(X,1,1)+np.roll(X,-1,1))
    f,k,dt,Du,Dv = params["f"], params["k"], params["dt"], params["diff_u"], params["diff_v"]
    for _ in range(params["iters"]):
        UVV = U*V*V
        U += (Du*lap(U) - UVV + f*(1-U))*dt
        V += (Dv*lap(V) + UVV - (f+k)*V)*dt
        U = np.clip(U,0,1); V = np.clip(V,0,1)
    # Map to color
    img = np.stack([U, V, 1-U], axis=-1)
    img = (img - img.min())/(img.max()-img.min()+1e-8)
    return (img*255).astype(np.uint8)


def render_spectral(h, w, params, seed) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((h,w), dtype=np.float32)
    for _ in range(params["bands"]):
        fx = rng.integers(1, min(64,w//4))
        fy = rng.integers(1, min(64,h//4))
        phase = rng.random()*2*np.pi
        Y, X = np.mgrid[0:h,0:w]
        img += np.sin(2*np.pi*(X*fx/w + Y*fy/h) + phase)
    img = img / max(1, params["bands"]) 
    # Nonlinear energy redistribution
    img = np.sign(img) * np.power(np.abs(img), params["alpha"]).astype(np.float32)
    # Simple color mapping
    rgb = np.stack([
        0.5+0.5*np.sin(2.1*img),
        0.5+0.5*np.sin(1.7*img+1.3),
        0.5+0.5*np.sin(1.3*img+2.1)
    ], axis=-1)
    rgb = (rgb - rgb.min())/(rgb.max()-rgb.min()+1e-8)
    return (rgb*255).astype(np.uint8)


def render(genome: Genome, h: int, w: int) -> np.ndarray:
    if genome.op == "flow":
        return render_flow(h,w,genome.params,genome.seed)
    elif genome.op == "rd":
        return render_rd(h,w,genome.params,genome.seed)
    else:
        return render_spectral(h,w,genome.params,genome.seed)

# --------------------------- Evaluation -------------------------------------

try:
    import imagehash
except Exception:
    imagehash = None


def compressibility_score(img: np.ndarray) -> float:
    # MDL proxy: smaller compressed size -> higher order (lower score). We invert so higher is better.
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
    # Predict next frame by linear extrapolation: prev + (prev - prevprev)
    pred = _prev_img.astype(np.float32) + (_prev_img.astype(np.float32) - _prev_prev_img.astype(np.float32))
    pred = np.clip(pred,0,255)
    err = np.mean((img.astype(np.float32) - pred)**2)
    # update history lazily outside to allow combined scoring decisions
    return float(err/ (255.0**2))


def homeostasis_score(img: np.ndarray) -> float:
    # Encourage multi-scale structure: reward mid-frequency energy via FFT band-pass
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
    # raw terms
    sc = {
        "compress": compressibility_score(img),   # wide range (≈40..200+)
        "surprise_raw": surprise_score(img),      # ≈0..0.2
        "homeo": homeostasis_score(img),          # ≈0.1..3.5
    }
    sc["novelty"] = novelty_score(img, archive)

    # shape surprise toward moderate change (avoid frozen or chaotic extremes)
    mu, sigma = 0.02, 0.02
    sr = sc["surprise_raw"]
    surprise_s = float(np.exp(-((sr - mu)**2) / (2*sigma**2)))

    # squash/normalize to ~[0,1] so weights are meaningful
    compress_s = float(np.tanh(sc["compress"] / 100.0))      # 100→~0.76, 200→~0.96
    homeo_s    = float(np.clip(sc["homeo"] / 2.5, 0.0, 1.0)) # ≥2.5 is “excellent” and saturates

    # keep raw terms for the ledger
    sc["compress_s"] = compress_s
    sc["homeo_s"] = homeo_s
    sc["surprise"] = surprise_s

    sc["total"] = (
        0.35*homeo_s +
        0.30*compress_s +
        0.25*surprise_s +
        0.10*sc["novelty"]
    )
    return sc




# --------------------------- Tournament & Refusal ----------------------------

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

def durability_score(img: np.ndarray, archive: Dict[str, float]) -> float:
    # 0.5px Gaussian blur = minimal perceptual perturbation
    imgf = gaussian_filter(img.astype(np.float32), sigma=(0.5, 0.5, 0))
    imgf = np.clip(imgf, 0, 255).astype(np.uint8)
    return joint_score(imgf, sensor_packet(), archive)["total"]


# --------------------------- Main Loop --------------------------------------

def save_image(img: np.ndarray, path: str):
    Image.fromarray(img).save(path)

archive: Dict[str, float] = {}

def run(out: str, minutes: float, width: int, height: int, epsilon: float):
    global _prev_img, _prev_prev_img
    ensure_dir(out)
    rng = random.Random()
    # bootstrap state
    current = make_genome(rng)
    current_img = render(current, height, width)
    current_score = joint_score(current_img, sensor_packet(), archive)
    _prev_prev_img = _prev_img = current_img
    start = time.time()
    n = 0
    while time.time() - start < minutes*60:
        cand, cand_img, cand_score = tournament(height, width, rng)

        if cand_score["total"] > current_score["total"] * (1 + epsilon):
            # Durability gate: candidate must still beat current after a tiny perturbation
            dur = durability_score(cand_img, archive)
            if dur <= current_score["total"] * (1 + epsilon):
                time.sleep(0.2)
                continue

            # ... proceed to promote
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

        n += 1
    # Save final image if not already saved recently
    tag = f"final_{int(time.time())}_{genome_hash(current)}"
    save_image(current_img, os.path.join(out, f"frame_{tag}.png"))
    print("[END] final score:", json.dumps(current_score))

# --------------------------- CLI -------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='out')
    p.add_argument('--minutes', type=float, default=1.0)
    p.add_argument('--width', type=int, default=1080)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--epsilon', type=float, default=0.06, help='min fractional improvement to replace')
    args = p.parse_args()
    run(args.out, args.minutes, args.width, args.height, args.epsilon)
