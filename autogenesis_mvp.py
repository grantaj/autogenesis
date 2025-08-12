"""
Autogenesis_1.0 — v2 autonomous visual generator
-----------------------------------------------------------
Pivot to coherence+distance gating:
- Structural signature (radial power spectrum, edge histogram/density, component stats)
- Coherence metric (mid-band balance + edge-density bell curve)
- Promotions require BOTH coherence gain and structural distance
- Time-window durability retained
- Novelty archive persistence retained
- Records config and structural signature in ledger
"""

from __future__ import annotations
import argparse, os, time, math, random, zlib, json, hashlib, pickle
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel, label

try:
    import torch
    TORCH = True
    CUDA = torch.cuda.is_available()
except Exception:
    TORCH = False
    CUDA = False

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# --- Ledger ---

@dataclass
class LedgerEntry:
    t: float
    genome_hash: str
    genome: Dict[str, Any]
    score: Dict[str, float]
    sensor_summary: Dict[str, float]
    config: Dict[str, Any]
    struct: Dict[str, Any]

# --- Tunable configuration (CLI-exposed) ---

@dataclass
class Config:
    # legacy knobs (still used for drift/epsilon/durability)
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
    # v2 coherence/distance params
    rps_bins: int
    edge_bins: int
    delta_struct: float          # min structural distance to consider “new”
    coh_w_rps: float             # coherence weights
    coh_w_edges: float
    coh_edge_density_mu: float   # target edge density (bell curve)
    coh_edge_density_sigma: float
    coh_improve: float           # fractional improvement in coherence

# --- Sensors & genome ---

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

def render(genome: Genome, h: int, w: int) -> np.ndarray:
    if genome.op == "flow":
        return render_flow(h,w,genome.params,genome.seed)
    elif genome.op == "rd":
        return render_rd(h,w,genome.params,genome.seed)
    else:
        return render_spectral(h,w,genome.params,genome.seed)

# --- Legacy novelty (kept; used only to populate archive) ---

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

# --- v2 structural signature & coherence ---

def _radial_power_spectrum(gray: np.ndarray, bins: int = 32) -> np.ndarray:
    F = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(F)
    h, w = mag.shape
    Y, X = np.mgrid[-h//2:h//2, -w//2:w//2]
    R = np.sqrt(X**2 + Y**2)
    Rn = (R / (0.5 * min(h, w))).ravel()
    M = mag.ravel()
    edges = np.linspace(0, 1.0, bins + 1)
    hist = np.zeros(bins, dtype=np.float32)
    for i in range(bins):
        m = (Rn >= edges[i]) & (Rn < edges[i+1])
        if np.any(m):
            hist[i] = np.mean(M[m])
    s = hist / (np.sum(hist) + 1e-8)
    return s.astype(np.float32)

def _edge_histogram(gray: np.ndarray, bins: int = 32) -> tuple[np.ndarray, float]:
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    mag = np.hypot(gx, gy)
    hist, _ = np.histogram(mag, bins=bins, range=(0, mag.max() + 1e-8), density=True)
    hist = hist.astype(np.float32)
    thresh = np.percentile(mag, 85.0)
    density = float((mag > thresh).mean())
    return (hist / (hist.sum() + 1e-8)).astype(np.float32), density

def _component_stats(gray: np.ndarray) -> tuple[float, float]:
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    mag = np.hypot(gx, gy)
    t = np.percentile(mag, 92.0)
    bw = (mag > t).astype(np.uint8)
    lbl, n = label(bw)
    if n == 0:
        return 0.0, 0.0
    sizes = np.bincount(lbl.ravel())[1:].astype(np.float32)
    return float(n), float(np.mean(sizes))

def struct_signature(img: np.ndarray, rps_bins: int, edge_bins: int) -> Dict[str, Any]:
    gray = np.mean(img.astype(np.float32), axis=-1)
    gray = (gray - gray.min()) / (np.ptp(gray) + 1e-8)
    rps = _radial_power_spectrum(gray, bins=rps_bins)
    eh, edens = _edge_histogram(gray, bins=edge_bins)
    ncc, mean_cc = _component_stats(gray)
    return {
        "rps": rps,
        "edge_hist": eh,
        "edge_density": float(edens),
        "n_components_log": float(np.log1p(ncc)),
        "mean_component_log": float(np.log1p(mean_cc)),
    }

def struct_distance(sigA: Dict[str, Any], sigB: Dict[str, Any]) -> float:
    d_rps = float(np.mean(np.abs(sigA["rps"] - sigB["rps"])))
    d_eh  = float(np.mean(np.abs(sigA["edge_hist"] - sigB["edge_hist"])))
    d_ed  = abs(sigA["edge_density"] - sigB["edge_density"])
    d_nc  = abs(sigA["n_components_log"] - sigB["n_components_log"])
    d_mc  = abs(sigA["mean_component_log"] - sigB["mean_component_log"])
    return 0.45*d_rps + 0.35*d_eh + 0.10*d_ed + 0.05*d_nc + 0.05*d_mc

def joint_score(img: np.ndarray, sensors: Dict[str,float], archive: Dict[str,float], cfg: Config) -> Dict[str, float]:
    """v2: coherence only (0..1) combining RPS mid-band balance and edge-density bell curve."""
    gray = np.mean(img.astype(np.float32), axis=-1)
    gray = (gray - gray.min()) / (np.ptp(gray) + 1e-8)
    rps = _radial_power_spectrum(gray, bins=cfg.rps_bins)
    b0 = int(0.10 * cfg.rps_bins); b1 = int(0.45 * cfg.rps_bins)
    mid = float(np.mean(rps[b0:b1]))
    low = float(np.mean(rps[:b0]) + 1e-8)
    high = float(np.mean(rps[b1:]) + 1e-8)
    rps_coh = mid / (mid + 0.5*(low + high) + 1e-8)
    _, edens = _edge_histogram(gray, bins=cfg.edge_bins)
    mu, sig = cfg.coh_edge_density_mu, cfg.coh_edge_density_sigma
    edge_coh = float(np.exp(-((edens - mu)**2) / (2*sig*sig)))
    coherence = float(cfg.coh_w_rps * rps_coh + cfg.coh_w_edges * edge_coh)
    return {"coherence": coherence, "rps_coh": rps_coh, "edge_coh": edge_coh,
            "edge_density": edens, "total": coherence}

# --- Selection & durability ---

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

# --- Novelty archive persistence ---

def save_archive(path="archive.pkl"):
    with open(path, "wb") as f:
        pickle.dump(archive, f)

def load_archive(path="archive.pkl"):
    global archive
    if os.path.exists(path):
        with open(path, "rb") as f:
            archive.update(pickle.load(f))

archive: Dict[str, float] = {}

# --- Main loop ---

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
    current_sig = struct_signature(current_img, cfg.rps_bins, cfg.edge_bins)

    _prev_prev_img = _prev_img = current_img

    while time.time() - start < minutes*60:
        cand, cand_img, cand_score = tournament(height, width, rng, cfg=cfg)
        cand_sig = struct_signature(cand_img, cfg.rps_bins, cfg.edge_bins)
        dist = struct_distance(current_sig, cand_sig)

        # Drifted acceptance threshold (coherence improvement)
        age_min = (time.time() - last_promo_t) / 60.0
        eps_eff = max(cfg.min_epsilon, cfg.epsilon * math.exp(-age_min / cfg.drift_tau_min)) if cfg.drift_tau_min > 0 else cfg.epsilon

        coh_ok = cand_score["total"] >= current_score["total"] * (1.0 + max(cfg.coh_improve, eps_eff))
        dist_ok = dist >= cfg.delta_struct

        if coh_ok and dist_ok:
            # durability gate
            if not time_window_durability(cand_img, cfg):
                time.sleep(0.2)
                continue

            last_promo_t = time.time()
            _prev_prev_img, _prev_img = _prev_img, cand_img

            if imagehash is not None:
                key = str(imagehash.phash(Image.fromarray(cand_img)))
                archive[key] = cand_score["total"]

            current, current_img, current_score = cand, cand_img, cand_score
            current_sig = cand_sig

            tag = f"{int(time.time())}_{genome_hash(current)}"
            save_image(current_img, os.path.join(out, f"frame_{tag}.png"))

            led = LedgerEntry(
                t=time.time(),
                genome_hash=genome_hash(current),
                genome={"op": current.op, "params": current.params, "seed": current.seed},
                score=current_score,
                sensor_summary=sensor_packet(),
                config=asdict(cfg),
                struct={"distance_from_prev": float(dist),
                        **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in current_sig.items()}}
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
    # legacy scoring/durability knobs (still used for drift/epsilon/durability)
    p.add_argument('--w_homeo', type=float, default=0.35)
    p.add_argument('--w_compress', type=float, default=0.30)
    p.add_argument('--w_surprise', type=float, default=0.25)
    p.add_argument('--w_novelty', type=float, default=0.10)
    p.add_argument('--surprise_mu', type=float, default=0.02)
    p.add_argument('--surprise_sigma', type=float, default=0.02)
    p.add_argument('--dur_window', type=int, default=12, help='durability window (seconds)')
    p.add_argument('--dur_step', type=int, default=4, help='durability recheck step (seconds)')
    p.add_argument('--epsilon', type=float, default=0.06, help='base fractional improvement to replace')
    p.add_argument('--drift_tau', type=float, default=45.0, help='minutes; 0 disables drift')
    p.add_argument('--min_epsilon', type=float, default=0.02)
    # v2 coherence/distance params
    p.add_argument('--delta_struct', type=float, default=0.15)
    p.add_argument('--rps_bins', type=int, default=32)
    p.add_argument('--edge_bins', type=int, default=32)
    p.add_argument('--coh_w_rps', type=float, default=0.6)
    p.add_argument('--coh_w_edges', type=float, default=0.4)
    p.add_argument('--coh_edge_density_mu', type=float, default=0.08)
    p.add_argument('--coh_edge_density_sigma', type=float, default=0.05)
    p.add_argument('--coh_improve', type=float, default=0.02)
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
        rps_bins=args.rps_bins,
        edge_bins=args.edge_bins,
        delta_struct=args.delta_struct,
        coh_w_rps=args.coh_w_rps,
        coh_w_edges=args.coh_w_edges,
        coh_edge_density_mu=args.coh_edge_density_mu,
        coh_edge_density_sigma=args.coh_edge_density_sigma,
        coh_improve=args.coh_improve,
    )
    run(args.out, args.minutes, args.width, args.height, cfg)
