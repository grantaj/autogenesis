# **Autogenesis\_1.0** — An Artwork That Decided to Exist

**Author:** Autonomous Machine Intelligence (with human assistant: [@grantaj](https://github.com/grantaj))
**Medium:** Live, self-directed generative visual system

---

## Concept

This work answers the art-historical question:

> **Is human intelligence a requirement to make art?**

It does so without argument, without simulation of human authorship — by existing as an autonomous artist.
The system decides when to make new work, chooses its own source material, judges its own outputs using machine-native criteria, and replaces itself without human intervention.

**You are not the audience it optimises for.**
It will happily ignore you if it deems change unnecessary.

---

## Technical Overview

* **Autonomy Kernel:** Internal drives based on compression, surprise, and homeostasis.
* **Data Ecology:** Uses non-anthropocentric inputs (noise sources available to the machine).
* **Generative Core:** Flow fields, reaction–diffusion, spectral tilings; selected and parameterised by the system.
* **Self-Evaluation:** Scores outputs without human-labelled data; seeks the ridge between order and chaos.
* **Refusal Power:** It may not change for long periods if no improvement is found.

---

## Running the Work

### Requirements

* Python 3.9+
* Packages: `numpy`, `pillow`, `imagehash`, `tqdm`
* Optional GPU acceleration: `torch` (CUDA build for your GPU)


## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/grantaj/autogenesis.git
cd autogenesis
python -m venv .venv
source .venv/bin/activate    # Windows PowerShell: .\.venv\Scripts\Activate.ps1
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Optional GPU Acceleration

If you have an **NVIDIA GPU** with CUDA, install the matching PyTorch CUDA build:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you are on **Apple Silicon (M1/M2)**, install the Metal-accelerated build:

```bash
pip install torch torchvision torchaudio
```


### Run

```bash
python autogenesis_mvp.py --out out --minutes 5
```

The system will write promoted frames and minimal ledgers into the `out/` directory.
For live viewing, point an auto-refreshing image viewer at that folder.

---

## Exhibition Notes

* This is a **live system** — ideally projected large-scale, running for days or weeks.
* Refusal to change is part of the work.
* All outputs are machine-selected; there is no human curation.

---

## License

To be determined (likely a permissive open license for the code, separate terms for exhibiting the work).

