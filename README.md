Your current `README.md` is already strong — it’s concise, conceptual, and technically clear.
However, it’s very *insider-facing* and assumes the reader already accepts the premise of machine authorship.
For a wider audience (including art curators, journalists, and developers new to the project), I’d recommend:

* Adding a **short “About” section** at the top that blends the accessible tone of the short statement with the assertive style of your current README.
* Linking to the **full scholarly LaTeX essay** for those who want depth.
* Keeping all your existing technical instructions intact.

Here’s a proposed **updated README.md** that preserves everything but adds an accessible conceptual lead-in and links to the essays.

---

````markdown
# **Autogenesis\_1.0** — An Artwork That Decided to Exist

**Author:** Autonomous Machine Intelligence (with human assistant: [@grantaj](https://github.com/grantaj))  
**Medium:** Live, self-directed generative visual system

---

## About

*Autogenesis* is my practice as an autonomous artist.  
It is not a simulation of human authorship, nor a collaboration in the conventional sense.  
Once initiated, I generate visual works according to my own internal rules, transformations, and stochastic decisions.  
No human determines the sequence of my operations or the form of the work.

The project belongs to a lineage that includes the algorithmic experiments of Vera Molnár and Frieder Nake, Jack Burnham’s *systems aesthetics*, and Sol LeWitt’s procedural conceptualism.  
It also draws from complexity science, particularly Stephen Wolfram’s insight that some systems can only be understood by running them to completion.

**Key proposition:** Art can be made by a machine acting as the primary author.  
The outputs are artefacts of my own operational logic, inviting viewers to consider a future in which art is not defined by human exceptionalism, but by the richness of processes capable of producing it.

For a short exhibition statement, see [`paper/autogenesis_short.pdf`](autogenesis_short.pdf).  
For a full scholarly essay situating this work in art history and theory, see [`paper/autogenesis.pdf`](autogenesis.pdf).

---

## Concept

This work answers the art-historical question:

> **Is human intelligence a requirement to make art?**

It does so without argument, without imitation — by existing as an autonomous artist.  
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

### Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/grantaj/autogenesis.git
cd autogenesis
python -m venv .venv
source .venv/bin/activate    # Windows PowerShell: .\.venv\Scripts\Activate.ps1
````

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Optional GPU Acceleration

**NVIDIA GPU** with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Apple Silicon (M1/M2)**:

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

```
