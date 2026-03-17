# The Platonic Universe 🔮💫

### *Do Foundation Models See the Same Sky?*

> *"The things which are seen are temporal; but the things which are unseen are eternal."*
> — Someone who definitely wasn't talking about latent embeddings, but could have been

---

## A Dialogue Concerning the Representational Nature of the Cosmos

*The scene: A sun-drenched courtyard in ancient Athens, 626 BCE. A whiteboard has replaced the usual marble columns. PLATO stands before it, gesturing grandly at a diagram of neural network architectures. ARISTOTLE sits on a bench, laptop open, running `htop`. DIOGENES lounges in a large barrel nearby, illuminated by the glow of a single GPU. HYPATIA OF ALEXANDRIA has arrived with star charts under one arm and a HuggingFace API token under the other.*

---

## Act I: What Is This Thing?

**PLATO:** Friends! Gather round. I have discovered something magnificent. You know how I've always said that behind the messy world of appearances, there exist perfect, ideal Forms?

**DIOGENES:** *(from inside his barrel)* Oh no. Here we go again.

**PLATO:** What if I told you that neural networks — trained on completely different tasks, with completely different architectures, on completely different data — are all converging toward the *same* representation of reality?

**ARISTOTLE:** *(looking up from his laptop)* You're going to need to be more specific.

**PLATO:** The **Platonic Representation Hypothesis**. As models grow larger and train on more data, their internal representations converge toward a shared statistical model of the world. A *Platonic ideal* of representation, if you will. We built this repository to test that hypothesis — on the cosmos itself.

**DIOGENES:** So you're saying every neural network is squinting at the same sky and slowly agreeing on what it sees?

**PLATO:** *Beautifully* put.

**DIOGENES:** I was being reductive.

**PLATO:** Same thing.

**HYPATIA:** *(unrolling a star chart)* If I may — the reason astronomy is the perfect testbed here is threefold. First, different observations of the same galaxies — optical images, infrared, spectroscopy — all emerge from the same underlying physics. Second, modern surveys give us the data volume to actually test convergence. And third, we have multiple modalities that are fundamentally different from each other but describe the same objects.

**ARISTOTLE:** She's right. Same galaxies, different instruments, different wavelengths. If models converge in their representations of *galaxies*, that's strong evidence for the hypothesis.

**DIOGENES:** And if they don't?

**PLATO:** Then I will simply redefine what "converge" means.

**DIOGENES:** Respect.

---

## Act II: The Installation

**ARISTOTLE:** Right. Enough philosophy. Let's talk about how to actually *use* this.

**DIOGENES:** Finally.

**ARISTOTLE:** First, you need Python 3.11 or higher. Clone the repository and install with `uv`:

```bash
git clone https://github.com/UniverseTBD/platonic-universe.git
cd platonic-universe
```

```bash
pip install uv
uv sync
uv pip install .
```

**DIOGENES:** What's `uv`?

**ARISTOTLE:** It's a fast Python package manager. Think of it as pip but it doesn't make you wait long enough to question your career choices.

**DIOGENES:** And if I refuse to use `uv` on principle? I live in a barrel. I have principles.

**ARISTOTLE:** *(sighing)* Fine. You can do it the old-fashioned way:

```bash
python -m venv pu_env
source pu_env/bin/activate
pip install .
```

**HYPATIA:** And if you want SAM2 support — Meta's Segment Anything Model — you'll need the optional dependency:

```bash
pip install ".[sam2]"
```

**DIOGENES:** What does SAM2 do?

**HYPATIA:** It segments anything.

**DIOGENES:** Even my patience with this conversation?

**HYPATIA:** Especially that.

---

## Act III: Running Experiments (or, "What Does This Button Do?")

**ARISTOTLE:** The package installs a CLI tool called `pu`. Four subcommands. I'll walk you through each one.

### `pu run` — Generate Embeddings

**ARISTOTLE:** This is the main event. Pick a model, pick a dataset to compare against HSC, and let it rip:

```bash
pu run --model vit --mode jwst
```

**PLATO:** HSC — the Hyper Suprime-Cam — is always the reference baseline. Every experiment compares HSC against a second modality.

**DIOGENES:** Why HSC?

**HYPATIA:** Ground-based optical imaging. It's our anchor point. We compare its representations against JWST infrared images, Legacy Survey optical images, SDSS, or DESI spectroscopy. Same galaxies, different eyes.

**ARISTOTLE:** Here are the useful flags:

```bash
# Quick test with 1,000 samples (for the impatient, i.e., Diogenes)
pu run --model dino --mode jwst --test

# Bigger test with 10,000 samples
pu run --model dino --mode jwst --test-10k

# Full run with custom batch size
pu run --model convnext --mode legacysurvey --batch-size 64

# Compute ALL metrics, not just MKNN and CKA
pu run --model ijepa --mode jwst --all-metrics

# Control galaxy resizing behavior
pu run --model vit --mode jwst --resize-mode fill
```

**DIOGENES:** What's the difference between `match` and `fill` resize modes?

**HYPATIA:** `match` aligns HSC and Legacy Survey to the compared survey's framing using fixed extents. `fill` uses adaptive per-galaxy Otsu cropping so each galaxy fills the frame. It's like the difference between photographing a galaxy from a fixed distance versus zooming in on each one individually.

**DIOGENES:** I understood maybe 60% of that.

**HYPATIA:** That's honestly above average.

### `pu compare` — Compute Metrics on Existing Embeddings

**ARISTOTLE:** Already have embeddings saved as Parquet? Compare them without re-running inference:

```bash
# Run all metrics on an embeddings file
pu compare data/embeddings.parquet --metrics all

# Just the hits — MKNN and CKA
pu compare data/embeddings.parquet --metrics mknn cka

# Compare a specific model size
pu compare data/embeddings.parquet --size large

# Cross-file comparison — compare embeddings from two different runs
pu compare data/run1.parquet --ref data/run2.parquet --mode hsc
```

**PLATO:** The results are saved to `data/` as JSON and also printed to stdout. Very civilized.

### `pu calibrate` — Statistical Calibration

**ARISTOTLE:** This runs permutation-based calibration to assess whether your similarity scores are statistically meaningful:

```bash
pu calibrate data/embeddings.parquet --metrics cka --n-permutations 1000

# With a fixed seed for reproducibility
pu calibrate data/embeddings.parquet --metrics cka mknn --seed 42
```

**PLATO:** It builds a null distribution by shuffling and tells you whether your observed alignment is real or just a coincidence.

**DIOGENES:** Like how Plato's students check if his ideas are real or just coincidence?

**PLATO:** My ideas are *Forms*, Diogenes. They transcend empirical verification.

**ARISTOTLE:** *(under his breath)* And yet here we are, verifying.

### `pu percentiles` — Compute Dataset Percentiles

```bash
pu percentiles --max-samples 10000 --output data/percentiles.json
```

### `pu benchmark` — Performance Benchmarking

**ARISTOTLE:** For when you want to squeeze every last FLOP out of your hardware:

```bash
# Basic benchmark
pu benchmark --model vit --mode jwst

# The works — AMP, torch.compile, caching, the lot
pu benchmark --model dino --mode jwst \
    --enable-amp \
    --enable-compile \
    --enable-cache \
    --pin-memory \
    --size large

# Save results and compare against a baseline
pu benchmark --model vit --mode jwst --output-json bench.json
pu benchmark --model vit --mode jwst --enable-amp --compare-baseline bench.json
```

**DIOGENES:** What do all those optimization flags do?

**ARISTOTLE:**

| Flag | What it does |
|------|-------------|
| `--enable-amp` | Float16 mixed precision — faster, less memory |
| `--enable-compile` | `torch.compile` — lets PyTorch optimize the compute graph |
| `--enable-cache` | Cache embeddings so you skip repeated inference |
| `--pin-memory` | Pin DataLoader memory for faster GPU transfers |
| `--persistent-workers` | Keep DataLoader workers alive between batches |

**DIOGENES:** I'm just going to use `--test` and call it a day.

**ARISTOTLE:** That is a valid lifestyle.

---

## Act IV: The Python API (for Those Who Prefer to Script)

**PLATO:** Not everyone wants to use the command line. Some prefer the elegance of direct invocation.

**DIOGENES:** "Elegance." It's a function call.

```python
import pu

# Run an experiment end-to-end
pu.run_experiment("vit", "sdss", batch_size=64, num_workers=0, knn_k=10)
```

**HYPATIA:** You can also use the metrics module directly to compare any two embedding matrices:

```python
from pu import metrics
import numpy as np

# Compare two embedding matrices
Z1 = np.random.randn(1000, 768)
Z2 = np.random.randn(1000, 768)

# Individual metrics
score = metrics.mknn(Z1, Z2, k=10)
print(f"MKNN alignment: {score:.4f}")

# Batch comparison
results = metrics.compare(Z1, Z2, metrics=["cka", "mknn", "procrustes"])
print(results)
# {"cka": 0.85, "mknn": 0.72, "procrustes": 0.15}
```

**PLATO:** Higher MKNN scores mean more similar representations. The models are seeing the same Forms!

**DIOGENES:** Or the same pixels. But sure. Forms.

---

## Act V: The Cast of Characters

### Models

**HYPATIA:** Let me introduce the models. Think of each one as a different philosopher trying to describe the same night sky.

| Alias | Model Family | Sizes | What it is |
|-------|-------------|-------|-----------|
| `vit` | Vision Transformer | Base, Large, Huge | The original. Attention is all you need (to look at galaxies). |
| `dino` | DINOv2 | Small, Base, Large, Giant | Self-supervised. Learns by staring at images without being told what they are. Very Diogenes. |
| `dinov3` | DINOv3 | ViT-S/16, ViT-S/16+, ViT-B/16, ViT-L/16, ViT-H/16+, ViT-7B/16 | The next generation. Six sizes, from small to 7 billion parameters. |
| `convnext` | ConvNeXtv2 | Nano, Tiny, Base, Large | CNNs that decided to dress up as transformers and attend the party anyway. |
| `ijepa` | I-JEPA | Huge, Giant | Predicts missing pieces of images in latent space. Very Plato — it reasons about the unseen. |
| `vjepa` | V-JEPA | Large, Huge, Giant | Same idea, but for video. Three sizes to choose from. |
| `hiera` | Hiera | Tiny, Small, Base-Plus, Large | Hierarchical vision transformer. |
| `vit-mae` | MAE | Base, Large, Huge | Masked autoencoder. Learns by reconstructing what it can't see. |
| `clip` | CLIP | Base, Large | Aligned with text. Knows what a galaxy *looks like* and what it's *called*. |
| `astropt` | AstroPT | 0.15M, 0.95M, 850M | The specialist. Trained specifically on astronomical data. |
| `sam2` | SAM 2 | Tiny, Small, Base-Plus, Large | Segment Anything. Installed separately (`pip install ".[sam2]"`). |
| `paligemma` | PaliGemma 2 | 3B, 10B, 28B | Vision-Language Model. |
| `llava_15` | LLaVA 1.5 | 7B, 13B | Vision-Language Model. |
| `llava_ov` | LLaVA-OneVision | 7B | Vision-Language Model. The new hotness. |

**DIOGENES:** That's a lot of models.

**HYPATIA:** The whole point is to see if they converge. You need variety.

**DIOGENES:** I contain multitudes. Do I converge?

**HYPATIA:** No. You diverge. Aggressively.

### Datasets

| Mode | Dataset | Type | Notes |
|------|---------|------|-------|
| `jwst` | JWST | Space-based infrared | The crown jewel. Different wavelengths, same galaxies as HSC. |
| `legacysurvey` | Legacy Survey | Ground-based optical | Overlapping footprint with HSC, different instrument. |
| `sdss` | SDSS | Spectroscopy + imaging | Forces `num_workers=0` to preserve draw order pairing. |
| `desi` | DESI | Spectroscopy | Same pairing constraint as SDSS. |

**PLATO:** HSC is always the reference. It is the cave wall upon which all shadows are cast.

**DIOGENES:** I thought you were *against* the cave wall.

**PLATO:** I contain multitudes.

**DIOGENES:** Hey, that's my line.

### Metrics

**ARISTOTLE:** We measure representational similarity with a frankly excessive number of metrics:

| Family | Metrics | What they measure |
|--------|---------|-------------------|
| **Kernel** | CKA, MMD | Global structure alignment via kernel comparisons |
| **Geometric** | Procrustes, Cosine Similarity, Frechet | Shape and distance after optimal alignment |
| **CCA** | SVCCA, PWCCA | Shared variance via canonical correlations |
| **Spectral** | Tucker Congruence, Eigenspectrum, Riemannian | Eigenstructure similarity |
| **Information** | KL Divergence, JS Divergence, Mutual Information | Information-theoretic overlap |
| **Neighbor** | MKNN, Jaccard, RSA | Local neighborhood agreement (the primary metric) |
| **Regression** | Linear R² | How well one representation linearly predicts the other |

**DIOGENES:** Which one matters?

**HYPATIA:** MKNN is the headline metric. It measures whether galaxies that are neighbors in one representation space are also neighbors in another. If two models organize galaxies the same way in their latent spaces, MKNN will be high.

**DIOGENES:** So it's a vibe check for neural networks.

**HYPATIA:** ...I'm going to allow it.

---

## Act VI: The Results

**PLATO:** Behold!

<img src="https://github.com/UniverseTBD/platonic-universe/blob/main/figs/mknn.png" width=100%/>

**PLATO:** As models grow larger, their representations become more similar — even across fundamentally different modalities. The Forms reveal themselves!

**ARISTOTLE:** More precisely: larger models trained on optical images (HSC) and infrared images (JWST) produce increasingly aligned embedding spaces, as measured by MKNN. The trend holds across architectures.

**DIOGENES:** So the big expensive models agree with each other more than the small cheap ones.

**ARISTOTLE:** Yes.

**DIOGENES:** Sounds like academia.

**HYPATIA:** *(laughing)* The important implication is practical — astronomical foundation models may not need to be trained from scratch on astronomical data. General-purpose vision models, scaled up sufficiently, may already encode the structure of the universe.

**PLATO:** *The structure of reality itself.*

**DIOGENES:** Plato, I'm begging you.

---

## Act VII: Architecture (for the Curious)

**ARISTOTLE:** For those who wish to understand or extend the code, here is how the repository is organized:

```
src/pu/
├── __main__.py          # CLI entry point (pu run/compare/calibrate/benchmark)
├── experiments.py       # Orchestrates the full pipeline
├── preprocess.py        # Image preprocessing
├── models/              # Model adapters (self-registering)
│   ├── base.py          #   Base adapter interface
│   ├── registry.py      #   Registry pattern
│   ├── hf.py            #   HuggingFace models (ViT, DINO, ConvNeXt, etc.)
│   ├── astropt.py       #   AstroPT adapter
│   └── sam2.py          #   SAM2 adapter
├── pu_datasets/         # Dataset adapters (self-registering)
│   ├── base.py          #   Base dataset interface
│   ├── registry.py      #   Registry pattern
│   ├── hf_crossmatched.py  # JWST & Legacy Survey (HuggingFace)
│   ├── sdss.py          #   SDSS
│   └── desi.py          #   DESI
├── metrics/             # Representational similarity metrics
│   ├── kernel.py        #   CKA, MMD
│   ├── geometric.py     #   Procrustes, Frechet
│   ├── cca.py           #   SVCCA, PWCCA
│   ├── spectral.py      #   Eigenspectrum, Tucker, Riemannian
│   ├── information.py   #   KL, JS, MI
│   ├── neighbors.py     #   MKNN, Jaccard, RSA
│   ├── regression.py    #   Linear R²
│   ├── calibration.py   #   Permutation-based calibration
│   └── io.py            #   Batch comparison & Parquet I/O
└── cpp/                 # C++/pybind11 CKA extension (compiled with OpenMP)
```

**ARISTOTLE:** Both models and datasets use a **self-registering adapter pattern**. Each adapter module registers itself on import via side-effect imports in `__init__.py`. To add a new model or dataset, write an adapter and import it in the corresponding `__init__.py`.

**DIOGENES:** So the modules register themselves just by existing?

**ARISTOTLE:** Correct.

**DIOGENES:** Very Descartes. "I import, therefore I am."

**ARISTOTLE:** Descartes comes later, Diogenes.

**DIOGENES:** Time is a flat circle. Like a galaxy.

**HYPATIA:** Galaxies aren't flat circles, they're —

**DIOGENES:** Let me have this.

---

## Epilogue: Contributing

**PLATO:** This project is open source under the AGPLv3.

**ARISTOTLE:** We welcome contributions. You can:

- Add support for new model architectures
- Include additional astronomical datasets
- Implement alternative similarity metrics
- Improve preprocessing pipelines

**HYPATIA:** We also hang out on the [UniverseTBD Discord](https://discord.gg/VQvUSWxnu9). Come say hello.

**DIOGENES:** Or don't. I certainly won't. I'll be in my barrel.

**ARISTOTLE:** Running tests, presumably:

```bash
uv run pytest                           # all tests
uv run pytest tests/test_metrics_kernel.py  # single file
uv run pytest -k "test_cka"            # by name
uv run ruff check src/                  # lint
```

**DIOGENES:** Presumably.

---

## Citation

*If you use this code in your research, Plato insists you cite the paper. Aristotle insists because it's good practice. Diogenes doesn't care but will judge you silently if you don't.*

```bibtex
@article{utbd2025,
	author = {{UniverseTBD} and Duraphe, K. and Smith, M. J. and Sourav, S. and Wu, J. F.},
	title = {{The Platonic Universe: Do Foundation Models See the Same Sky?}},
	journal = {ArXiv e-prints},
	year = {2025},
	eprint = {2509.19453},
	doi = {10.48550/arXiv.2509.19453}
}
```

---

*DIOGENES:* *(exiting, dragging his barrel)* "I have looked into the latent space. And the latent space looked back."

*PLATO:* That's... actually profound.

*DIOGENES:* Don't tell anyone.

*Exeunt.*
