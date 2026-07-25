# Concept-LDA: concept directions in embedding space with Latent Dirichlet Allocation

Recover interpretable **concept directions** in a model's embedding space using
**Latent Dirichlet Allocation** (Blei, Ng & Jordan 2003), and steer along them.

This reframes the goal of Klerings et al. (EMNLP 2025,
[*Steering Language Models in Multi-Token Generation*](https://aclanthology.org/2025.emnlp-main.435/))
— which finds concept directions with **Linear** Discriminant Analysis and
steers generation — using a topic model instead. The two share the acronym *LDA*
and nothing else; making the generative, bag-of-words model act on continuous
embeddings is the contribution.

## Contents

| File | What it is |
|------|-----------|
| [`concept_lda_tutorial.ipynb`](concept_lda_tutorial.ipynb) | The conference-style tutorial, runnable end-to-end on CPU, with figures. |
| [`galaxy_toy.py`](galaxy_toy.py) | Self-contained cheap galaxy-morphology classifier (procedural images + tiny CNN) for the stretch goal. |
| `../../src/pu/analysis/concept_lda.py` | The reusable, tested method (tokenizers, `ConceptLDA`, decoder, metrics). |
| `../../tests/test_concept_lda.py` | Unit tests. |

## The method in three steps

1. **Tokenize** each embedding into a bag of codewords. Two hand-designed rules:
   - `PercentileSignTokenizer` — a word per coordinate firing high/low. Axis-aligned,
     maximally interpretable; use it when *you* designed the geometry.
   - `CodebookTokenizer` — product-quantization bag-of-visual-words. Model-agnostic;
     use it for real, non-axis-aligned model embeddings.
2. **Fit** Latent Dirichlet Allocation (`ConceptLDA`) over the count matrix to get
   topics (`beta`) and per-embedding topic mixtures (`theta`).
3. **Decode** each topic to a direction, `topic_directions`:
   `v_k = normalize(sum_w beta[k, w] * word_direction[w])`.

Evaluate with `align_topics_to_labels` (Hungarian topic↔class matching, accuracy,
NMI), `recovery_cosine` (vs. planted ground truth), and `direction_separation`
(1-vs-rest AUC along each recovered direction). `steer(x, v, alpha)` moves an
embedding along a direction.

## What the notebook demonstrates (all executed on CPU)

- **Recovery on designed data** — three hand-planted classes (framed as verb
  tense): topic↔class accuracy ≈ 0.98, direction AUC = 1.0, recovered-vs-planted
  cosine ≈ 0.82.
- **Interpretability** — each topic's top codewords are *exactly* the coordinates
  that class was planted on, with the right signs.
- **Steering** — a "present" embedding steered along the recovered "past"
  direction crosses into the past cluster.
- **Axis-alignment stress test** — a random rotation collapses per-coordinate
  interpretability to chance (real models are not axis-aligned); the codebook
  tokenizer recovers the concepts anyway (AUC ≈ 1.0).
- **Model-agnostic stretch** — on a cheap galaxy CNN's embeddings, Concept-LDA
  recovers galaxy morphology unsupervised: edge-on and elliptical with AUC 1.0;
  spiral (morphologically intermediate: bulge + disk) is the honest hard case.

## Run it

```bash
pip install ".[analysis]"
jupyter nbconvert --to notebook --execute --inplace \
    experiments/lda_concepts/concept_lda_tutorial.ipynb
# or open it in Jupyter and Run All

pytest tests/test_concept_lda.py          # unit tests (CPU, deterministic)
```

To run the stretch goal on **real** data, swap the two marked lines in the galaxy
section for real Galaxy10 thumbnails and any astronomical foundation model, e.g.
`pu.models.get_adapter("dinov2")` / `"astropt"`; the Concept-LDA code downstream
does not change.

## How it fits Platonic Universe

A recovered concept direction is a **label-free, model-agnostic probe**. Extract
the same concept's direction from two models and compare them with the repo's
existing MKNN / CKA / Procrustes metrics to ask whether both models carry that
concept along comparable directions — the modality adapters supply the
embeddings, this supplies the concepts.
