r"""Concept-LDA: Latent Dirichlet Allocation over continuous embeddings.

Motivation
----------
Klerings et al. (EMNLP 2025, *Steering Language Models in Multi-Token
Generation*) find **interpretable, orthogonal concept directions** in a model's
residual space (tense, aspect) and steer generation along them. They use *Linear*
Discriminant Analysis. This module pursues the same goal — recover directions in
embedding space that correspond to abstract concept classes, then steer — with
**Latent Dirichlet Allocation** (Blei, Ng & Jordan 2003) instead.

The gap, and the modification
-----------------------------
Latent Dirichlet Allocation is a generative model of **bags of discrete tokens**
(a document is a vector of word counts). Embeddings are **continuous, dense**
vectors, so LDA cannot be applied out of the box. The contribution here is a pair
of hand-designed **tokenizers** that turn an embedding into a bag of codewords,
plus a **decoder** that maps each learned topic back to a direction in the
original embedding space:

1. :class:`PercentileSignTokenizer` — axis-aligned, maximally interpretable.
   Rule (chosen by hand): standardize each coordinate; a coordinate that lands
   above a high percentile emits the word ``dim{d}+``, below a low percentile
   emits ``dim{d}-``, with multiplicity encoding how many "steps" beyond the
   threshold it is. Vocabulary size ``2 D``. A topic is then a distribution over
   "which coordinates fire high / low", i.e. a soft, signed, axis-aligned
   direction. This is the tokenizer to reach for when you *designed* the
   embedding geometry (the hand-built experiment).

2. :class:`CodebookTokenizer` — model-agnostic, non-axis-aligned. Split the
   embedding into contiguous sub-vectors and quantize each against its own
   ``k``-means codebook (product quantization), so a document becomes a genuine
   bag of prototype codewords that co-occur across the corpus. A topic is a
   distribution over prototypes, decoded to the prototype-weighted direction.
   Use this for real model embeddings, where concepts are not aligned with
   coordinate axes (the galaxy-classifier stretch goal).

Decoder (:func:`topic_directions`): for topic ``k`` with word distribution
``beta_k``, the concept direction is ``v_k = normalize(sum_w beta_k[w] * d_w)``
where ``d_w`` is each word's direction in embedding space (``±e_d`` for the
percentile-sign vocabulary; the centered centroid for the codebook vocabulary).
All directions live in the tokenizer's **standardized** coordinates; multiply by
``tokenizer.scale_`` to return to raw units.

Everything is deterministic given ``random_state`` and depends only on numpy,
scipy and scikit-learn (already repo dependencies).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score

# --------------------------------------------------------------------------- #
# Standardization shared by both tokenizers                                   #
# --------------------------------------------------------------------------- #

@dataclass
class _Standardizer:
    """Per-coordinate z-scoring; stores ``mean_`` / ``scale_`` after :meth:`fit`."""

    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "_Standardizer":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0  # guard constant columns
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_


# --------------------------------------------------------------------------- #
# Tokenizer 1: percentile-sign (axis-aligned, hand rules)                     #
# --------------------------------------------------------------------------- #

class PercentileSignTokenizer:
    """Discretize embeddings by thresholding each standardized coordinate.

    Parameters
    ----------
    hi, lo : float
        Upper / lower percentiles (0-100) of the standardized activations that
        define the "high" / "low" firing thresholds, per coordinate.
    max_count : int
        Cap on a word's multiplicity within one document (graded evidence).
    """

    def __init__(self, hi: float = 75.0, lo: float = 25.0, max_count: int = 4):
        if not 0 < lo < hi < 100:
            raise ValueError("require 0 < lo < hi < 100")
        self.hi, self.lo, self.max_count = hi, lo, max_count
        self.scaler_ = _Standardizer()
        self.thr_hi_: Optional[np.ndarray] = None
        self.thr_lo_: Optional[np.ndarray] = None
        self.dim_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "PercentileSignTokenizer":
        Z = self.scaler_.fit(X).transform(X)
        self.dim_ = Z.shape[1]
        self.thr_hi_ = np.percentile(Z, self.hi, axis=0)
        self.thr_lo_ = np.percentile(Z, self.lo, axis=0)
        return self

    @property
    def vocab(self) -> List[Tuple[int, int]]:
        """Ordered ``(dim, sign)`` pairs; index ``2d`` is ``(d, +1)``, ``2d+1`` is ``(d, -1)``."""
        return [(d, s) for d in range(self.dim_) for s in (+1, -1)]

    @property
    def feature_names(self) -> List[str]:
        return [f"dim{d}{'+' if s > 0 else '-'}" for d, s in self.vocab]

    @property
    def scale_(self) -> np.ndarray:
        return self.scaler_.scale_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return an ``(N, 2 D)`` non-negative integer count matrix."""
        Z = self.scaler_.transform(X)
        n, d = Z.shape
        counts = np.zeros((n, 2 * d), dtype=np.int64)
        hi_excess = Z - self.thr_hi_                    # >0 where coordinate fires high
        lo_excess = self.thr_lo_ - Z                    # >0 where coordinate fires low
        hi_c = np.clip(np.floor(hi_excess) + 1, 0, self.max_count)
        lo_c = np.clip(np.floor(lo_excess) + 1, 0, self.max_count)
        hi_c[hi_excess <= 0] = 0
        lo_c[lo_excess <= 0] = 0
        counts[:, 0::2] = hi_c.astype(np.int64)         # (d, +1)
        counts[:, 1::2] = lo_c.astype(np.int64)         # (d, -1)
        return counts

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def word_directions(self) -> np.ndarray:
        """``(2 D, D)`` matrix: row for ``(d, s)`` is ``s * e_d`` (standardized space)."""
        eye = np.eye(self.dim_)
        rows = []
        for d, s in self.vocab:
            rows.append(s * eye[d])
        return np.asarray(rows)


# --------------------------------------------------------------------------- #
# Tokenizer 2: codebook / bag-of-visual-words (model-agnostic)                #
# --------------------------------------------------------------------------- #

class CodebookTokenizer:
    """Discretize embeddings with product-quantization bag-of-words.

    A single continuous vector has to become a *bag* of tokens for LDA. A soft
    histogram over one global codebook collapses to one-hot in high dimensions
    (the nearest centroid dominates), leaving one token per document and no
    co-occurrence for LDA to learn from. Instead we **split the embedding into
    ``n_groups`` contiguous sub-vectors and quantize each group against its own
    ``words_per_group``-entry codebook**, so every document is a genuine
    ``n_groups``-word bag whose codewords co-occur across the corpus (the classic
    product-quantization / bag-of-visual-words construction).

    Vocabulary size is ``n_groups * words_per_group``; word ``(g, v)`` means
    "sub-vector ``g`` looks like its ``v``-th prototype". A topic is a
    distribution over these prototypes, decoded to the prototype-weighted
    direction in embedding space.

    Parameters
    ----------
    n_groups : int
        Number of contiguous coordinate groups to quantize independently.
    words_per_group : int
        Codebook size within each group.
    counts_per_group : int
        Pseudo-counts each group contributes, spread over its codewords by a
        softmax of negative distance (temperature ``tau``); ``tau -> 0`` gives a
        hard single token per group.
    """

    def __init__(self, n_groups: int = 8, words_per_group: int = 8,
                 tau: float = 0.3, counts_per_group: int = 6, random_state: int = 0):
        self.n_groups = n_groups
        self.words_per_group = words_per_group
        self.tau = tau
        self.counts_per_group = counts_per_group
        self.random_state = random_state
        self.scaler_ = _Standardizer()
        self.slices_: List[np.ndarray] = []              # coordinate indices per group
        self.centroids_: List[np.ndarray] = []           # (Vg, group_dim) per group
        self._scale_g: List[float] = []                  # per-group softmax length scale

    @property
    def n_words(self) -> int:
        return self.n_groups * self.words_per_group

    def fit(self, X: np.ndarray) -> "CodebookTokenizer":
        Z = self.scaler_.fit(X).transform(X)
        d = Z.shape[1]
        self.slices_ = [np.asarray(s) for s in np.array_split(np.arange(d), self.n_groups)]
        self.centroids_, self._scale_g = [], []
        for g, sl in enumerate(self.slices_):
            km = KMeans(n_clusters=self.words_per_group,
                        random_state=self.random_state + g, n_init=10).fit(Z[:, sl])
            self.centroids_.append(km.cluster_centers_)
            d2 = ((Z[:, sl][:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(-1)
            self._scale_g.append(float(np.median(d2.min(axis=1))) or 1.0)
        return self

    @property
    def feature_names(self) -> List[str]:
        return [f"g{g}_w{v}" for g in range(self.n_groups)
                for v in range(self.words_per_group)]

    @property
    def scale_(self) -> np.ndarray:
        return self.scaler_.scale_

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler_.transform(X)
        n = Z.shape[0]
        counts = np.zeros((n, self.n_words), dtype=np.int64)
        for g, sl in enumerate(self.slices_):
            cg = self.centroids_[g]                       # (Vg, gd)
            d2 = ((Z[:, sl][:, None, :] - cg[None, :, :]) ** 2).sum(-1)  # (N, Vg)
            logits = -d2 / (self.tau * self._scale_g[g] + 1e-12)
            logits -= logits.max(axis=1, keepdims=True)
            w = np.exp(logits)
            w /= w.sum(axis=1, keepdims=True)
            block = np.rint(self.counts_per_group * w).astype(np.int64)
            empty = block.sum(axis=1) == 0
            if empty.any():
                block[empty, d2[empty].argmin(axis=1)] = 1
            counts[:, g * self.words_per_group:(g + 1) * self.words_per_group] = block
        return counts

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def word_directions(self) -> np.ndarray:
        """``(V, D)`` matrix: each group prototype placed in its coordinate slice.

        Row ``(g, v)`` is the group-``g`` centroid ``v`` embedded into the full
        ``D`` space (zeros outside the group), centered by the group's mean
        prototype so a uniform topic decodes to ~0.
        """
        d = int(sum(len(s) for s in self.slices_))
        rows = []
        for g, sl in enumerate(self.slices_):
            centered = self.centroids_[g] - self.centroids_[g].mean(axis=0, keepdims=True)
            for v in range(self.words_per_group):
                vec = np.zeros(d)
                vec[sl] = centered[v]
                rows.append(vec)
        return np.asarray(rows)


# --------------------------------------------------------------------------- #
# Concept-LDA                                                                 #
# --------------------------------------------------------------------------- #

class ConceptLDA:
    """Latent Dirichlet Allocation over a count matrix from a tokenizer.

    Thin wrapper over :class:`sklearn.decomposition.LatentDirichletAllocation`
    with a fixed ``batch`` learning method for determinism.
    """

    def __init__(self, n_topics: int, random_state: int = 0,
                 max_iter: int = 200, doc_topic_prior: Optional[float] = None,
                 topic_word_prior: Optional[float] = None):
        self.n_topics = n_topics
        self.lda_ = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            learning_method="batch",
            max_iter=max_iter,
            doc_topic_prior=doc_topic_prior,
            topic_word_prior=topic_word_prior,
        )

    def fit(self, counts: np.ndarray) -> "ConceptLDA":
        self.lda_.fit(counts)
        return self

    def doc_topic(self, counts: np.ndarray) -> np.ndarray:
        """``(N, K)`` document-topic distribution (rows sum to 1)."""
        theta = self.lda_.transform(counts)
        return theta / theta.sum(axis=1, keepdims=True)

    @property
    def topic_word(self) -> np.ndarray:
        """``(K, V)`` topic-word distribution (rows sum to 1)."""
        comp = self.lda_.components_
        return comp / comp.sum(axis=1, keepdims=True)

    def fit_doc_topic(self, counts: np.ndarray) -> np.ndarray:
        return self.fit(counts).doc_topic(counts)


def topic_directions(lda: ConceptLDA, tokenizer, normalize: bool = True) -> np.ndarray:
    """Decode topics to ``(K, D)`` concept directions in standardized embedding space.

    ``v_k = sum_w beta[k, w] * word_direction[w]``, optionally L2-normalized.
    """
    beta = lda.topic_word                     # (K, V)
    word_dirs = tokenizer.word_directions()   # (V, D)
    dirs = beta @ word_dirs                    # (K, D)
    if normalize:
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        dirs = dirs / norms
    return dirs


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class TopicClassMatch:
    """Result of Hungarian-matching topics to labels."""

    topic_to_class: Dict[int, int]           # topic index -> class label
    class_to_topic: Dict[int, int]           # class label -> topic index
    accuracy: float                          # matched argmax-topic accuracy
    nmi: float                               # label-invariant agreement
    confusion: np.ndarray = field(repr=False)  # (K, C) counts


def align_topics_to_labels(doc_topic: np.ndarray, labels: np.ndarray) -> TopicClassMatch:
    """Match each topic to a class by maximizing argmax-topic / label agreement.

    Uses the Hungarian algorithm on the topic×class co-occurrence matrix, so the
    matching is optimal and label-permutation invariant.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    K = doc_topic.shape[1]
    assign = doc_topic.argmax(axis=1)
    conf = np.zeros((K, len(classes)), dtype=np.int64)
    for k in range(K):
        for j, c in enumerate(classes):
            conf[k, j] = int(np.sum((assign == k) & (labels == c)))
    rows, cols = linear_sum_assignment(-conf)              # maximize agreement
    topic_to_class = {int(k): int(classes[j]) for k, j in zip(rows, cols)}
    class_to_topic = {v: k for k, v in topic_to_class.items()}
    matched = sum(conf[k, list(classes).index(topic_to_class[k])]
                  for k in topic_to_class)
    accuracy = matched / len(labels)
    nmi = normalized_mutual_info_score(labels, assign)
    return TopicClassMatch(topic_to_class, class_to_topic, float(accuracy),
                           float(nmi), conf)


def direction_separation(directions: np.ndarray, X_std: np.ndarray,
                         labels: np.ndarray, class_to_topic: Dict[int, int]) -> Dict[int, float]:
    """Per-class AUC of projecting standardized embeddings onto the class direction.

    For class ``c`` matched to topic ``k``, project ``X_std`` onto ``v_k`` and
    score how well that 1-D projection separates class ``c`` from the rest (ROC
    AUC). ``1.0`` = perfectly separable along the recovered direction.
    """
    labels = np.asarray(labels)
    out = {}
    for c, k in class_to_topic.items():
        proj = X_std @ directions[k]
        y = (labels == c).astype(int)
        if 0 < y.sum() < len(y):
            out[int(c)] = float(roc_auc_score(y, proj))
    return out


def recovery_cosine(recovered: np.ndarray, planted: np.ndarray,
                    class_to_topic: Dict[int, int],
                    class_order: Optional[List[int]] = None) -> Dict[int, float]:
    """Absolute cosine between each recovered concept direction and its planted truth.

    ``planted`` is ``(C, D)`` in class order ``class_order`` (defaults to
    ``0..C-1``). Sign is ignored (a direction and its negation are the same
    axis), so the score is ``|cos|`` in ``[0, 1]``.
    """
    if class_order is None:
        class_order = list(range(planted.shape[0]))
    idx = {c: i for i, c in enumerate(class_order)}
    out = {}
    for c, k in class_to_topic.items():
        u = recovered[k]
        v = planted[idx[c]]
        denom = np.linalg.norm(u) * np.linalg.norm(v)
        out[int(c)] = float(abs(u @ v) / denom) if denom > 0 else 0.0
    return out


def steer(x_std: np.ndarray, direction: np.ndarray, alpha: float) -> np.ndarray:
    """Steer a standardized embedding along a (unit) concept direction by ``alpha``."""
    return np.asarray(x_std) + alpha * np.asarray(direction)


# --------------------------------------------------------------------------- #
# Controlled dataset with planted concept directions                          #
# --------------------------------------------------------------------------- #

def make_planted_concepts(
    n_per_class: int = 200,
    dim: int = 36,
    concept_dims: Optional[List[List[int]]] = None,
    signs: Optional[List[List[int]]] = None,
    concept_strength: float = 3.5,
    noise: float = 1.0,
    rotate: bool = False,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Synthesize embeddings with known, planted concept directions.

    Three abstract classes are planted by hand as sparse, signed patterns over a
    few coordinates (the rest are nuisance noise). Returns ``(X, y, planted,
    names)`` where ``planted`` is the ``(C, D)`` ground-truth direction matrix
    (unit rows). With ``rotate=True`` the whole space is hit with a fixed random
    rotation so the concepts are no longer axis-aligned (used to motivate the
    codebook tokenizer).

    The default three concepts are framed as verb tense — the abstract classes
    the reference paper steers — but the geometry is generic.
    """
    rng = np.random.default_rng(random_state)
    if concept_dims is None:
        # Each abstract class is planted over its own block of six coordinates,
        # with a hand-chosen sign pattern (the "rules made by hand").
        concept_dims = [[0, 1, 2, 3, 4, 5],
                        [12, 13, 14, 15, 16, 17],
                        [24, 25, 26, 27, 28, 29]]              # PAST / PRESENT / FUTURE
    if signs is None:
        signs = [[+1, +1, +1, -1, -1, -1],
                 [+1, -1, +1, -1, +1, -1],
                 [+1, +1, -1, -1, +1, +1]]
    names = ["past", "present", "future"][: len(concept_dims)]

    planted = np.zeros((len(concept_dims), dim))
    for c, (dims, sg) in enumerate(zip(concept_dims, signs)):
        for d, s in zip(dims, sg):
            planted[c, d] = s
        planted[c] /= np.linalg.norm(planted[c])

    X, y = [], []
    for c in range(len(concept_dims)):
        base = rng.normal(0, noise, size=(n_per_class, dim))
        base += concept_strength * planted[c][None, :]
        X.append(base)
        y.extend([c] * n_per_class)
    X = np.concatenate(X, axis=0)
    y = np.asarray(y)

    if rotate:
        q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        X = X @ q
        planted = planted @ q

    perm = rng.permutation(len(y))
    return X[perm], y[perm], planted, names
