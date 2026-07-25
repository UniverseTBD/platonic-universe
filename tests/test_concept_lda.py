"""Tests for pu.analysis.concept_lda (Latent Dirichlet Allocation on embeddings).

Deterministic, CPU-only, no downloads. Small sizes keep it fast; the substantive
end-to-end recovery numbers live in the tutorial notebook
(experiments/lda_concepts/).
"""
import numpy as np
import pytest

from pu.analysis import concept_lda as cl


def _planted(rotate=False, seed=0):
    return cl.make_planted_concepts(n_per_class=120, dim=36, rotate=rotate,
                                    random_state=seed)


# --------------------------------------------------------------------------- #
# Tokenizers                                                                   #
# --------------------------------------------------------------------------- #

def test_percentile_sign_shapes_and_vocab():
    X, y, _, _ = _planted()
    tok = cl.PercentileSignTokenizer().fit(X)
    counts = tok.transform(X)
    assert counts.shape == (len(X), 2 * 36)
    assert counts.min() >= 0 and counts.max() <= tok.max_count
    assert len(tok.feature_names) == 2 * 36
    assert tok.feature_names[0] == "dim0+" and tok.feature_names[1] == "dim0-"
    assert tok.word_directions().shape == (2 * 36, 36)


def test_percentile_sign_word_directions_are_signed_axes():
    X, _, _, _ = _planted()
    tok = cl.PercentileSignTokenizer().fit(X)
    wd = tok.word_directions()
    # word 0 = (dim0, +1) -> +e_0 ; word 1 = (dim0, -1) -> -e_0
    assert wd[0, 0] == 1.0 and wd[1, 0] == -1.0
    assert wd[0, 1:].sum() == 0.0


def test_percentile_sign_rejects_bad_percentiles():
    with pytest.raises(ValueError):
        cl.PercentileSignTokenizer(hi=20, lo=80)


def test_codebook_is_product_quantized_bag():
    X, _, _, _ = _planted()
    tok = cl.CodebookTokenizer(n_groups=6, words_per_group=5, random_state=0).fit(X)
    counts = tok.transform(X)
    assert counts.shape == (len(X), 30)
    assert tok.n_words == 30
    # every document is a genuine multi-word bag (not one-hot).
    assert (counts.sum(axis=1) > 1).all()
    assert tok.word_directions().shape == (30, 36)


def test_codebook_group_directions_live_in_their_slice():
    X, _, _, _ = _planted()
    tok = cl.CodebookTokenizer(n_groups=6, words_per_group=5, random_state=0).fit(X)
    wd = tok.word_directions()
    # a word from group 0 must be zero outside group 0's coordinate slice.
    sl0 = set(tok.slices_[0].tolist())
    outside = [j for j in range(36) if j not in sl0]
    assert np.allclose(wd[0, outside], 0.0)


# --------------------------------------------------------------------------- #
# ConceptLDA + decoding                                                       #
# --------------------------------------------------------------------------- #

def test_doc_topic_and_topic_word_are_distributions():
    X, _, _, _ = _planted()
    counts = cl.PercentileSignTokenizer().fit_transform(X)
    lda = cl.ConceptLDA(n_topics=3, random_state=0, max_iter=40).fit(counts)
    theta = lda.doc_topic(counts)
    assert theta.shape == (len(X), 3)
    assert np.allclose(theta.sum(axis=1), 1.0)
    beta = lda.topic_word
    assert beta.shape == (3, 2 * 36)
    assert np.allclose(beta.sum(axis=1), 1.0)


def test_topic_directions_unit_norm():
    X, _, _, _ = _planted()
    tok = cl.PercentileSignTokenizer().fit(X)
    lda = cl.ConceptLDA(3, random_state=0, max_iter=40).fit(tok.transform(X))
    dirs = cl.topic_directions(lda, tok, normalize=True)
    assert dirs.shape == (3, 36)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)


def test_lda_is_deterministic():
    X, _, _, _ = _planted()
    counts = cl.PercentileSignTokenizer().fit_transform(X)
    a = cl.ConceptLDA(3, random_state=0, max_iter=40).fit(counts).topic_word
    b = cl.ConceptLDA(3, random_state=0, max_iter=40).fit(counts).topic_word
    assert np.allclose(a, b)


# --------------------------------------------------------------------------- #
# Evaluation utilities                                                        #
# --------------------------------------------------------------------------- #

def test_align_topics_recovers_planted_classes_axis_aligned():
    X, y, _, _ = _planted()
    tok = cl.PercentileSignTokenizer().fit(X)
    lda = cl.ConceptLDA(3, random_state=0, max_iter=100).fit(tok.transform(X))
    match = cl.align_topics_to_labels(lda.doc_topic(tok.transform(X)), y)
    # the whole point: unsupervised topics line up with the hand labels.
    assert match.accuracy > 0.85
    assert set(match.class_to_topic) == {0, 1, 2}
    assert match.confusion.shape == (3, 3)


def test_direction_separation_and_recovery_are_high_axis_aligned():
    X, y, planted, _ = _planted()
    tok = cl.PercentileSignTokenizer().fit(X)
    lda = cl.ConceptLDA(3, random_state=0, max_iter=100).fit(tok.transform(X))
    match = cl.align_topics_to_labels(lda.doc_topic(tok.transform(X)), y)
    dirs = cl.topic_directions(lda, tok)
    sep = cl.direction_separation(dirs, tok.scaler_.transform(X), y, match.class_to_topic)
    rec = cl.recovery_cosine(dirs, planted, match.class_to_topic)
    assert min(sep.values()) > 0.9          # each class separable along its direction
    assert np.mean(list(rec.values())) > 0.6  # recovered dirs align with planted truth


def test_steer_moves_along_direction():
    d = np.array([1.0, 0.0, 0.0])
    x = np.zeros(3)
    assert np.allclose(cl.steer(x, d, 2.5), [2.5, 0.0, 0.0])


def test_make_planted_concepts_shapes_and_rotation():
    X, y, planted, names = cl.make_planted_concepts(n_per_class=50, dim=36, random_state=1)
    assert X.shape == (150, 36) and set(np.unique(y)) == {0, 1, 2}
    assert planted.shape == (3, 36)
    assert np.allclose(np.linalg.norm(planted, axis=1), 1.0)
    assert names == ["past", "present", "future"]
    Xr, _, plr, _ = cl.make_planted_concepts(n_per_class=50, dim=36, rotate=True, random_state=1)
    # rotation preserves unit norm but de-sparsifies the planted directions.
    assert np.allclose(np.linalg.norm(plr, axis=1), 1.0)
    assert (np.abs(plr) > 1e-6).sum() > (np.abs(planted) > 1e-6).sum()
