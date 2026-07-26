"""Analysis tools for embedding spaces.

Currently exposes ``concept_lda``: Latent Dirichlet Allocation adapted to run on
continuous embeddings, recovering interpretable concept directions.
"""
from pu.analysis.concept_lda import (  # noqa: F401
    CodebookTokenizer,
    ConceptLDA,
    PercentileSignTokenizer,
    align_topics_to_labels,
    direction_separation,
    make_planted_concepts,
    recovery_cosine,
    steer,
    topic_directions,
)

__all__ = [
    "PercentileSignTokenizer",
    "CodebookTokenizer",
    "ConceptLDA",
    "topic_directions",
    "align_topics_to_labels",
    "direction_separation",
    "recovery_cosine",
    "steer",
    "make_planted_concepts",
]
