"""Advanced GLMM internals used by :class:`splisosm.SplisosmGLMM`.

Most users interact with :class:`splisosm.SplisosmGLMM`. The classes and
helpers exported here are available for advanced model inspection, custom
fitting workflows, and tests.
"""

from __future__ import annotations

from splisosm.glmm.dataset import IsoDataset
from splisosm.glmm.glm import MultinomGLM
from splisosm.glmm.glmm import MultinomGLMM
from splisosm.glmm.likelihood import (
    log_prob_fastmult,
    log_prob_fastmult_batched,
    log_prob_fastmvn,
    log_prob_fastmvn_batched,
)
from splisosm.glmm.logger import PatienceLogger

__all__ = [
    "IsoDataset",
    "MultinomGLM",
    "MultinomGLMM",
    "PatienceLogger",
    "log_prob_fastmult",
    "log_prob_fastmult_batched",
    "log_prob_fastmvn",
    "log_prob_fastmvn_batched",
]
