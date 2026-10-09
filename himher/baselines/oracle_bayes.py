"""Oracle Bayesian baseline (Section 5.4): the same fixed-library
posterior update as FixedLibraryBayesAgent, but with the true held-out
counterpart type included in the hypothesis space from the start -- an
upper-bound comparison, not a separate inference procedure.
"""

from __future__ import annotations

from typing import Sequence

from himher.baselines.fixed_library_bayes import FixedLibraryBayesAgent
from himher.domains.handoff.counterpart import BernoulliHandModel


def make_oracle_bayes_agent(
    library_models: Sequence[BernoulliHandModel], true_model: BernoulliHandModel
) -> FixedLibraryBayesAgent:
    hypotheses = (
        tuple(library_models)
        if true_model in library_models
        else (*library_models, true_model)
    )
    return FixedLibraryBayesAgent(hypotheses)
