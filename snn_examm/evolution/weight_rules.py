"""Weight initialization rules.

Port of weights/weight_rules.hxx.
"""

from __future__ import annotations

from enum import Enum


class WeightType(Enum):
    RANDOM = "random"
    XAVIER = "xavier"
    LAMARCKIAN = "lamarckian"


class WeightRules:
    """Three independent weight strategies, mirroring EXAMM's WeightRules.

    - weight_initialize: how to init the initial population (cannot be LAMARCKIAN)
    - weight_inheritance: how to handle weights during crossover
    - mutated_component_weight: how to init new components added by mutation
    """

    def __init__(
        self,
        weight_initialize: str = "xavier",
        weight_inheritance: str = "lamarckian",
        mutated_component_weight: str = "lamarckian",
    ):
        if weight_initialize == "lamarckian":
            raise ValueError("weight_initialize cannot be lamarckian (no parents for initial population)")

        self.weight_initialize = weight_initialize
        self.weight_inheritance = weight_inheritance
        self.mutated_component_weight = mutated_component_weight
