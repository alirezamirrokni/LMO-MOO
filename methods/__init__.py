from methods.baselines import (
    CAGrad, DynamicWeightAverage, FAMO, GradDrop, LinearScalarization, MGDA, MOON, NashMTL, PCGrad, RLW,
    ScaleInvariantLinearScalarization, STL, Uncertainty,
)
from methods.lmo_mgda import EntropicLMOMGDA
from methods.mgda_variants import MGDAWarm, MoCo, MoDo

METHODS = {
    "stl": STL,
    "ls": LinearScalarization,
    "scaleinvls": ScaleInvariantLinearScalarization,
    "rlw": RLW,
    "dwa": DynamicWeightAverage,
    "uw": Uncertainty,
    "mgda": MGDA,
    "pcgrad": PCGrad,
    "graddrop": GradDrop,
    "cagrad": CAGrad,
    "moco": MoCo,
    "modo": MoDo,
    "nashmtl": NashMTL,
    "famo": FAMO,
    "mgda_warm": MGDAWarm,
    "moon": MOON,
    "entropic_lmo_mgda": EntropicLMOMGDA,
}
