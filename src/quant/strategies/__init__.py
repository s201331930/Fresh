from .base import Strategy
from .trend_following import TrendFollowing
from .mean_reversion import MeanReversion
from .dual_momentum import DualMomentum
from .breakout import Breakout

STRATEGY_REGISTRY = {
    "trend_following": TrendFollowing,
    "mean_reversion": MeanReversion,
    "dual_momentum": DualMomentum,
    "breakout": Breakout,
}

__all__ = [
    "Strategy",
    "TrendFollowing",
    "MeanReversion",
    "DualMomentum",
    "Breakout",
    "STRATEGY_REGISTRY",
]
