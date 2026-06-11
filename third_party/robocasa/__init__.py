"""RoboCasa subset vendored for REVAMP TurnOnSinkFaucet experiments.

Only the kitchen base environment and sink-faucet atomic tasks are exposed in
this release artifact. The broader RoboCasa task suite was intentionally
pruned to keep the reviewer-facing code focused and compact.
"""

try:
    from robosuite.environments.base import make
except ModuleNotFoundError:
    def make(*args, **kwargs):
        raise ModuleNotFoundError(
            "robosuite is required to instantiate RoboCasa environments. "
            "Set REVAMP_ROBOSUITE_ROOT or install robosuite before running real rollouts."
        )


try:
    from robocasa.environments.kitchen.kitchen import Kitchen
    from robocasa.environments.kitchen.atomic.kitchen_sink import (
        AdjustWaterTemperature,
        TurnOffSinkFaucet,
        TurnOnSinkFaucet,
        TurnSinkSpout,
    )
except ModuleNotFoundError:
    Kitchen = None
    TurnOnSinkFaucet = None
    TurnOffSinkFaucet = None
    TurnSinkSpout = None
    AdjustWaterTemperature = None

__all__ = [
    "make",
    "Kitchen",
    "TurnOnSinkFaucet",
    "TurnOffSinkFaucet",
    "TurnSinkSpout",
    "AdjustWaterTemperature",
]
