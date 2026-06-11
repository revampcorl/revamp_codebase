"""TurnOnSinkFaucet-only RoboCasa dataset registry for the REVAMP release."""

from collections import OrderedDict

from robocasa.utils.dataset_registry_utils import get_ds_soup


ATOMIC_TASK_DATASETS = OrderedDict(
    TurnOnSinkFaucet=dict(
        target=dict(
            human_path="v1.0/target/atomic/TurnOnSinkFaucet/20250812",
        ),
        horizon=300,
    ),
)

COMPOSITE_TASK_DATASETS = OrderedDict()

TASK_SET_REGISTRY = {
    "atomic_seen": ["TurnOnSinkFaucet"],
    "target_atomic_seen": ["TurnOnSinkFaucet"],
    "turn_on_sink_faucet": ["TurnOnSinkFaucet"],
}

DATASET_SOUP_REGISTRY = {
    "target_atomic_seen": get_ds_soup(split="target", task_set="atomic_seen", source="human"),
    "turn_on_sink_faucet": get_ds_soup(split="target", task_set="atomic_seen", source="human"),
}
