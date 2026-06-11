# External Assets And Experiment Files

This source package stores launch configs and source code only. Datasets
and checkpoints are intentionally excluded from this folder and should be
downloaded as a separate artifact bundle when full experiments are run.

## Included In This Code Package

- `configs/turn_on_sink_faucet.json`: main launch config.
- `configs/world_model/*.source.yaml`: source world-model and QAM configs.

## Expected Dataset Artifact

Place the dataset artifact at the repository root so these paths exist:

- `datasets/assets/t5_embedding/prompts/robocasa/TurnOnSinkFaucet.pt`: prompt embedding.
- `datasets/origin`: shared trimmed success/failure rollouts for Stage3 QAM and world-model retraining.
- `datasets/online_recovery_rollouts`: online branch-success recovery rollouts.
- `datasets/new_scene_data`: new-scene / new-environment success/failure rollouts merged into the same online WM update.

## Expected Checkpoint Artifact

Place the checkpoint artifact at the repository root so these paths exist:

- `checkpoints/stage2_q_initial`: complete Stage-2 WorldModel checkpoint used by Stage3 QAM for both Q gradients and A2 imagination.
- `checkpoints/openpi_pi0_turn`: OpenPI pi0 checkpoint used by Stage3 QAM.
- `checkpoints/world_model_initial`: initial Stage-1 world-model checkpoint used as the resume point for online world-model retraining.
- `checkpoints/world_model_after_online_update`: final Stage-1 checkpoint after merged branch + new-scene online update; only needed for rollout-check / final-result verification.

## Checkpoint Packaging

Do not commit full checkpoint weights to the main source repository. This
source package's `.gitignore` ignores the whole `checkpoints/` and
`datasets/` trees by default.

Recommended split:

- Source repo: REVAMP code, configs, docs, requirements, and trimmed
  `third_party/` compatibility subsets.
- Artifact bundle: `checkpoints/stage2_q_initial`,
  `checkpoints/openpi_pi0_turn`, `checkpoints/world_model_initial`, and
  optionally `checkpoints/world_model_after_online_update`.

Minimum checkpoint sets by entrypoint:

- `run_imagination_policy.py`: `stage2_q_initial` and `openpi_pi0_turn`.
- `run_world_model_update.py`: `world_model_initial`.
- `run_world_model_update.py --with-rollout-check`: `world_model_initial` and
  `world_model_after_online_update`.

## External Large Assets

Wan2.2-TI2V-5B weights and RoboCasa model assets are not copied by default.
Point to them with:

```bash
export REVAMP_WAN_CHECKPOINT_DIR=/path/to/Wan2.2-TI2V-5B
export REVAMP_ROBOCASA_ASSETS_ROOT=/path/to/robocasa_models/assets
```
