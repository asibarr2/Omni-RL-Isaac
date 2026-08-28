# Omni-RL-Isaac

A custom reinforcement learning environment for goal-directed robot navigation in NVIDIA Isaac Sim, built on top of Isaac Sim's Carter robot and trained with PPO via Stable-Baselines3.

## Overview

The robot (an NVIDIA Carter, extending Isaac Sim's JetBot example) is placed in a walled arena with a randomly positioned goal cube. Using a custom OpenAI Gym-compatible environment (`JetBotEnv`), the agent learns to navigate toward the goal while avoiding collisions with the surrounding walls.

## Environment

- **Simulator**: NVIDIA Isaac Sim (built against the 2021.2.1 standalone API)
- **Robot**: Carter, controlled via differential wheel velocity actions
- **Observations**: first-person RGB camera feed (128x128) plus a vector observation slot reserved for LiDAR-based range readings (partially implemented — see Known Limitations)
- **Sensors**: a contact sensor on the chassis for collision detection, plus an onboard LiDAR intended to feed distance readings into the observation space
- **Reward shaping**:
  - Positive reward proportional to reduction in distance to the goal each step
  - `+200` bonus and episode termination on reaching the goal
  - `-100` penalty and episode termination on colliding with any of the four surrounding walls
- **Episode length**: capped at 1500 steps

## Training

- **Algorithm**: PPO (Stable-Baselines3)
- **Policy**: MLP with `Tanh` activation, custom actor/critic network architecture (`[64, 32]` each)
- **Timesteps**: 500,000
- **Checkpointing**: periodic policy checkpoints saved every 10,000 steps

## Requirements

- NVIDIA Isaac Sim (2021.2.1 standalone examples environment) with the Isaac Sim Python API (`omni.isaac.*` modules) available
- `stable-baselines3`
- `torch`
- `gym`
- An NVIDIA GPU (training is configured for `device="cuda"`)

This project depends on Isaac Sim's bundled Python environment and internal APIs (`omni.isaac.core`, `omni.isaac.jetbot`, `omni.isaac.contact_sensor`, `omni.isaac.range_sensor`, `omni.isaac.occupancy_map`), so it must be run from within an Isaac Sim Python environment rather than a standalone virtualenv.

## Setup

Before training, create the policy checkpoint directory referenced in `train.py`:

```bash
mkdir mlp_policy
```

## Usage

**Train:**
```bash
python train.py
```
Set `headless=True` in `JetBotEnv(...)` to run training without the Isaac Sim viewport (faster), or `False` to visually watch training progress.

**Evaluate a trained policy:**
```bash
python eval.py
```

## Project structure

```
env.py          # Custom Gym environment wrapping the Isaac Sim Carter robot
train.py        # PPO training script (Stable-Baselines3)
eval.py         # Runs a trained policy in the environment
testenv.py      # Environment sanity checks
carter_test/    # Carter robot test assets/scripts
```

## Known limitations

- LiDAR integration is incomplete — `get_lidar_obs()` currently prints sensor readings for debugging but is not wired into `observation_space` or returned from `step()`/`reset()`; the environment currently learns from RGB only.
- File paths (e.g. the policy checkpoint directory in `train.py`) are hardcoded to a specific local Isaac Sim installation path and will need to be updated to match your own environment.
- Built against Isaac Sim 2021.2.1's API; newer Isaac Sim versions have since changed or renamed several of the `omni.isaac.*` modules used here.
