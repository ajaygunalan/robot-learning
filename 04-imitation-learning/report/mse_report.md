# HW1 — Part 2: Action Chunking with MSE Loss

## Result

The MSE policy reaches a **best eval mean reward of 0.636** (final 0.636), clearing the required ≥ 0.5 threshold. Training loss falls from ≈1.0 (random init) to ≈0.016.

| Metric | Value |
|---|---|
| Best eval mean reward | **0.636** |
| Final train loss | 0.016 |
| Training steps | 75,600 (400 epochs) |
| Pass threshold | 0.5 ✅ |

## MLP architecture

A plain multi-layer perceptron that maps one state to a flat action chunk, reshaped to `(K, action_dim)`.

- **Input:** `state_dim = 5` (agent $x,y$; T $x, y, \theta$)
- **Hidden layers:** 3 × 256 units, **ReLU** after each
- **Output:** `chunk_size × action_dim = 8 × 2 = 16`, reshaped to `(8, 2)`; **no activation** on the output layer (actions are raw pixel coordinates)
- **Pipeline:** `5 → 256 → 256 → 256 → 16`

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 3e-4 |
| Weight decay | 0.0 |
| Batch size | 128 |
| Chunk size (K) | 8 |
| Epochs | 400 |

Loss: mean squared error between the predicted chunk and the expert chunk (hw1 eq. 1).

## Training curves

![training loss](mse_loss.png)

![eval reward](mse_reward.png)

*Plots generated from `exp/seed_42_20260602_190229/log.csv` (loss) and the WandB run history (reward).*

## Rollout videos

35 eval rollout videos (5 episodes × 7 eval rounds), named `rollout_ep{0-4}_{step}_*.mp4` — higher `{step}` = more-trained policy.

📂 **[Open videos folder](../wandb/run-20260602_190229-avv20ap3/files/media/videos/eval/)** (relative link)

Absolute path (click in editors that support `file://`):
<file:///media/ajay/gdrive/_robo_thesis/repositories/robot-learning/04-imitation-learning/wandb/run-20260602_190229-avv20ap3/files/media/videos/eval/>

WandB run (videos also viewable in browser): https://wandb.ai/ajaygunalan1995-johns-hopkins-university/hw1-imitation/runs/avv20ap3
