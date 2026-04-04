# Robot Learning

Exercises in flow matching, diffusion, and reinforcement learning — combining materials from two courses:

- **[MIT 6.S184/6.S975](https://diffusion.csail.mit.edu)** — Introduction to flow matching and diffusion models ([labs repo](https://github.com/eje24/iap-diffusion-labs), 01–03)
- **[UC Berkeley CS 285](https://github.com/berkeleydeeprlcourse/homework_spring2026)** — Deep reinforcement learning, Spring 2026 (exercises 04–08, projects 09–10)

The intended learning path: finish the flow matching and diffusion exercises first, then move on to reinforcement learning.

## Contents

| # | Directory | Topic | Source |
|---|-----------|-------|--------|
| 01 | `01-odes-and-sdes/` | Simulating ODEs and SDEs | MIT |
| 02 | `02-flow-and-score-matching/` | Flow matching and score matching | MIT |
| 03 | `03-conditional-image-generation/` | Conditional generation with CFG and diffusion transformers | MIT |
| 04 | `04-imitation-learning/` | Imitation learning (behavioral cloning, flow matching policy) | Berkeley |
| 05 | `05-policy-gradients/` | REINFORCE and policy gradient methods | Berkeley |
| 06 | `06-q-learning-and-actor-critic/` | DQN and Soft Actor-Critic | Berkeley |
| 07 | `07-llm-rl/` | RL for LLMs (GRPO, REINFORCE) | Berkeley |
| 08 | `08-offline-rl/` | Offline RL | Berkeley |
| 09 | `09-llm-rl-project/` | Final project: RLHF for instruction following | Berkeley |
| 10 | `10-offline-to-online-rl-project/` | Final project: offline-to-online RL | Berkeley |

## Setup

Each directory has its own README with setup instructions. Most Berkeley exercises use [`uv`](https://github.com/astral-sh/uv) for package management, [Weights & Biases](https://wandb.ai) for experiment tracking, and [Modal](https://modal.com) for cloud GPU compute. See `04-imitation-learning/README.md` for initial environment setup.

The MIT flow matching labs are Jupyter notebooks — no additional setup beyond a Python environment with PyTorch.
