# Robot Learning

How do you teach a robot to act? Start by learning how to transform simple distributions into complex ones (flow matching), then apply that machinery to learn behaviors — first from demonstrations, then from reward signals (reinforcement learning).

Materials drawn from two courses:

- [MIT 6.S184/6.S975](https://diffusion.csail.mit.edu) — Introduction to flow matching and diffusion models ([labs](https://github.com/eje24/iap-diffusion-labs))
- [UC Berkeley CS 285](https://github.com/berkeleydeeprlcourse/homework_spring2026) — Deep reinforcement learning, Spring 2026

Exercise 04 bridges the two: it uses the flow matching from 01–03 to learn robot action distributions from demonstrations.

| # | Directory | What you build |
|---|-----------|----------------|
| 01 | `01-odes-and-sdes/` | Simulate ODEs and SDEs; implement Euler and Euler-Maruyama integrators; run Langevin dynamics to sample from target distributions |
| 02 | `02-flow-and-score-matching/` | Train neural networks to learn vector fields that transform distributions using conditional flow matching and score matching |
| 03 | `03-conditional-image-generation/` | Generate MNIST digits with a diffusion transformer; implement classifier-free guidance for conditional generation |
| 04 | `04-imitation-learning/` | Build a flow matching policy for Push-T robot manipulation; compare against behavioral cloning |
| 05 | `05-policy-gradients/` | Implement REINFORCE with baselines and GAE on CartPole; move from imitating demonstrations to maximizing reward |
| 06 | `06-q-learning-and-actor-critic/` | Implement DQN for Atari (MsPacman) and Soft Actor-Critic for continuous control (HalfCheetah, Hopper) |
| 07 | `07-llm-rl/` | Fine-tune LLMs with REINFORCE and GRPO on math and format-copying tasks |
| 08 | `08-offline-rl/` | Learn robot manipulation policies from fixed datasets without environment interaction (SAC-BC, IQL) |
| 09 | `09-llm-rl-project/` | Implement RLHF methods (DPO, GRPO, reward modeling) and beat performance thresholds on instruction following |
| 10 | `10-offline-to-online-rl-project/` | Bridge offline and online RL for robot control; start from demonstrations, then improve with environment interaction |

## Setup

Every directory has its own `pyproject.toml` and uses [`uv`](https://github.com/astral-sh/uv) for environment management. To get started:

```bash
cd 01-odes-and-sdes
uv run jupyter notebook odes-and-sdes.ipynb
```

Berkeley exercises (04–10) also use [Weights & Biases](https://wandb.ai) for logging and [Modal](https://modal.com) for cloud GPU — see `04-imitation-learning/README.md` for first-time setup.
