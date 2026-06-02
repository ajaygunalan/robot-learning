# Robot Learning

> **Always answer concisely and briefly.**
> **Always explain top-down: high level first, then drill to low level, step by step.**


How do you teach a robot to act? Start by learning how to transform simple distributions into complex ones (flow matching), then apply that machinery to learn behaviors — first from demonstrations, then from reward signals (reinforcement learning).

Materials drawn from two courses:

- [MIT 6.S184/6.S975](https://diffusion.csail.mit.edu) — Introduction to flow matching and diffusion models ([labs](https://github.com/eje24/iap-diffusion-labs))
- [UC Berkeley CS 285](https://github.com/berkeleydeeprlcourse/homework_spring2026) — Deep reinforcement learning, Spring 2026

Exercise 04 bridges the two: it uses the flow matching from 01–03 to learn robot action distributions from demonstrations.

| # | Directory | What you build |
|---|-----------|----------------|
| 01 | `01-odes-and-sdes/` | Simulate ODEs and SDEs; Euler and Euler-Maruyama integrators; Langevin dynamics to sample from target distributions |
| 02 | `02-flow-and-score-matching/` | Train networks to learn vector fields via conditional flow matching and score matching |
| 03 | `03-conditional-image-generation/` | Generate MNIST digits with a diffusion transformer; classifier-free guidance |
| 04 | `04-imitation-learning/` | Flow matching policy for Push-T manipulation; compare against behavioral cloning |
| 05 | `05-policy-gradients/` | REINFORCE with baselines and GAE on CartPole; imitation → reward maximization |
| 06 | `06-q-learning-and-actor-critic/` | DQN for Atari (MsPacman) and Soft Actor-Critic for continuous control (HalfCheetah, Hopper) |
| 07 | `07-llm-rl/` | Fine-tune LLMs with REINFORCE and GRPO on math and format-copying tasks |
| 08 | `08-offline-rl/` | Learn manipulation policies from fixed datasets, no env interaction (SAC-BC, IQL) |
| 09 | `09-llm-rl-project/` | RLHF methods (DPO, GRPO, reward modeling); beat instruction-following thresholds |
| 10 | `10-offline-to-online-rl-project/` | Bridge offline and online RL; start from demonstrations, improve via env interaction |

## Setup

Every directory has its own `pyproject.toml` and uses [`uv`](https://github.com/astral-sh/uv). Venvs live at `~/.venvs/robot-learning/<lab>/` (ext4 — the repo itself is on exFAT, which can't host venvs).

```bash
cd 01-odes-and-sdes
uv run jupyter notebook odes-and-sdes.ipynb
```

Berkeley exercises (04–10) use [Weights & Biases](https://wandb.ai) for logging and [Modal](https://modal.com) for optional cloud GPU — see `04-imitation-learning/README.md` for first-time setup. WandB is already authenticated via `~/.netrc` (machine `api.wandb.ai`).

## Running a lab (exFAT gotcha — read first)

`uv run` **fails** in-repo: it tries to symlink `.venv/` onto exFAT → `Operation not permitted`. Fix: point uv at the ext4 venv. Run **from the lab dir**:

```bash
UV_PROJECT_ENVIRONMENT=~/.venvs/robot-learning/<lab>/ \
  uv run src/<pkg>/train.py [flags]
```

Example (lab 04 MSE):
```bash
cd 04-imitation-learning
UV_PROJECT_ENVIRONMENT=~/.venvs/robot-learning/04-imitation-learning \
  uv run src/hw1_imitation/train.py --policy-type mse
```

- **WandB is online** (`wandb login` already done). The first **~3 min after `Using device: cuda` is silent** — imports + CUDA init off the exFAT/gdrive mount, **not a hang**. Don't kill it.
- Dataset auto-downloads to `data/` (31 MB); pass `--data-dir ~/.cache/robot-learning-data` to reuse the cached copy.
- Stop criterion: runs all epochs (no early-stop). Reward target is a **grading bar, not a stop trigger** — Ctrl+C once reward clears it if impatient.
- Plotting: `matplotlib` isn't in the venvs — add with `uv pip install --python ~/.venvs/robot-learning/<lab>/bin/python matplotlib`.
- **`log.csv` bug:** `Logger` freezes its header on the first `log()` call (loss only), so `eval/mean_reward` is dropped from the CSV — pull reward from the **WandB run history** (`wandb.Api().run(...).history(...)`) for plots.

## Local hardware

- CPU: i9-13980HX (24c/32t), 30 GB RAM
- GPU: RTX 4060 Laptop, **8 GB VRAM**

## Where each lab runs

Verified by reading the code. Classic-RL labs are CPU-bound (env stepping); the i9 beats Modal's 2-core T4. Run them **locally** — `run.py`/`run_dqn.py` auto-detect the 4060 (`pytorch_util.py:init_gpu`), `--no_gpu` forces CPU. Only the two LLM-RL labs genuinely need cloud.

| Lab | Run | Why |
|---|---|---|
| 04 imitation | local | small MLP / flow; fits 8 GB |
| 05 policy-grad | local | tiny MLPs, CPU-bound (README says so) |
| 06 q-learning + AC | local | small nets incl. Atari CNN; long but fits |
| 07 **llm-rl** | **cloud** | Qwen2.5-1.5B + LoRA; 8 GB too tight/slow |
| 08 offline-rl | local | small MLPs; 12h timeout = a *sweep*, not one run |
| 09 **llm-rl-project** | **cloud** | 2× resident 1.5B (policy+RM); OOMs 8 GB |
| 10 offline→online | local | small MLPs, CPU-bound |

## Modal (cloud) — only 07 & 09

- Per-second billing; $30/mo free credit + academic grant up to $10k ([modal.com/academics](https://modal.com/academics)).
- Both default to **H100 ($3.95/hr)** but only need **A100-40GB ($2.10/hr)** — peak VRAM ~10–25 GB (LoRA, no vLLM).
  - Lab 07: edit `DEFAULT_GPU` → `"A100-40GB"` (`scripts/modal_train.py:14`).
  - Lab 09: `DEFAULT_GPU` is bypassed (functions hardcode `"H100"`); use the existing `train_remote_a100` entrypoint instead.
- Modal containers lack `~/.netrc` → pass the WandB key as a `modal.Secret` (`WANDB_API_KEY` overrides netrc).

## Exercise mode

Stubs (`raise NotImplementedError`, `TODO`) are the learner's to write — guide, don't hand over the answer.

### STEP workflow (use for every homework)

**Setup — number all blanks in DO-ORDER.** Scan every file the homework asks you to fill. Mark each blank with a banner, numbered in the order they're *done* (not file position):

```python
# ╔═══════════════════ STEP N — START ═══════════════════╗
# WHAT: <one line — what this block produces>
# WHY : <one line — why it exists / where it fits>
# >>> write STEP N here <<<
# ╚═══════════════════ STEP N — END ═════════════════════╝
```

Banners stay **high-level: WHAT + WHY only, no code, no recipe, no formulas, no shapes.** All the detail — pseudocode, tensor shapes, line-by-line — lives in chat, not the comment. The banner just marks the spot and its purpose.

Then give the learner the STEP table: `STEP | file | spot | status`. If numbers ever drift from do-order, renumber so label == order.

**Going step → step:** finish a STEP fully, mark it `✅`, then point to the next number. Do dependency-driven order (e.g. define a policy → write its runner → run it → next policy), not file order.

**Within a STEP — context first, then line by line:**
1. **Context** — where this STEP sits in the overall script, and **what / why / how** before any code.
2. **One line at a time** — add a single line, explain what it does + the shape it produces, then **stop and wait** for the learner to say `ok`.
3. Repeat until the STEP is complete; remove the `raise` on the last line.

**Validate when asked / when unsure** — confirm an idiom against Context7 + real codebases (GitHub) before trusting it; report the match.

Each homework: number once, walk 1→N, context-then-line-by-line inside each. Everything stays easy and ordered.
