import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import argparse
from natsort import natsorted
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default="banana_steer")
parser.add_argument("--n_trajs", type=int, default=10)
args = parser.parse_args()

evals_dir = os.path.join(args.checkpoint_path, "evals")
assert os.path.exists(evals_dir), f"No evals directory found at {evals_dir}"

# Find all eval pkl files
pkl_files = natsorted(glob.glob(os.path.join(evals_dir, "eval_traj_*.pkl")))
assert len(pkl_files) > 0, f"No eval pkl files found in {evals_dir}"

steps = []
success_rates = []
avg_times = []
std_times = []

for fp in pkl_files:
    # Parse step from filename: eval_traj_2000_0.pkl -> 2000
    basename = os.path.basename(fp)
    step = int(basename.split("_")[2])

    with open(fp, "rb") as f:
        transitions = pkl.load(f)

    # Group into episodes by done=True
    episodes = []
    current = []
    for t in transitions:
        current.append(t)
        if t["dones"]:
            episodes.append(current)
            current = []
    if current:  # incomplete last episode
        episodes.append(current)

    successes = [any(t["rewards"] > 0.5 for t in ep) for ep in episodes]
    times = []
    for ep in episodes:
        # estimate time as number of steps (each step ~control_freq seconds)
        # use info if available
        if "info" in ep[-1] and "episode" in ep[-1]["info"]:
            dt = ep[-1]["info"]["episode"].get("episode_duration", None)
            if dt is not None:
                times.append(dt)

    success_rate = np.mean(successes) if successes else 0.0
    steps.append(step)
    success_rates.append(success_rate)
    avg_times.append(np.mean(times) if times else 0.0)
    std_times.append(np.std(times) if times else 0.0)

    print(f"Step {step:>6} | Success rate: {success_rate:.1%} ({sum(successes)}/{len(successes)}) | Avg time: {np.mean(times):.1f}s" if times else f"Step {step:>6} | Success rate: {success_rate:.1%} ({sum(successes)}/{len(successes)})")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("STEER Eval — Banana Task", fontsize=14, fontweight="bold")

# Success rate
ax = axes[0]
ax.plot(steps, [r * 100 for r in success_rates], "o-", color="#2563eb", linewidth=2, markersize=8)
ax.fill_between(steps, 0, [r * 100 for r in success_rates], alpha=0.1, color="#2563eb")
ax.set_xlabel("Actor Checkpoint Step")
ax.set_ylabel("Success Rate (%)")
ax.set_title("Success Rate over Training")
ax.set_ylim(0, 105)
ax.set_xticks(steps)
ax.grid(True, alpha=0.3)
for x, y in zip(steps, success_rates):
    ax.annotate(f"{y:.0%}", (x, y * 100 + 2), ha="center", fontsize=9)

# Avg completion time (only if we have it)
ax2 = axes[1]
if any(t > 0 for t in avg_times):
    ax2.errorbar(steps, avg_times, yerr=std_times, fmt="o-", color="#16a34a",
                 linewidth=2, markersize=8, capsize=5)
    ax2.set_xlabel("Actor Checkpoint Step")
    ax2.set_ylabel("Avg Completion Time (s)")
    ax2.set_title("Completion Time over Training")
    ax2.set_xticks(steps)
    ax2.grid(True, alpha=0.3)
else:
    ax2.text(0.5, 0.5, "No timing data available", ha="center", va="center",
             transform=ax2.transAxes, fontsize=12, color="gray")
    ax2.set_title("Completion Time over Training")

plt.tight_layout()
out_path = os.path.join(args.checkpoint_path, "eval_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {out_path}")
plt.show()