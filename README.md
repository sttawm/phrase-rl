# phrase-rl

Advantage-weighted rephrase tuning against a frozen VLA. Full design: [EXPERIMENT.md](EXPERIMENT.md).

## Layout

- `src/phrase_rl/` — library + phase entry points
- `scripts/setup_pod.sh` — one-shot RunPod bootstrap (clone → run → done)
- `results/experiments.json` — one entry per experiment: commit, command, overview, metrics
- `results/checkpoints/` — trained models (best-val + final) + `metrics.json`; gitignored, synced explicitly
- `results/charts/` — committed plots

## Remote workflow

All pod work goes through GitHub — push scripts from local, `git pull && run` on the pod.
No scp'ing code. Pod: 1× 48GB GPU (A40/A6000) through Phase 0/1; 2× for Phase 2.
