# Image generation on GitHub Actions

Runs Stable Diffusion Turbo (SD-Turbo, 1–4 steps) on the free
GitHub-hosted CPU runner and commits the result to `output/`.

This repo is public, so standard runners are free and unlimited and
the runs do not touch the account's private-repo Actions minutes.

## How to run

- **Push a prompt:** edit `prompt.txt` and push — the run generates
  one image from that prompt. Pushing `generate.py` or the workflow
  also triggers a run with the current `prompt.txt`.
- **Manual:** Actions tab → "Generate image (SD-Turbo, CPU)" →
  Run workflow → type any prompt and step count (1–4).

## Results

Each run writes:

- `output/image.png` — the generated 768×768 image (latest run wins)
- `output/TIMING.md` — model load time, generation time, peak RAM

The run commits these back with `[skip ci]`, so result commits never
trigger another run.

## Specs

- Runner: `ubuntu-latest` — 4 vCPU, 16 GB RAM, 14 GB disk on public repos
- Model: `stabilityai/sd-turbo` (fp16 weights cast to float32, 2.6 GB download)
- No GPU. CPU-only, a few minutes per image. Keep volume low.
