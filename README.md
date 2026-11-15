# Image generation on GitHub Actions

Runs a Stable Diffusion model on the free GitHub-hosted CPU runner
and commits the result to `output/`.

This repo is public, so standard runners are free and unlimited and
the runs do not touch the account's private-repo Actions minutes.

## Current model

`Lykon/dreamshaper-8` — SD 1.5 based, 25 steps, guidance 7.5,
DPM++ 2M Karras sampler, 512×512 (its native size — larger sizes
risk duplicated figures).

## How to run

- **Push a prompt:** edit `prompt.txt` and push — the run generates
  one image from that prompt. Pushing `generate.py` or the workflow
  also triggers a run with the current `prompt.txt`.
- **Manual:** Actions tab → "Generate image (SD, CPU)" →
  Run workflow → type any prompt and step count.

## Results

Each run writes:

- `output/image.png` — the generated image (latest run wins)
- `output/TIMING.md` — model load time, generation time, peak RAM
- `generate.log` — the full console log of the latest run, committed
  even when the run fails (GitHub hides Actions logs from signed-out
  viewers, so the log file is how failures get diagnosed)

The run commits these back with `[skip ci]`, so result commits never
trigger another run.

## Swapping models

Change `DEFAULT_MODEL` in `generate.py`. Turbo models automatically
resolve to 4 steps / guidance 0.0 with their built-in scheduler;
standard models resolve to 25 steps / guidance 7.5 with DPM++ Karras.
Tried before: `stabilityai/sd-turbo` (fast, 28–42s at 512px, but
prone to duplicated figures at 768px).

## Specs

- Runner: `ubuntu-latest` — 4 vCPU, 16 GB RAM, 14 GB disk on public repos
- Weights: fp16 files cast to float32 (half the download, same CPU speed)
- No GPU. CPU-only, a few minutes per image. Keep volume low.
