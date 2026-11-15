# Image generation on GitHub Actions

Runs a Stable Diffusion model on the free GitHub-hosted CPU runner
and commits the result to `output/`. It also runs a Kokoro voice
model the same way — see Voice generation below.

This repo is public, so standard runners are free and unlimited and
the runs do not touch the account's private-repo Actions minutes.

## Current model

`RunDiffusion/juggernaut-xl-v9` — SDXL-based, tuned for photorealistic
portraits. 25 steps, guidance 7.5, DPM++ 2M Karras sampler, 768×768,
bfloat16 compute (float32 would need ~15 GB of weights alone and
would not fit the 16 GB runner). Slower than the SD 1.5 models — tens
of minutes per image on CPU — but visibly better faces and detail.

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

## Voice generation

Kokoro-82M (`kokoro-onnx`, int8, ~120 MB) turns text into
natural-sounding speech on the same CPU runner — no API key, no
quota. Default voice is `am_adam` (man, steady/clear); `am_michael`
(deeper/slower) and `am_onyx` (deepest/fastest) are one-setting
alternatives.

- **Push text:** edit `voice.txt` and push — the run speaks that text.
  Pushing `speak.py` or the workflow also triggers a run.
- **Manual:** Actions tab → "Generate voice (Kokoro, CPU)" →
  Run workflow → type any text, voice, and speed.

Each run writes `voice/audio.wav` (24 kHz mono WAV), `voice/TIMING.md`
(download/load/synth time, duration, peak RAM), and `speak.log` (full
console log, committed even on failure). Long text is split into
sentences and joined with short pauses.

## Swapping models

Change `DEFAULT_MODEL` in `generate.py`. Turbo models automatically
resolve to 4 steps / guidance 0.0 with their built-in scheduler;
standard models resolve to 25 steps / guidance 7.5 with DPM++ Karras.
XL models resolve to 768px + bfloat16; SD 1.5 models to 512px +
float32. Tried before: `stabilityai/sd-turbo` (fast, 28–42s at 512px,
but prone to duplicated figures at 768px); `Lykon/dreamshaper-8`
(clean single figure at 512px in 247.6s, but Riyan judged the quality
not good enough). Verified open, not yet tried: `stabilityai/sdxl-turbo`
and `Lykon/dreamshaper-xl-v2-turbo` (fast XL options),
`segmind/SSD-1B` (distilled SDXL).

## Specs

- Runner: `ubuntu-latest` — 4 vCPU, 16 GB RAM, 14 GB disk on public repos
- Weights: fp16 files cast to float32 (half the download, same CPU speed)
- No GPU. CPU-only, a few minutes per image. Keep volume low.
