#!/usr/bin/env python3
"""Generate speech with Chatterbox (Resemble AI) on CPU — no API key.

Chatterbox is a ~1B-parameter open TTS model widely regarded as the
most natural-sounding open voice model, with emotion/exaggeration
control and optional voice cloning. It is far heavier than Kokoro:
~3.2 GB of weights, several GB of RAM, and minutes of CPU time for
seconds of audio — this script measures all of that. Text source, in
order: --text argument, voice-chatterbox.txt in the repo root,
built-in default (same text as the Kokoro demo, for a fair listen).
Long text is split into sentences, synthesized per sentence at the
requested exaggeration/cfg, and joined with short silence gaps.
Prints a CHATTERBOX_TIMING line, saves 24 kHz mono WAV to
voice-chatterbox/, and writes voice-chatterbox/TIMING.md.
"""
import argparse
import json
import os
import re
import resource
import time

SAMPLE_RATE = 24000
DEFAULT_TEXT = (
    "Hello, this is Adam. I am a natural sounding voice, running on a "
    "free computer in the cloud. I can read your scripts out loud, "
    "with the tone and pace of a real person speaking."
)
GAP_S = 0.25
FADE_SAMPLES = 120


def resolve_text(explicit):
    if explicit:
        return explicit
    try:
        with open("voice-chatterbox.txt") as f:
            text = f.read().strip()
            if text:
                return text
    except FileNotFoundError:
        pass
    return DEFAULT_TEXT


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None)
    ap.add_argument("--exaggeration", type=float, default=0.5)
    ap.add_argument("--cfg-weight", type=float, default=0.5)
    ap.add_argument("--out", default="voice-chatterbox/audio.wav")
    args = ap.parse_args()
    text = resolve_text(args.text)
    sentences = split_sentences(text)
    if not sentences:
        raise SystemExit("No text to speak.")

    import numpy as np
    import soundfile as sf
    import torch

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    from chatterbox.tts import ChatterboxTTS

    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device="cpu")
    t_load = time.time() - t0

    t1 = time.time()
    parts = []
    per_sentence = []
    for i, sentence in enumerate(sentences):
        ts = time.time()
        wav = model.generate(
            sentence,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
        )
        per_sentence.append(round(time.time() - ts, 1))
        samples = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
        fade = min(FADE_SAMPLES, len(samples) // 2)
        if fade > 0:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            samples[:fade] *= ramp
            samples[-fade:] *= ramp[::-1]
        parts.append(samples)
        if i < len(sentences) - 1:
            parts.append(np.zeros(int(round(GAP_S * SAMPLE_RATE)), dtype=np.float32))
    combined = np.concatenate(parts) if len(parts) > 1 else parts[0]
    t_synth = time.time() - t1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sf.write(args.out, combined, SAMPLE_RATE, subtype="PCM_16")
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    report = {
        "model": "chatterbox (ResembleAI, ~1B)",
        "voice": "default (no reference clip)",
        "exaggeration": args.exaggeration,
        "cfg_weight": args.cfg_weight,
        "text": text,
        "words": len(text.split()),
        "sentences": len(sentences),
        "per_sentence_seconds": per_sentence,
        "duration_seconds": round(len(combined) / SAMPLE_RATE, 2),
        "sample_rate": SAMPLE_RATE,
        "model_load_seconds": round(t_load, 1),
        "synth_seconds": round(t_synth, 1),
        "total_seconds": round(t_load + t_synth, 1),
        "peak_ram_mb": round(peak_mb, 1),
        "cpus": os.cpu_count(),
        "output": args.out,
        "output_bytes": os.path.getsize(args.out),
    }
    print("CHATTERBOX_TIMING " + json.dumps(report))
    with open(os.path.join(os.path.dirname(args.out), "TIMING.md"), "w") as f:
        f.write("# Chatterbox voice generation timing\n\n")
        for k, v in report.items():
            f.write(f"- **{k}**: {v}\n")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
