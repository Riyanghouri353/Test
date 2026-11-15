#!/usr/bin/env python3
"""Generate speech with Kokoro-82M on CPU (no API key, no quota).

Kokoro-82M via kokoro-onnx: an 82M-parameter open-weight model
(Apache 2.0) that runs on CPU in the runner itself. Text source, in
order: --text argument, voice.txt in the repo root, built-in default.
The model (~92 MB int8 ONNX) + voices file (~28 MB) download once
into ./kokoro-model from the kokoro-onnx GitHub release; no system
packages needed (the phonemizer is bundled via espeakng-loader).
Long text is split into sentences, synthesized per sentence at the
requested speed, and joined with short silence gaps. Prints a
VOICE_TIMING line, saves 24 kHz mono WAV to voice/, and writes
voice/TIMING.md with the same numbers.
"""
import argparse
import json
import os
import re
import resource
import time
import urllib.request

MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.int8.onnx"
)
VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/voices-v1.0.bin"
)
MODEL_FILE = "kokoro-v1.0.int8.onnx"
VOICES_FILE = "voices-v1.0.bin"
MODEL_MIN_BYTES = 50_000_000
VOICES_MIN_BYTES = 10_000_000
LANG = "en-us"
SAMPLE_RATE = 24000
DEFAULT_VOICE = "am_adam"
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
        with open("voice.txt") as f:
            text = f.read().strip()
            if text:
                return text
    except FileNotFoundError:
        pass
    return DEFAULT_TEXT


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def download(url, dest, min_bytes):
    """Download with retries; atomic rename so a partial file is never
    mistaken for a good model on the next run."""
    tmp = dest + ".part"
    last_error = None
    for attempt, delay in enumerate((2.0, 8.0, 30.0)):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            size = os.path.getsize(tmp)
            if size < min_bytes:
                raise IOError(f"downloaded {size} bytes, expected at least {min_bytes}")
            os.replace(tmp, dest)
            return
        except Exception as exc:
            last_error = exc
            print(f"Download attempt {attempt + 1} failed: {exc}")
            try:
                os.remove(tmp)
            except OSError:
                pass
            if attempt < 2:
                time.sleep(delay)
    raise IOError(f"Failed to download {url}: {last_error}")


def ensure_model_files(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, MODEL_FILE)
    voices_path = os.path.join(model_dir, VOICES_FILE)
    if not os.path.exists(model_path) or os.path.getsize(model_path) < MODEL_MIN_BYTES:
        print("Downloading Kokoro model (~92 MB)...")
        download(MODEL_URL, model_path, MODEL_MIN_BYTES)
    if not os.path.exists(voices_path) or os.path.getsize(voices_path) < VOICES_MIN_BYTES:
        print("Downloading Kokoro voices (~28 MB)...")
        download(VOICES_URL, voices_path, VOICES_MIN_BYTES)
    return model_path, voices_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None)
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", default="voice/audio.wav")
    ap.add_argument("--model-dir", default="kokoro-model")
    args = ap.parse_args()
    text = resolve_text(args.text)
    if not 0.5 <= args.speed <= 2.0:
        raise SystemExit(f"Speed {args.speed} outside 0.5-2.0.")
    sentences = split_sentences(text)
    if not sentences:
        raise SystemExit("No text to speak.")

    import numpy as np
    import soundfile as sf
    from kokoro_onnx import Kokoro

    t0 = time.time()
    model_path, voices_path = ensure_model_files(args.model_dir)
    t_download = time.time() - t0
    session = Kokoro(model_path, voices_path)
    t_load = time.time() - t0

    t1 = time.time()
    parts = []
    for i, sentence in enumerate(sentences):
        samples, sr = session.create(sentence, voice=args.voice, speed=args.speed, lang=LANG)
        if sr != SAMPLE_RATE:
            raise SystemExit(f"Kokoro returned sample rate {sr}, expected {SAMPLE_RATE}.")
        samples = np.asarray(samples, dtype=np.float32)
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
        "model": "kokoro-82m-int8 (kokoro-onnx)",
        "voice": args.voice,
        "speed": args.speed,
        "text": text,
        "words": len(text.split()),
        "sentences": len(sentences),
        "duration_seconds": round(len(combined) / SAMPLE_RATE, 2),
        "sample_rate": SAMPLE_RATE,
        "download_seconds": round(t_download, 1),
        "model_load_seconds": round(t_load, 1),
        "synth_seconds": round(t_synth, 1),
        "total_seconds": round(t_load + t_synth, 1),
        "peak_ram_mb": round(peak_mb, 1),
        "cpus": os.cpu_count(),
        "output": args.out,
        "output_bytes": os.path.getsize(args.out),
    }
    print("VOICE_TIMING " + json.dumps(report))
    with open(os.path.join(os.path.dirname(args.out), "TIMING.md"), "w") as f:
        f.write("# Voice generation timing\n\n")
        for k, v in report.items():
            f.write(f"- **{k}**: {v}\n")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
