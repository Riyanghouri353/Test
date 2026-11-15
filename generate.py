#!/usr/bin/env python3
"""Generate one image with SD-Turbo on CPU.

Prints a TIMING line with model-load and generation seconds, saves the
image to output/, and writes output/TIMING.md with the same numbers.
"""
import argparse
import json
import os
import resource
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a cozy bakery storefront at sunrise, warm light, photorealistic")
    ap.add_argument("--out", default="output/image.png")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from diffusers import AutoPipelineForText2Image

    # fp16 weights (2.6 GB download) cast to float32: same CPU speed as
    # full fp32 weights, half the disk. The 14 GB runner disk stays safe.
    t0 = time.time()
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo", torch_dtype=torch.float32, variant="fp16"
    ).to("cpu")
    t_load = time.time() - t0

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    t1 = time.time()
    image = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        height=args.size,
        width=args.size,
        generator=generator,
    ).images[0]
    t_gen = time.time() - t1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    image.save(args.out)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    report = {
        "model": "stabilityai/sd-turbo",
        "prompt": args.prompt,
        "steps": args.steps,
        "size": args.size,
        "seed": args.seed,
        "model_load_seconds": round(t_load, 1),
        "generate_seconds": round(t_gen, 1),
        "total_seconds": round(t_load + t_gen, 1),
        "peak_ram_mb": round(peak_mb, 1),
        "cpus": os.cpu_count(),
        "output": args.out,
        "output_bytes": os.path.getsize(args.out),
    }
    print("TIMING " + json.dumps(report))
    with open(os.path.join(os.path.dirname(args.out), "TIMING.md"), "w") as f:
        f.write("# Image generation timing\n\n")
        for k, v in report.items():
            f.write(f"- **{k}**: {v}\n")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
