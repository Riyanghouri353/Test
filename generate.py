#!/usr/bin/env python3
"""Generate one image with a Stable Diffusion model on CPU.

Model is chosen by --model (default: RunDiffusion/juggernaut-xl-v9).
Steps and guidance resolve per model family: Turbo models use 4 steps
/ guidance 0.0 with their built-in scheduler; everything else uses 25
steps / guidance 7.5 with an explicitly built DPM++ 2M Karras
scheduler. Size resolves per family too: 768px for XL models, 512px
otherwise. Dtype resolves per family: XL weights are ~6.5 GB in fp16
and would need ~15 GB cast to float32, which does not fit the 16 GB
runner next to activations — so XL models run in bfloat16 (half the
memory, fine quality), while SD 1.5 models keep float32. Prompt
source, in order: --prompt argument, prompt.txt in the repo root,
built-in default. Prints a TIMING line, saves the image to output/,
and writes output/TIMING.md with the same numbers.
"""
import argparse
import json
import os
import resource
import time

DEFAULT_MODEL = "RunDiffusion/juggernaut-xl-v9"
DEFAULT_PROMPT = "a cozy bakery storefront at sunrise, warm light, photorealistic"
DEFAULT_NEGATIVE = "ugly, deformed, blurry, low quality, duplicate, two people, extra limbs, bad anatomy, watermark, text"


def resolve_prompt(explicit):
    if explicit:
        return explicit
    try:
        with open("prompt.txt") as f:
            text = f.read().strip()
            if text:
                return text
    except FileNotFoundError:
        pass
    return DEFAULT_PROMPT


def is_turbo(model_id):
    return "turbo" in model_id.lower()


def is_xl(model_id):
    mid = model_id.lower()
    return "xl" in mid or "ssd-1b" in mid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    ap.add_argument("--out", default="output/image.png")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "float32", "bfloat16", "float16"])
    args = ap.parse_args()
    prompt = resolve_prompt(args.prompt)
    turbo = is_turbo(args.model)
    xl = is_xl(args.model)
    steps = args.steps if args.steps is not None else (4 if turbo else 25)
    guidance = args.guidance if args.guidance is not None else (0.0 if turbo else 7.5)
    size = args.size if args.size is not None else (768 if xl else 512)

    import torch
    from diffusers import AutoPipelineForText2Image

    if args.dtype == "auto":
        dtype = torch.bfloat16 if xl else torch.float32
    else:
        dtype = getattr(torch, args.dtype)

    # fp16 weight files, cast to the compute dtype: half the download
    # and disk of fp32 weights. XL in bfloat16 keeps weights ~7.4 GB
    # so the run fits the 16 GB runner; SD 1.5 keeps float32.
    t0 = time.time()
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model, torch_dtype=dtype, variant="fp16"
    ).to("cpu")
    if xl:
        pipe.enable_attention_slicing()
        # VAE slicing lives on the VAE itself, not the XL pipeline
        # (pipe.enable_vae_slicing() raises AttributeError on
        # StableDiffusionXLPipeline in current diffusers).
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
    if not turbo:
        from diffusers import DPMSolverMultistepScheduler

        # Build the scheduler explicitly: inheriting the model's own
        # scheduler config via from_config can carry an incompatible
        # algorithm_type (DreamShaper ships DEIS; DPM++ rejects
        # final_sigmas_type="zero" with algorithm_type="deis" and the
        # run dies before generating anything).
        pipe.scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            solver_order=2,
            prediction_type="epsilon",
            thresholding=False,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            lower_order_final=True,
            use_karras_sigmas=True,
        )
    t_load = time.time() - t0

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    t1 = time.time()
    image = pipe(
        prompt=prompt,
        negative_prompt=args.negative,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=size,
        width=size,
        generator=generator,
    ).images[0]
    t_gen = time.time() - t1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    image.save(args.out)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    report = {
        "model": args.model,
        "prompt": prompt,
        "negative_prompt": args.negative,
        "steps": steps,
        "guidance": guidance,
        "size": size,
        "seed": args.seed,
        "dtype": str(dtype).replace("torch.", ""),
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
