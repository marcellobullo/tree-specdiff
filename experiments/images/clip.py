"""CLIP score for saved SD3 samples. No generation.

FID is not the measurement for SD3: these are samples of a *text conditional*,
not of a dataset distribution, so there is no real set to be Frechet-distant
from. CLIPScore (Hessel et al. 2021) answers the question that does apply --
whether two rules that sample the same law at temperature 1 produce equally
prompt-faithful images:

    CLIPScore = max(100 * cos(E_image, E_text), 0)

    python experiments/images/clip.py --samples results/sd3/*/ \\
        --device cuda:0 --output results/sd3/clip_report.json

Per-image scores are the point
------------------------------
Every cell is scored image by image and the whole vector is kept, not just its
mean. When two cells were generated from the same (caption, seed) pairs -- which
`run_sd3.py --prompts` guarantees -- the **paired** difference has far lower
variance than the difference of two means, so a real gap shows up at sample
counts where the marginal means are indistinguishable. `--baseline` reports that
paired comparison directly; without per-image scores the pairing built into
generation would be thrown away at scoring time.

Captions come from each cell's `meta.json` (`prompts_file` -> line i for image
i, else `prompt` for all), so the scorer reconstructs exactly what each image
was generated from. `--prompts` overrides, for samples predating the field.

Writes `clip.json` beside each `samples.pt` and a combined `--output` report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_MODEL = "openai/clip-vit-large-patch14"
DTYPES = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--samples", required=True, nargs="+",
                   help="cell directories, each holding samples.pt + meta.json")
    p.add_argument("--baseline", default=None,
                   help="cell to compare the others against, paired per image. "
                        "Defaults to the first --samples entry when it looks "
                        "like a plain-target run")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="HF CLIP id; scores are not comparable across different "
                        "ones, so keep this fixed")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dtype", default="float32", choices=sorted(DTYPES),
                   help="float16 is ~2x faster and moves the score by <0.01")
    p.add_argument("--device", default=None,
                   help="default: cuda:0 when visible, else cpu")
    p.add_argument("--prompts", default=None,
                   help="override meta.json: a file (line i for image i) or a "
                        "literal caption for every image")
    p.add_argument("--overwrite", action="store_true",
                   help="rescore cells that already have clip.json")
    p.add_argument("--output", default=None, help="combined JSON report")
    return p.parse_args(argv)


def cell_prompts(cell: Path, num_images: int, override: Optional[str]) -> List[str]:
    """The caption per image, from `meta.json` unless overridden.

    Raises rather than guessing: a silently wrong caption-to-image alignment
    produces a plausible-looking score that means nothing.
    """
    source = override
    if source is None:
        meta = json.loads((cell / "meta.json").read_text())
        source = meta.get("prompts_file") or meta.get("prompt")
        if source is None:
            raise SystemExit(
                f"{cell}: meta.json has neither `prompts_file` nor `prompt`; "
                "pass --prompts"
            )
    path = Path(source)
    # A path that exists is a caption file; anything else is a literal caption.
    if not (len(str(source)) < 4096 and path.exists()):
        return [str(source)] * num_images

    lines = path.read_text().splitlines()
    prompts = [ln.strip() for ln in lines if ln.strip()]
    if len(prompts) != len(lines):
        raise SystemExit(f"{path}: blank lines break the line-to-image map")
    if len(prompts) < num_images:
        raise SystemExit(
            f"{path} has {len(prompts)} captions but {cell} holds {num_images} images"
        )
    # run_sd3.py assigns image i to line i, so a prefix is the mapping.
    return prompts[:num_images]


def projected(out) -> torch.Tensor:
    """The projected embedding, whatever the transformers version returns.

    <=4.x `get_*_features` returned the projected tensor directly; 5.x returns a
    `BaseModelOutputWithPooling` with the projection in `pooler_output`. Reading
    `.pooler_output` off a bare tensor would raise, and silently scoring an
    *unprojected* embedding would give a plausible but wrong number -- so branch
    on the type rather than pinning a version.
    """
    return out if isinstance(out, torch.Tensor) else out.pooler_output


class Scorer:
    """A CLIP image/text encoder pair, loaded once and reused across cells."""

    def __init__(self, model_id: str, device, dtype) -> None:
        from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast

        self.model = CLIPModel.from_pretrained(model_id).to(device, dtype).eval()
        self.image_processor = CLIPImageProcessor.from_pretrained(model_id)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
        self.device, self.dtype = device, dtype
        self._text_cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def text_features(self, prompts: List[str]) -> torch.Tensor:
        """Unit-norm text embeddings, memoised -- cells reuse the same captions.

        Across a twelve-cell sweep every caption is encoded once rather than
        twelve times, and within a cell a repeated caption costs nothing.
        """
        missing = [p for p in dict.fromkeys(prompts) if p not in self._text_cache]
        for i in range(0, len(missing), 256):
            chunk = missing[i : i + 256]
            tok = self.tokenizer(chunk, padding=True, truncation=True,
                                 max_length=77, return_tensors="pt").to(self.device)
            feats = projected(self.model.get_text_features(**tok)).to(torch.float32)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            for p, f in zip(chunk, feats):
                self._text_cache[p] = f
        return torch.stack([self._text_cache[p] for p in prompts])

    @torch.no_grad()
    def image_features(self, images: torch.Tensor) -> torch.Tensor:
        """uint8 ``(B, 3, H, W)`` -> unit-norm image embeddings.

        Goes through the HF processor's resize/crop/normalise, which is what
        published CLIP scores use; it wants channels-last uint8 arrays.
        """
        arrays = list(images.permute(0, 2, 3, 1).numpy())
        px = self.image_processor(images=arrays, return_tensors="pt")["pixel_values"]
        feats = projected(
            self.model.get_image_features(px.to(self.device, self.dtype))
        ).to(torch.float32)
        return feats / feats.norm(dim=-1, keepdim=True)

    def scores(self, images: torch.Tensor, prompts: List[str],
               batch_size: int) -> torch.Tensor:
        """Per-image ``max(100 cos, 0)``."""
        out = []
        for i in range(0, images.shape[0], batch_size):
            img = self.image_features(images[i : i + batch_size])
            txt = self.text_features(prompts[i : i + batch_size]).to(img.device)
            out.append(100.0 * (img * txt).sum(-1))
        return torch.cat(out).clamp(min=0.0).cpu() if out else torch.zeros(0)


def summarise(cell: Path, scores: torch.Tensor, model: str, override) -> dict:
    n = int(scores.numel())
    std = float(scores.std(unbiased=True)) if n > 1 else 0.0
    return {
        "cell": str(cell),
        "num_images": n,
        "clip_model": model,
        "clip_score_mean": float(scores.mean()),
        "clip_score_std": std,
        "clip_score_sem": std / (n**0.5) if n > 1 else 0.0,
        "prompt_source": override,
        "per_image": [round(v, 4) for v in scores.tolist()],
    }


def paired_delta(a: dict, b: dict) -> Optional[dict]:
    """``a - b`` per image, when the two cells are comparable.

    Only meaningful if both were generated from the same (caption, seed) pairs;
    equal image counts is the check available here, so the caller is trusted for
    the rest. The paired standard error is what makes a small real difference
    visible where the two marginal means overlap.
    """
    if a["num_images"] != b["num_images"] or a["num_images"] < 2:
        return None
    d = torch.tensor(a["per_image"]) - torch.tensor(b["per_image"])
    n = d.numel()
    sem = float(d.std(unbiased=True)) / (n**0.5)
    mean = float(d.mean())
    return {
        "vs": b["cell"],
        "paired_mean_delta": mean,
        "paired_sem": sem,
        # Rough two-sided z; at |z| < 2 the cells are indistinguishable, which
        # is the *expected* result -- they sample the same law.
        "z": mean / sem if sem > 0 else 0.0,
        "unpaired_sem": (a["clip_score_sem"] ** 2 + b["clip_score_sem"] ** 2) ** 0.5,
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    scorer = None                                   # loaded lazily: cached cells skip it

    results: Dict[str, dict] = {}
    for spec in args.samples:
        cell = Path(spec)
        cached = cell / "clip.json"
        if cached.exists() and not args.overwrite:
            results[str(cell)] = json.loads(cached.read_text())
            print(f"  {cell}: reusing clip.json "
                  f"(CLIP {results[str(cell)]['clip_score_mean']:.3f})", flush=True)
            continue

        images = torch.load(cell / "samples.pt", map_location="cpu", weights_only=True)
        if images.dtype != torch.uint8:
            raise SystemExit(f"{cell}: expected uint8 images, got {images.dtype}")
        prompts = cell_prompts(cell, int(images.shape[0]), args.prompts)
        if scorer is None:
            scorer = Scorer(args.model, device, DTYPES[args.dtype])

        entry = summarise(cell, scorer.scores(images, prompts, args.batch_size),
                          args.model, args.prompts)
        meta = json.loads((cell / "meta.json").read_text())
        entry.update({k: meta[k] for k in
                      ("rule", "branching", "lookahead", "guidance_scale",
                       "speedup", "end_to_end_speedup", "acceptance_rate")
                      if k in meta})
        cached.write_text(json.dumps(entry, indent=2) + "\n")
        results[str(cell)] = entry
        print(f"  {cell}: CLIP {entry['clip_score_mean']:.3f} +/- "
              f"{entry['clip_score_sem']:.3f} (n={entry['num_images']})"
              + (f"  {entry['rule']} speedup {entry['speedup']:.2f}x"
                 if "speedup" in entry else ""), flush=True)

    base_key = args.baseline or next(iter(results), None)
    if base_key is not None and len(results) > 1:
        base = results.get(str(Path(base_key)))
        if base is None:
            raise SystemExit(f"--baseline {base_key} is not among --samples")
        print(f"\npaired against {base['cell']}:")
        for key, entry in results.items():
            if key == base["cell"]:
                continue
            d = paired_delta(entry, base)
            if d is None:
                print(f"  {key}: not comparable (different image counts)")
                continue
            entry["paired"] = d
            print(f"  {key}: {d['paired_mean_delta']:+.4f} +/- {d['paired_sem']:.4f} "
                  f"(z {d['z']:+.2f}; unpaired sem would be {d['unpaired_sem']:.4f})")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"model": args.model, "runs": results}, f, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
