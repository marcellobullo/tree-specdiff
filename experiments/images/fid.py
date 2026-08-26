"""Score saved samples: FID, and optionally Inception Score.

Scoring is separate from generation so a 50,000-sample run can be measured
against multiple reference sets or additional metrics. The script reads the
``samples.pt`` written by ``run_edm.py`` (uint8 ``(N, C, H, W)``),
and scores several runs in one invocation against one shared real set:

    python experiments/images/fid.py --samples results/edm/*/ \\
        --dataset cifar10 --num-real 50000 --device cuda:0 \\
        --output results/edm/fid_report.json

FID is computed with `torchmetrics`' `FrechetInceptionDistance(feature=2048,
normalize=False)` -- the same implementation the reference implementation uses,
so results are comparable with existing measurements from that implementation.
FID values from different Inception implementations should not be compared.

The real-side statistics are cached. FID depends on the real images only
through three accumulators -- `sum(f)`, `sum(f f^T)` and the count -- so
caching them preserves the metric exactly. InceptionV3 therefore processes the
real set once per cache. The cache filename includes the dataset, real count, and
resolution, because FID against a different real N or size is a different
number and require distinct cache entries.

`--dataset cifar10` fetches the real images through Hugging Face `datasets`.
`--dataset ffhq` needs `--data <directory or zip of images>`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Iterator

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_IMAGE_SIZE = {"cifar10": 32, "ffhq": 64}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

# The three accumulators torchmetrics keeps per side. FID is a function of
# these alone, which is why caching the real ones reproduces the uncached
# number exactly.
_REAL_STATE = ("real_features_sum", "real_features_cov_sum", "real_features_num_samples")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--samples", required=True, nargs="+",
                   help="run directories (containing samples.pt) or .pt files")
    p.add_argument("--dataset", default="cifar10", choices=tuple(DEFAULT_IMAGE_SIZE))
    p.add_argument("--data", default=None,
                   help="ffhq: directory or zip of the real images")
    p.add_argument("--num-real", type=int, default=50_000,
                   help="real images the statistics cover. FID against a "
                        "different value is a different number")
    p.add_argument("--image-size", type=int, default=None,
                   help="defaults to 32 (cifar10) / 64 (ffhq); samples of "
                        "another resolution are refused rather than scored")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default=None,
                   help="where the real statistics live (default: the first "
                        "--samples directory)")
    p.add_argument("--inception-score", action="store_true")
    p.add_argument("--is-splits", type=int, default=10)
    p.add_argument("--output", default=None, help="write the report as JSON")
    return p.parse_args(argv)


# ------------------------------------------------------------------ real images
def cifar10_batches(batch_size: int, limit: int) -> Iterator[torch.Tensor]:
    """The CIFAR-10 train split as uint8 `(B, 3, 32, 32)`."""
    try:
        from datasets import load_dataset
    except ImportError as exc:                                    # noqa: BLE001
        raise SystemExit(
            "--dataset cifar10 needs the `datasets` package: pip install datasets"
        ) from exc
    import numpy as np

    ds = load_dataset("uoft-cs/cifar10", split="train")
    ds = ds.with_format("np", columns=["img"], output_all_columns=False)
    n = min(limit, len(ds))
    for i in range(0, n, batch_size):
        chunk = ds[i : min(i + batch_size, n)]["img"]
        yield torch.from_numpy(np.stack(chunk)).permute(0, 3, 1, 2).contiguous()


def image_file_batches(path: Path, batch_size: int, limit: int, size: int):
    """A directory or zip of images as uint8 `(B, 3, size, size)`."""
    import numpy as np
    import PIL.Image

    def load(fh) -> torch.Tensor:
        img = PIL.Image.open(fh).convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), PIL.Image.BICUBIC)
        return torch.from_numpy(np.asarray(img)).permute(2, 0, 1)

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = sorted(n for n in zf.namelist()
                           if Path(n).suffix.lower() in IMAGE_SUFFIXES)[:limit]
            for i in range(0, len(names), batch_size):
                with_batch = []
                for name in names[i : i + batch_size]:
                    with zf.open(name) as fh:
                        with_batch.append(load(fh))
                yield torch.stack(with_batch)
    else:
        names = sorted(p for p in path.rglob("*")
                       if p.suffix.lower() in IMAGE_SUFFIXES)[:limit]
        if not names:
            raise SystemExit(f"no images found under {path}")
        for i in range(0, len(names), batch_size):
            yield torch.stack([load(p) for p in names[i : i + batch_size]])


def real_batches(args, size: int):
    if args.dataset == "cifar10":
        return cifar10_batches(args.batch_size, args.num_real)
    if not args.data:
        raise SystemExit(f"--dataset {args.dataset} needs --data (the real images)")
    return image_file_batches(Path(args.data), args.batch_size, args.num_real, size)


# ----------------------------------------------------------------------- samples
def load_samples(spec: str) -> torch.Tensor:
    """`samples.pt` from a run directory, or a `.pt` file directly."""
    path = Path(spec)
    if path.is_dir():
        path = path / "samples.pt"
    if not path.exists():
        raise SystemExit(f"no samples at {path}")
    samples = torch.load(path, map_location="cpu", weights_only=True)
    if samples.dtype != torch.uint8:
        raise SystemExit(f"{path}: expected uint8 samples, got {samples.dtype}")
    return samples


def real_cache_signature(args, size: int) -> dict:
    """Identify the real-image source whose feature accumulators are cached."""
    base = {"version": 1, "dataset": args.dataset,
            "num_real": args.num_real, "size": size}
    if args.dataset == "cifar10":
        base["source"] = {"dataset_id": "uoft-cs/cifar10", "split": "train"}
        return base
    if not args.data:
        raise SystemExit(f"--dataset {args.dataset} needs --data (the real images)")
    path = Path(args.data).resolve()
    if path.is_file():
        stat = path.stat()
        base["source"] = {"path": str(path), "size": stat.st_size,
                          "mtime_ns": stat.st_mtime_ns}
        return base
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    for item in files:
        stat = item.stat()
        digest.update(str(item.relative_to(path)).encode())
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    base["source"] = {"path": str(path), "files": len(files),
                      "manifest_sha256": digest.hexdigest()}
    return base


# --------------------------------------------------------------------------- fid
def fill_real(metric, args, size: int, cache: Path) -> int:
    """Populate the real side, from cache when possible."""
    signature = real_cache_signature(args, size)
    if cache.exists():
        state = torch.load(cache, map_location=args.device, weights_only=True)
        if state.get("_signature") == signature:
            for name in _REAL_STATE:
                setattr(metric, name, state[name].to(args.device))
            n = int(metric.real_features_num_samples.item())
            print(f"real stats: {n} images from cache {cache.name}")
            return n
        print(f"real stats: ignoring stale cache {cache.name}")

    print(f"real stats: featurising up to {args.num_real} {args.dataset} images "
          f"(cached afterwards at {cache.name})")
    seen = 0
    for batch in real_batches(args, size):
        if batch.shape[-1] != size:
            raise SystemExit(
                f"real images are {batch.shape[-1]}px but --image-size is {size}"
            )
        metric.update(batch.to(args.device), real=True)
        seen += batch.shape[0]
        if seen % (args.batch_size * 20) == 0:
            print(f"  {seen}/{args.num_real}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    state = {n: getattr(metric, n) for n in _REAL_STATE}
    state["_signature"] = signature
    torch.save(state, str(cache) + ".tmp")
    Path(str(cache) + ".tmp").replace(cache)
    return seen


def score(metric, samples: torch.Tensor, args) -> float:
    # `reset_real_features=False` at construction is what makes `reset()` clear
    # only the fake side, so the real statistics survive across runs -- the
    # whole point of scoring a sweep in one invocation.
    metric.reset()
    for i in range(0, samples.shape[0], args.batch_size):
        metric.update(samples[i : i + args.batch_size].to(args.device), real=False)
    return float(metric.compute().item())


def inception_score(samples: torch.Tensor, args) -> tuple[float, float]:
    from torchmetrics.image.inception import InceptionScore

    metric = InceptionScore(splits=args.is_splits, normalize=False).to(args.device)
    for i in range(0, samples.shape[0], args.batch_size):
        metric.update(samples[i : i + args.batch_size].to(args.device))
    mean, std = metric.compute()
    return float(mean.item()), float(std.item())


def main(argv=None) -> None:
    args = parse_args(argv)
    from torchmetrics.image.fid import FrechetInceptionDistance

    size = args.image_size or DEFAULT_IMAGE_SIZE[args.dataset]
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.samples[0])
    if cache_dir.is_file():
        cache_dir = cache_dir.parent
    # The count and size are in the name: FID against a different real N or
    # resolution is a different number and must not reuse this file.
    cache = cache_dir / f"fid_real_{args.dataset}_{args.num_real}_{size}px.pt"

    metric = FrechetInceptionDistance(
        feature=2048, normalize=False, reset_real_features=False
    ).to(args.device)
    num_real = fill_real(metric, args, size, cache)

    report = {"dataset": args.dataset, "num_real": num_real, "image_size": size,
              "runs": {}}
    for spec in args.samples:
        samples = load_samples(spec)
        if samples.shape[-1] != size:
            raise SystemExit(
                f"{spec}: samples are {samples.shape[-1]}px, real set is {size}px"
            )
        entry = {"num_samples": int(samples.shape[0]), "fid": score(metric, samples, args)}
        if args.inception_score:
            entry["is_mean"], entry["is_std"] = inception_score(samples, args)
        meta_path = Path(spec) / "meta.json" if Path(spec).is_dir() else None
        if meta_path is not None and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            entry.update({k: meta[k] for k in
                          ("rule", "branching", "lookahead", "speedup",
                           "end_to_end_speedup", "acceptance_rate")
                          if k in meta})
        report["runs"][str(spec)] = entry
        line = f"  {spec}: FID {entry['fid']:.3f}  n={entry['num_samples']}"
        if "speedup" in entry:
            line += f"  {entry['rule']} speedup {entry['speedup']:.2f}x"
        print(line, flush=True)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
