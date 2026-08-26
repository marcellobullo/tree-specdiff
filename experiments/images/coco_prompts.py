"""Build a deterministic COCO caption set for text-to-image sampling.

Turns `captions_val2014.json` (from the official `annotations_trainval2014.zip`)
into the flat one-caption-per-line file that `run_sd3.py --prompts` consumes.

    python experiments/images/coco_prompts.py \\
        --annotations /path/captions_val2014.json \\
        --num 30000 --output prompts/coco30k

Writes, next to `--output`:
    coco30k.txt             the captions, one per line, in selection order
    coco30k.image_ids.txt   the COCO image_id per line, aligned to the captions
    coco30k.meta.json       the exact procedure, so the set can be re-derived

Prefix-stable selection
-----------------------
The published zero-shot number for text-to-image models is FID-30K on COCO 2014
validation, so the full set contains 30,000 captions. Smaller sweeps use the
same ordered caption set to remain comparable with full runs.

The order depends only on `(--seed, --num-pool)`, never on `--num`:
`--num 500` produces exactly the first 500 lines of the 30,000-caption file.
Increasing `--num` therefore extends an existing sweep without changing its
previous caption assignments.

One caption per image (COCO has ~5), chosen as the lowest annotation id, which
is the file's own first caption for that image -- deterministic, no RNG. Images
are then ordered by a seeded shuffle of the *sorted* image_id list, because dict
order reflects file order and the set should not depend on that.

Captions are whitespace-normalised: some contain literal newlines and many have
stray leading or trailing space, and one-caption-per-line is only unambiguous if
no caption can contain a line break.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import List, Tuple


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--annotations", required=True,
                   help="captions_val2014.json (or captions_train2014.json)")
    p.add_argument("--output", required=True,
                   help="path prefix; .txt / .image_ids.txt / .meta.json are appended")
    p.add_argument("--num", type=int, default=30_000,
                   help="captions to write; a prefix of the shuffled pool")
    p.add_argument("--num-pool", type=int, default=0,
                   help="shuffle pool size, 0 = every captioned image. Changing "
                        "this changes the order -- leave it alone to keep --num "
                        "a true prefix")
    p.add_argument("--seed", type=int, default=20260727,
                   help="shuffle seed; part of the set's identity")
    return p.parse_args(argv)


def normalise(caption: str) -> str:
    """Collapse every whitespace run to one space and strip. Keeps lines atomic."""
    return " ".join(caption.split())


def select(path: str, num: int, num_pool: int, seed: int) -> Tuple[List[str], List[int]]:
    with open(path) as f:
        data = json.load(f)

    # Lowest annotation id = the file's first caption for that image.
    first = {}
    for ann in data["annotations"]:
        img, aid = ann["image_id"], ann["id"]
        if img not in first or aid < first[img][0]:
            first[img] = (aid, ann["caption"])

    # Sort before shuffling: dict order reflects file order, which is not a
    # property the caption set should depend on.
    image_ids = sorted(first)
    if num_pool:
        image_ids = image_ids[:num_pool]
    random.Random(seed).shuffle(image_ids)

    if num > len(image_ids):
        raise SystemExit(
            f"--num {num} exceeds the {len(image_ids)} captioned images available"
        )
    chosen = image_ids[:num]
    return [normalise(first[i][1]) for i in chosen], chosen


def main(argv=None) -> None:
    args = parse_args(argv)
    captions, image_ids = select(args.annotations, args.num, args.num_pool, args.seed)
    if any(not c for c in captions):
        raise SystemExit("a caption normalised to the empty string; refusing to write")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".txt").write_text("\n".join(captions) + "\n")
    out.with_suffix(".image_ids.txt").write_text(
        "\n".join(str(i) for i in image_ids) + "\n"
    )

    src = Path(args.annotations)
    with open(src, "rb") as f:                     # identity of the source file
        digest = hashlib.sha256(f.read()).hexdigest()
    out.with_suffix(".meta.json").write_text(json.dumps({
        "source": str(src.resolve()),
        "source_sha256": digest,
        "num": args.num,
        "num_pool": args.num_pool,
        "seed": args.seed,
        "caption_rule": "lowest annotation id per image",
        "order": "seeded shuffle of ascending image_id",
        "normalisation": "whitespace runs collapsed to single spaces, stripped",
    }, indent=2) + "\n")

    print(f"{len(captions)} prompts -> {out.with_suffix('.txt')}")
    for c in captions[:3]:
        print(f"  {c}")


if __name__ == "__main__":
    main()
