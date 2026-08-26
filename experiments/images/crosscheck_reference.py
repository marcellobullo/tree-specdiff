"""Compare this adapter with the reference implementation on identical inputs.

The local test suite validates the adapter's internal consistency. This script
adds a cross-implementation check for comparisons with previously published
results.

It runs both transition implementations side by side and reports their
difference. A separate `accelerating-diffusion-sampling` checkout is required.

    python experiments/images/crosscheck_reference.py \\
        --reference-repo /path/to/accelerating-diffusion-sampling

Expected: velocity and transition standard deviation agree exactly; the sigma
grid and kernel mean agree within float32 precision (approximately `1e-7`).
Larger differences indicate that results from the two implementations are not
directly comparable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from images import models  # noqa: E402
from images.toy import GaussianEDMPrecond  # noqa: E402

TOL = 1e-6


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference-repo", required=True,
                   help="checkout of accelerating-diffusion-sampling")
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--eps", type=float, default=0.25)
    args = p.parse_args(argv)

    sys.path.insert(0, args.reference_repo)
    try:
        from src.models.edm import EDMPixelModel
        from src.schedulers import ChurnFlowMatchEulerScheduler
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit(
            f"cannot import the reference implementation from "
            f"{args.reference_repo}: {exc}"
        )

    T, eps = args.num_steps, args.eps
    # The same network object drives both sides, so any disagreement is in the
    # step math and not in the weights.
    net = GaussianEDMPrecond(img_resolution=8, img_channels=3)
    ref_sched = ChurnFlowMatchEulerScheduler()
    ref_sched.eps = eps
    ref_sched.set_timesteps(T)
    ref_model = EDMPixelModel(net, device="cpu")
    mine = models.EDMDenoiser(net)

    gen = torch.Generator().manual_seed(11)
    x = torch.randn((6, 3, 8, 8), generator=gen)
    grid = models.sigma_grid(T, shift=float(ref_sched.config.shift))
    steps = torch.tensor([1, 3, 7, 11, 15, 18])

    results = {}
    results["sigma grid"] = (grid - ref_sched.sigmas.double()).abs().max().item()

    t = torch.tensor([0.02, 0.1, 0.3, 0.5, 0.8, 0.95])
    ntt = ref_model._scheduler_config["num_train_timesteps"]
    results["velocity"] = (
        ref_model.velocity(x, t * ntt) - mine.velocity(x, t)
    ).abs().max().item()

    # The reference takes the velocity as an argument where this adapter
    # computes it internally, so it must be evaluated at each row's OWN step
    # sigma or the two are not comparable.
    v_at = mine.velocity(x, grid[steps].float())
    m_ref, s_ref = ref_sched.kernel_params(x, v_at, steps, eps=eps)
    target = models.ChurnKernelTarget(mine, grid, eps)
    m_mine, s_mine = target.kernel(x, tuple(int(i) for i in steps))
    results["kernel mean"] = (m_ref - m_mine).abs().max().item()
    results["kernel std"] = (s_ref - s_mine).abs().max().item()

    zeros = torch.zeros((T, 3, 8, 8))
    v_all = mine.velocity(zeros, grid[:-1].float())
    _, s_all_ref = ref_sched.kernel_params(zeros, v_all, torch.arange(T), eps=eps)
    s_all_mine = models.churn_std_grid(grid, eps).float()
    results["std grid"] = (s_all_ref - s_all_mine).abs().max().item()

    worst = 0.0
    for name, diff in results.items():
        print(f"  {name:<14} max|diff| = {diff:.3e}")
        worst = max(worst, diff)

    ref_zero = torch.nonzero(s_all_ref == 0).flatten().tolist()
    mine_zero = torch.nonzero(s_all_mine == 0).flatten().tolist()
    print(f"  {'std zeros':<14} reference {ref_zero}, adapter {mine_zero}")
    if ref_zero != mine_zero:
        print("FAIL: the two disagree on which steps are deterministic")
        return 1
    if worst > TOL:
        print(f"FAIL: worst disagreement {worst:.3e} exceeds {TOL:.0e}")
        return 1
    print(f"OK: agreement within {TOL:.0e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
