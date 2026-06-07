"""Extend Schwedes Table 2.2 beyond ``h_max / h_min = 128``.

Same setup as ``table_2_2.py`` but with larger target ratios, to
confirm that the L^2 and H^1 rows stay flat while the l^2 row keeps
growing.
"""

from argparse import ArgumentParser

from meshdep.meshes import graded_unit_square
from meshdep.problems.poisson_control import solve_poisson_control


ROWS = [
    ("tao_lmvm", "l2"),
    ("tao_lmvm", "L2"),
    ("tao_lmvm", "H1"),
]


def run_extended(ratios, alpha=1.0e-4):
    """Run TAO/LMVM at the three Riesz maps on each ratio."""

    results = {}
    for ratio in ratios:
        print(f"Building mesh with target h_max/h_min = {ratio}...",
              flush=True)
        mesh, realised = graded_unit_square(h_ratio=float(ratio))
        print(f"  realised ratio = {realised:.2f}")
        label = str(ratio)

        for optimiser_name, riesz_map in ROWS:
            print(f"  {label}, {optimiser_name}, {riesz_map}...",
                  flush=True)
            result = solve_poisson_control(mesh, alpha=alpha,
                                           riesz_map=riesz_map)
            results[(label, optimiser_name, riesz_map)] = result["iterations"]
            print(
                f"    iters={result['iterations']:5d}  "
                f"J={result['final_J']:.3e}",
                flush=True,
            )
    return results


def print_console_table(ratios, results):
    """Print the iteration grid."""

    print()
    print(f"{'inner product':>14}  {'implementation':>16}  "
          + "  ".join(f"{r:>5}" for r in ratios))
    print("-" * (32 + 7 * len(ratios)))
    for optimiser_name, riesz_map in ROWS:
        cells = []
        for ratio in ratios:
            value = results.get((str(ratio), optimiser_name, riesz_map))
            cells.append(f"{value:>5}" if value is not None else "  -- ")
        print(f"{riesz_map:>14}  {optimiser_name:>16}  "
              + "  ".join(cells))


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ratios", type=int, nargs="+",
        default=[4, 8, 16, 32, 64, 128, 256, 512],
        help="Target h_max / h_min ratios.",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0e-4,
        help="Tikhonov regularisation parameter (default: 1e-4).",
    )
    args = parser.parse_args()

    results = run_extended(args.ratios, alpha=args.alpha)
    print_console_table(args.ratios, results)


if __name__ == "__main__":
    main()
