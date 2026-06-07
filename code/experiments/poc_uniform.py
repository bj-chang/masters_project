"""Proof-of-concept run on uniform meshes.

Lighter version of ``table_2_2.py`` that uses uniform meshes and so
needs no Netgen. The l^2 row grows with N; the L^2 and H^1 rows stay
roughly flat.
"""

from argparse import ArgumentParser

from meshdep.meshes import uniform_unit_square
from meshdep.problems.poisson_control import solve_poisson_control


def run_poc_uniform(resolutions, riesz_maps):
    """Run the experiment. Returns ``{(label, "tao_lmvm", riesz_map): iters}``."""

    results = {}
    for n in resolutions:
        mesh = uniform_unit_square(n)
        for riesz_map in riesz_maps:
            print(f"  N={n}, riesz_map={riesz_map}...", flush=True)
            result = solve_poisson_control(mesh, riesz_map=riesz_map, verbose=True)
            results[(f"N={n}", "tao_lmvm", riesz_map)] = result["iterations"]
            print(
                f"    iters={result['iterations']:4d}  "
                f"J={result['final_J']:.3e}  "
                f"converged={result['converged']}",
                flush=True,
            )
    return results


def print_console_table(resolutions, riesz_maps, results):
    """Print the iteration grid."""

    print(f"\n{'inner product':>14}  {'':10}  "
          + "  ".join(f"{n:>5}" for n in resolutions))
    print("-" * (28 + 7 * len(resolutions)))
    for riesz_map in riesz_maps:
        cells = []
        for n in resolutions:
            value = results.get((f"N={n}", "tao_lmvm", riesz_map))
            cells.append(f"{value:>5}" if value is not None else "  -- ")
        print(f"{riesz_map:>14}  {'TAO (LMVM)':10}  " + "  ".join(cells))


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--resolutions", type=int, nargs="+",
        default=[8, 16, 32, 64, 128],
        help="Mesh resolutions to test.",
    )
    parser.add_argument(
        "--riesz-maps", nargs="+", default=["l2", "L2", "H1"],
        choices=["l2", "L2", "H1"],
        help="Inner products on the control space to test.",
    )
    args = parser.parse_args()

    results = run_poc_uniform(args.resolutions, args.riesz_maps)
    print_console_table(args.resolutions, args.riesz_maps, results)


if __name__ == "__main__":
    main()