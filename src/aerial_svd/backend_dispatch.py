"""Select the optional JAX runtime before legacy entry points import Torch."""
import argparse
import sys


def dispatch(command):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", choices=("torch", "jax"), default="torch")
    args, remaining = parser.parse_known_args()
    if args.backend == "jax":
        from aerial_svd.jax_cli import main
        main(command, remaining)
        raise SystemExit(0)
    sys.argv[1:] = remaining
