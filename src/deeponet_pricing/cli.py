"""CLI entry point: ``deeponet-pricing train configs/heston_ivol.yaml``."""

from __future__ import annotations

import argparse
import logging
import sys

from deeponet_pricing.config import ExperimentConfig

logger = logging.getLogger(__name__)


def cmd_train(args: argparse.Namespace) -> None:
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.iterations:
        cfg.training.iterations = args.iterations
    if args.lr:
        cfg.training.lr = args.lr

    from deeponet_pricing.training import Trainer

    trainer = Trainer(cfg)
    trainer.train()


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = ExperimentConfig.from_yaml(args.config)
    model_path = args.model

    from deeponet_pricing.training import Trainer

    trainer = Trainer(cfg)
    trainer.load_data()

    if cfg.model == "heston":
        from deeponet_pricing.models.heston import HestonIVolSolver
        from deeponet_pricing.evaluation.heston import evaluate_heston

        solver = HestonIVolSolver()
        solver.load(
            model_path,
            branch_layers=cfg.network.branch_layers,
            trunk_layers=cfg.network.trunk_layers,
        )
        evaluate_heston(
            solver, trainer.branch, trainer.trunk, trainer.targets,
            do_verify=args.verify_prices,
            output_dir=cfg.output.output_dir,
            do_plot=args.plot,
        )

    elif cfg.model == "rbergomi":
        from deeponet_pricing.models.rbergomi import RBergomiSolver
        from deeponet_pricing.evaluation.rbergomi import evaluate_rbergomi

        solver = RBergomiSolver()
        solver.load(
            model_path,
            branch_layers=cfg.network.branch_layers,
            trunk_layers=cfg.network.trunk_layers,
        )
        evaluate_rbergomi(
            solver, trainer.branch, trainer.trunk, trainer.targets,
            output_dir=cfg.output.output_dir,
            do_plot=args.plot,
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    parser = argparse.ArgumentParser(prog="deeponet-pricing")
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train a DeepONet model")
    p_train.add_argument("config", help="YAML config file")
    p_train.add_argument("--iterations", type=int, default=None)
    p_train.add_argument("--lr", type=float, default=None)
    p_train.set_defaults(func=cmd_train)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate a saved model")
    p_eval.add_argument("config", help="YAML config file")
    p_eval.add_argument("--model", required=True, help="Path to saved model (base name)")
    p_eval.add_argument("--plot", action="store_true")
    p_eval.add_argument("--verify-prices", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
