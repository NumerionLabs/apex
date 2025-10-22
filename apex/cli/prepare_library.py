#!/usr/bin/env python
"""
This script prepares a CSL for accelerated approximate inference with a trained
factorizer, encoder, and probe triplet.

```
prepare_library \
--config configs/apex.yaml \
--library_path $CSL_PATH \
--encoder_weights_path $ENCODER_WEIGHTS_PATH \
--probe_weights_path $PROBE_WEIGHTS_PATH \
--factorizer_weights_path $FACTORIZER_WEIGHTS_PATH \
--output_dir $OUTPUT_DIR \
--run_id $RUN_ID
```

Outputs are saved as follows:

```
$OUTPUT_DIR/
└── $RUN_ID/
    ├── config.yaml               (Provided config)
    ├── logs.log                  (Log statements)
    └── checkpoints/
        ├── encoder.pt            (Provided encoder weights)
        ├── probe.pt              (Provided probe weights)
        ├── factorizer.pt         (Provided factorizer weights)
        └── apex.pt               (APEX weights, specific to the provided CSL)
```
"""

# Standard
import argparse
from datetime import datetime
import logging
import os
import shutil
from typing import Optional
import uuid

# Third party
import pandas as pd
import torch
import yaml

# APEX
from apex.data.atom_bond_features import Vocabulary
from apex.dataset.csl_dataset import CSLDataset
from apex.nn.apex import APEXFactorizedCSL, APEXFactorizer
from apex.nn.encoder import LigandEncoder
from apex.nn.probe import LinearProbe
from apex.utils.other_utils import init_logging, makedirs


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "This script prepares a CSL on a trained encoder, probe, and "
            "factorizer triplet for running APEX accelerated search."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        action="store",
        required=True,
        help="Path to YAML file specifying the configuration.",
    )
    parser.add_argument(
        "--library_path",
        type=str,
        action="store",
        required=True,
        help="Path to directory containing the CSL, comprised of a pair of"
        "reactions.parquet and synthons.parquet files.",
    )
    parser.add_argument(
        "--encoder_weights_path",
        type=str,
        action="store",
        required=True,
        help="Path to weights for encoder.",
    )
    parser.add_argument(
        "--probe_weights_path",
        type=str,
        action="store",
        required=True,
        help="Path to weights for probe.",
    )
    parser.add_argument(
        "--factorizer_weights_path",
        type=str,
        action="store",
        required=True,
        help="Path to weights for factorizer.",
    )
    parser.add_argument(
        "--load_synthons_and_reactions",
        type=bool,
        action="store",
        required=False,
        default=False,
        help="Whether to load the CSL's synthon SMILES and reaction SMARTS.",
    )
    parser.add_argument(
        "--fast_smiles",
        type=bool,
        action="store",
        required=False,
        default=True,
        help="Whether to construct SMILES via string manipulation or RDKit.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        action="store",
        required=False,
        default=None,
        help="Path to output directory. If not provided, defaults to current"
        "working directory.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        action="store",
        required=False,
        default=None,
        help="Run ID, used in constructing output directories and in logs to"
        "tensorboard. If not given, a random sixteen character alpha-numeric "
        "run_id will be generated and used.",
    )
    return parser.parse_args()


def run(
    config: str,
    library_path: str,
    encoder_weights_path: str,
    probe_weights_path: str,
    factorizer_weights_path: str,
    load_synthons_and_reactions: bool = True,
    fast_smiles: bool = True,
    run_id: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    # Start-up
    start_time = datetime.now()
    run_id = str(uuid.uuid4().hex[:16] if run_id is None else run_id)
    output_dir = str(os.getcwd() if output_dir is None else output_dir)
    outdir = os.path.join(output_dir, run_id)
    makedirs(os.path.join(outdir, "checkpoints"))
    shutil.copy(config, os.path.join(outdir, "config.yaml"))

    # Set up logging
    init_logging(filename=os.path.join(outdir, "logs.log"))
    logging.info(f"Run ID: {run_id}.")
    logging.info(f"All outputs will be written to: {outdir}.")
    logging.info(
        f"Number of GPUs: {torch.cuda.device_count()}. "
        f"Number of CPUs: {os.cpu_count()}."
    )

    # Parse the config
    with open(config, "r") as fp:
        config_all = yaml.safe_load(fp)

    # Register vocab
    config_vocab = config_all.get("vocabulary", {})
    Vocabulary.register_atom_vocab(config_vocab.get("atom", None))
    Vocabulary.register_bond_vocab(config_vocab.get("bond", None))

    # Load the CSL
    logging.info("Loading CSLDataset.")
    reaction_df = pd.read_parquet(
        os.path.join(library_path, "reactions.parquet")
    )
    synthon_df = pd.read_parquet(os.path.join(library_path, "synthons.parquet"))
    csl_dataset = CSLDataset(
        reaction_df=reaction_df,
        synthon_df=synthon_df,
        load_synthons_and_reactions=load_synthons_and_reactions,
        fast_smiles=fast_smiles,
    )
    logging.info(f"Number of reactions: {csl_dataset.num_reactions:,}.")
    logging.info(f"Number of synthons: {csl_dataset.num_synthons:,}.")
    logging.info(f"Number of products: {len(csl_dataset):,}.")

    # Get device
    device = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"

    # Load the encoder, probe, and factorizer
    encoder = LigandEncoder.load(encoder_weights_path).to(device)
    probe = LinearProbe.load(probe_weights_path).to(device)
    factorizer = APEXFactorizer.load(factorizer_weights_path).to(device)
    encoder.save(os.path.join(outdir, "checkpoints", "encoder.pt"))
    probe.save(os.path.join(outdir, "checkpoints", "probe.pt"))
    factorizer.save(os.path.join(outdir, "checkpoints", "factorizer.pt"))

    # Instantiate APEX model
    apex = APEXFactorizedCSL(
        encoder=encoder,
        factorizer=factorizer,
        dataset=csl_dataset,
        probe=probe,
    ).to(device)

    # Update library tensors
    apex.update_library_tensors()
    torch.save(
        {"model_state_dict": apex.state_dict()},
        os.path.join(outdir, "checkpoints", "apex.pt"),
    )

    logging.info(f"Elapsed time: {datetime.now() - start_time}.")


def main():
    args = parse_arguments()
    run(
        args.config,
        args.library_path,
        args.encoder_weights_path,
        args.probe_weights_path,
        args.factorizer_weights_path,
        args.load_synthons_and_reactions,
        args.fast_smiles,
        args.run_id,
        args.output_dir,
    )
