#!/usr/bin/env python
"""
This script trains an APEX factorizer on a CSL to reconstruct the embeddings of
an LBM surrogate.

Saves outputs to $OUTPUT_DIR:
- Config: ${OUTPUT_DIR}/config.yaml
- Logs: ${OUTPUT_DIR}/logs.log
- Factorizer weights: ${OUTPUT_DIR}/checkpoints/factorizer_${ITERATION}.pt
- Final APEX weights: ${OUTPUT_DIR}/checkpoints/apex.pt
- Tensorboard events file

train_factorizer \\
  --config atomwise/cslvae/configs/apex.yaml \\
  --encoder_weights_path $ENCODER_WEIGHTS_PATH \\
  --probe_weights_path $PROBE_WEIGHTS_PATH \\
  --reaction_df_path $REACTION_DF_PATH \\
  --synthon_df_path $SYNTHON_DF_PATH \\
  --output_dir $OUTPUT_DIR \\
  --run_id $RUN_ID
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
from torch import LongTensor, Tensor
from torch.utils.tensorboard import SummaryWriter
import yaml

# Atomwise
from apex.data.atom_bond_features import Vocabulary
from apex.dataset.csl_dataset import CSLDataset
from apex.nn.apex import APEXFactorizedCSL, APEXFactorizer
from apex.nn.encoder import LigandEncoder
from apex.nn.probe import LinearProbe
from apex.utils.other_utils import init_logging, makedirs


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "This script trains APEX on a CSL to distill the predictions of an "
            "LBM surrogate."
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
        "--reaction_df_path",
        type=str,
        action="store",
        required=True,
        help="Path to dataframe of reaction SMARTS for the CSL.",
    )
    parser.add_argument(
        "--synthon_df_path",
        type=str,
        action="store",
        required=True,
        help="Path to dataframe of synthon SMILES for the CSL.",
    )
    parser.add_argument(
        "--factorizer_weights_path",
        type=str,
        action="store",
        required=False,
        default=None,
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


def train(
    apex: APEXFactorizedCSL,
    config: dict,
    outdir: Optional[str] = None,
):
    # Unpack config
    device = config.get("device", next(apex.factorizer.parameters()).device)
    batch_num_products_per_reaction = int(
        config.get("batch_num_products_per_reaction")
    )
    batch_num_reactions = int(config.get("batch_num_reactions"))
    logging_iterations = int(config.get("logging_iterations", 100))
    checkpoint_iterations = int(config.get("checkpoint_iterations", 1000))
    max_iterations = int(config.get("max_iterations"))

    # Instantiate writer
    writer = None if outdir is None else SummaryWriter(log_dir=outdir)

    # Instantiate optimizer
    apex.to(device)
    optimizer_name = config.get("optimizer_name")
    optimizer_kwargs = config.get("optimizer_kwargs")
    optimizer_cls = getattr(torch.optim, optimizer_name)
    optimizer = optimizer_cls(
        apex.factorizer.parameters(),
        **optimizer_kwargs,
    )

    # Instantiate dataloader
    dataloader = apex.dataset.create_dataloader_stratified_uniformly_at_random(
        num_products_per_reaction=batch_num_products_per_reaction,
        num_reactions=batch_num_reactions,
        max_iterations=max_iterations,
    )

    # Start training
    logging.info(f"Beginning training APEX for {max_iterations:,} iterations.")
    (sum_loss, count) = (0, 0)
    for iteration, batch in enumerate(dataloader):
        # Only APEX parameters are to be trained
        apex.factorizer.train()
        apex.encoder.eval()

        # Reset gradient
        optimizer.zero_grad()

        # Unpack batch items
        library_indexes: dict[str, LongTensor] = {
            k: v.to(device) for k, v in batch["library_indexes"].items()
        }
        synthon2rgroup = library_indexes["synthon2rgroup"].to(device)
        block2product = batch["block2product"].to(device)
        block2rgroup = batch["block2rgroup"].to(device)
        block2synthon = batch["block2synthon"].to(device)
        synthons = batch["synthons"].to(device)
        products = batch["products"].to(device)

        block_synthon2rgroup = torch.stack([block2synthon[1], block2rgroup[1]])
        _, block2asynthon = torch.where(
            torch.sum(
                torch.abs(
                    block_synthon2rgroup[:, :, None]
                    - synthon2rgroup[:, None, :]
                ),
                dim=0,
            )
            == 0
        )

        # Get synthon embeddings
        synthon_feats = apex.factorizer.synthon_graph_encoder(synthons)

        # Encode the library
        library_tensors: dict[str, Tensor] = apex.encode_library(
            synthon_feats,
            library_indexes,
        )

        # Embed the products
        with torch.no_grad():
            embeds: Tensor = apex.encoder(products)

        # Get APEX predicted embeddings
        apex_embeds: Tensor = apex.factorizer.embed_predictor_pooling_function(
            library_tensors["synthon_associative_embeds"][block2asynthon],
            block2product,
        )

        # Calculate loss and backprop
        loss = torch.mean(
            (embeds - apex_embeds).pow(2)
            / embeds.var(0, unbiased=False, keepdim=True)
        )
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        count += 1

        # Logging
        if (iteration % logging_iterations == 0) or (
            iteration == max_iterations
        ):
            avg_loss = sum_loss / count
            logging.info(f"Iteration: {iteration}. Loss: {avg_loss:.4f}.")
            if writer is not None:
                writer.add_scalar("Loss", avg_loss, iteration)
            (sum_loss, count) = (0, 0)

        # Checkpointing
        if (iteration % checkpoint_iterations == 0) or (
            iteration == max_iterations
        ):
            logging.info(f"Check-pointing model iteration {iteration}.")

            if outdir is not None:
                apex.factorizer.save(
                    os.path.join(
                        outdir,
                        "checkpoints",
                        f"factorizer_{iteration}.pt",
                    )
                )
                apex.factorizer.save(
                    os.path.join(
                        outdir,
                        "checkpoints",
                        "factorizer_latest.pt",
                    )
                )

        if iteration == max_iterations:
            logging.info(f"Maximum iterations ({max_iterations}) reached.")
            break

    checkpoint_path = os.path.join(outdir, "checkpoints", "factorizer_final.pt")
    apex.factorizer.save(checkpoint_path)

    if writer is not None:
        writer.close()


def main(
    config: str,
    encoder_weights_path: str,
    probe_weights_path: str,
    reaction_df_path: str,
    synthon_df_path: str,
    factorizer_weights_path: Optional[str] = None,
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

    # Get config parts
    config_vocab = config_all.get("vocabulary", {})
    config_factorizer = config_all.get("factorizer")
    config_train_apex = config_all.get("train_apex")

    # Register vocab
    Vocabulary.register_atom_vocab(config_vocab.get("atom", None))
    Vocabulary.register_bond_vocab(config_vocab.get("bond", None))

    # Load the CSL
    logging.info("Loading CSLDataset.")
    reaction_df = pd.read_csv(reaction_df_path)
    synthon_df = pd.read_csv(synthon_df_path)
    Vocabulary.register_atom_vocab()
    Vocabulary.register_bond_vocab()
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

    # Load the encoder and probe
    encoder = LigandEncoder.load(encoder_weights_path).to(device)
    probe = LinearProbe.load(probe_weights_path).to(device)
    encoder.save(os.path.join(outdir, "checkpoints", "encoder.pt"))
    probe.save(os.path.join(outdir, "checkpoints", "probe.pt"))

    # Instantiate APEX factorizer and pair with encoder + CSL
    if factorizer_weights_path is None:
        factorizer = APEXFactorizer(**config_factorizer).to(device)
    else:
        logging.info(f"Loading factorizer from {factorizer_weights_path}.")
        factorizer = APEXFactorizer.load(factorizer_weights_path).to(device)
    apex = APEXFactorizedCSL(
        encoder=encoder,
        factorizer=factorizer,
        dataset=csl_dataset,
        probe=probe,
    ).to(device)
    num_params = sum([p.numel() for p in factorizer.parameters()])
    logging.info(f"Number of factorizer parameters: {num_params:,}.")

    # Train factorizer on encoder distillation task
    train(apex, config_train_apex, outdir)

    # Update library tensors
    apex.update_library_tensors()
    torch.save(
        {"model_state_dict": apex.state_dict()},
        os.path.join(outdir, "checkpoints", "apex.pt"),
    )

    logging.info(f"Elapsed time: {datetime.now() - start_time}.")


if __name__ == "__main__":
    args = parse_arguments()
    main(
        args.config,
        args.encoder_weights_path,
        args.probe_weights_path,
        args.reaction_df_path,
        args.synthon_df_path,
        args.library_name,
        args.factorizer_weights_path,
        args.load_synthons_and_reactions,
        args.fast_smiles,
        args.run_id,
        args.output_dir,
    )
