#!/usr/bin/env python
"""
This script trains an LBM surrogate that is adaptable to APEX factorization. The
surrogate is decomposed into an encoder and a (linear) probe. The training setup
is multi-task regression, and we use deep-fried least squares for training the
probe.

Saves outputs to $OUTPUT_DIR:
- Config: ${OUTPUT_DIR}/config.yaml
- Logs: ${OUTPUT_DIR}/logs.log
- Encoder weights: ${OUTPUT_DIR}/checkpoints/encoder_${ITERATION}.pt
- Probe weights: ${OUTPUT_DIR}/checkpoints/probe_${ITERATION}.pt
- Tensorboard events file

train_surrogate \\
--config atomwise/cslvae/configs/apex.yaml \\
--parquet_path $PARQUET_PATH \\
--training_folds 0 \\
--validation_folds 1 \\
--property_columns mol_wt logp n_hba n_hbd score_met ... \\
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
import duckdb
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml

# Atomwise
from apex.data.atom_bond_features import Vocabulary
from apex.dataset.labeled_smiles_dataset import LabeledSmilesDataset
from apex.nn.encoder import LigandEncoder
from apex.nn.probe import LinearProbe
from apex.utils.other_utils import init_logging, makedirs


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "This script trains an LBM surrogate on a labeled dataset "
            "and subsequently trains APEX on a CSL to distill the LBM "
            "predictions."
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
        "--parquet_path",
        type=str,
        action="store",
        required=True,
        help="Path to dataframe for fitting the LBM surrogate. It should have a"
        "categorical column named 'fold' indicating the fold a row belongs to.",
    )
    parser.add_argument(
        "--training_folds",
        type=str,
        action="store",
        nargs="+",
        required=True,
        help="List of training fold indexes.",
    )
    parser.add_argument(
        "--validation_folds",
        type=str,
        action="store",
        nargs="+",
        required=True,
        help="List of validation fold indexes.",
    )
    parser.add_argument(
        "--property_columns",
        type=str,
        action="store",
        nargs="+",
        required=True,
        help="List of property columns.",
    )
    parser.add_argument(
        "--encoder_weights_path",
        type=str,
        action="store",
        required=False,
        default=None,
        help="Path to weights for encoder.",
    )
    parser.add_argument(
        "--probe_weights_path",
        type=str,
        action="store",
        required=False,
        default=None,
        help="Path to weights for probe.",
    )
    parser.add_argument(
        "--smiles_column",
        type=str,
        action="store",
        required=False,
        default="smiles",
        help="Name of SMILES column in the training dataframe.",
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
    encoder: LigandEncoder,
    probe: LinearProbe,
    train_dataset: LabeledSmilesDataset,
    val_dataset: LabeledSmilesDataset,
    config: dict,
    outdir: str,
):
    # Create checkpoints directory
    checkpoints_path = os.path.join(outdir, "checkpoints")
    makedirs(checkpoints_path)

    # Instantiate probe
    device = next(encoder.parameters()).device
    probe = probe.to(device)
    assert (
        train_dataset.property_columns
        == val_dataset.property_columns
        == probe.output_names
    )

    # Unpack config
    batch_size = int(config.get("batch_size"))
    max_iterations = int(config.get("max_iterations"))
    logging_iterations = int(config.get("logging_iterations", 100))
    checkpoint_iterations = int(config.get("checkpoint_iterations", 1000))
    noise_scale = float(config.get("noise_scale", 1.0))

    # Instantiate writer
    writer = SummaryWriter(log_dir=outdir)

    # Instantiate optimizer
    optimizer_name = config.get("optimizer_name")
    optimizer_kwargs = config.get("optimizer_kwargs")
    optimizer_cls = getattr(torch.optim, optimizer_name)
    optimizer = optimizer_cls(encoder.parameters(), **optimizer_kwargs)

    # Instantiate dataloaders
    train_dataloader = train_dataset.create_dataloader(
        batch_size=batch_size,
        max_iterations=max_iterations,
    )
    val_dataloader = val_dataset.create_dataloader(
        batch_size=batch_size,
        max_iterations=max_iterations,
    )
    data_iterator = enumerate(zip(train_dataloader, val_dataloader))

    # Start training
    logging.info(f"Training surrogate for {max_iterations:,} iterations.")
    (sum_train_loss, sum_val_loss) = (0, 0)
    (sum_train_losses, sum_val_losses) = (0, 0)
    iter_count = 0
    for iteration, (train_batch, val_batch) in data_iterator:
        # Set to train mode and reset gradient
        encoder.train()
        probe.train()

        # Unpack batch items
        train_mols = train_batch["mols"].to(device)
        train_values = train_batch["values"].to(device)

        # Get embeddings and add noise
        train_embeds = encoder(train_mols)
        train_embeds += noise_scale * torch.randn_like(train_embeds)

        # Get predicted values
        pred_train_values = probe(train_embeds, train_values)

        # Calculate loss (normalized MSE) and backprop
        train_losses = (train_values - pred_train_values).pow(2).mean(
            0
        ) / train_values.var(0, unbiased=False)
        train_loss = torch.mean(train_losses)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            encoder.eval()
            probe.eval()
            val_mols = val_batch["mols"].to(device)
            val_values = val_batch["values"].to(device)
            val_embeds = encoder(val_mols)
            pred_val_values = probe(val_embeds)
            val_losses = (val_values - pred_val_values).pow(2).mean(
                0
            ) / val_values.var(0, unbiased=False)
            val_loss = torch.mean(val_losses)

        sum_train_loss += train_loss.item()
        sum_val_loss += val_loss.item()
        sum_train_losses += train_losses.detach()
        sum_val_losses += val_losses.detach()
        iter_count += 1

        # Logging
        if (iteration % logging_iterations == 0) or (
            iteration == max_iterations
        ):
            avg_train_loss = sum_train_loss / iter_count
            avg_val_loss = sum_val_loss / iter_count
            avg_train_losses = sum_train_losses / iter_count
            avg_val_losses = sum_val_losses / iter_count
            logging.info(
                f"Iteration: {iteration}. "
                f"Loss (train): {avg_train_loss:.4f}. "
                f"Loss (validation): {avg_val_loss:.4f}. "
            )
            if writer is not None:
                writer.add_scalar("Loss (train)", avg_train_loss, iteration)
                writer.add_scalar(
                    "Loss (validation)",
                    avg_val_loss,
                    iteration,
                )
                for i, name in enumerate(probe.output_names):
                    writer.add_scalar(
                        f"Loss, {name} (train)",
                        avg_train_losses[i],
                        iteration,
                    )
                    writer.add_scalar(
                        f"Loss, {name} (validation)",
                        avg_val_losses[i],
                        iteration,
                    )
            (sum_train_loss, sum_val_loss) = (0, 0)
            (sum_train_losses, sum_val_losses) = (0, 0)
            iter_count = 0

        # Checkpointing
        if checkpoint_iterations is not None:
            if (iteration % checkpoint_iterations == 0) or (
                iteration == max_iterations
            ):
                logging.info(f"Check-pointing model iteration {iteration}.")
                encoder.save(
                    os.path.join(checkpoints_path, f"encoder_{iteration}.pt")
                )
                encoder.save(os.path.join(checkpoints_path, "encoder.pt"))
                probe.save(
                    os.path.join(checkpoints_path, f"probe_{iteration}.pt")
                )
                probe.save(os.path.join(checkpoints_path, "probe.pt"))

        if iteration == max_iterations:
            logging.info(f"Maximum iterations ({max_iterations}) reached.")
            break

    writer.close()


def main(
    config: str,
    parquet_path: str,
    training_folds: list[str],
    validation_folds: list[str],
    property_columns: list[str],
    encoder_weights_path: Optional[str] = None,
    probe_weights_path: Optional[str] = None,
    smiles_column: str = "smiles",
    run_id: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    # Start-up
    start_time = datetime.now()
    run_id = str(uuid.uuid4().hex[:16] if run_id is None else run_id)
    output_dir = str(os.getcwd() if output_dir is None else output_dir)
    outdir = os.path.join(output_dir, run_id)
    makedirs(outdir)
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
    config_train_surrogate = config_all.get("train_surrogate")

    # Register vocab
    Vocabulary.register_atom_vocab(config_vocab.get("atom", None))
    Vocabulary.register_bond_vocab(config_vocab.get("bond", None))

    # Get device
    device = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"

    # Load the train and validation datasets
    con = duckdb.connect()
    query = f"SELECT {smiles_column},{','.join(property_columns)} "
    query += f"FROM '{parquet_path}' WHERE "
    train_df = con.execute(query + f"fold IN ({','.join(training_folds)})").df()
    val_df = con.execute(query + f"fold IN ({','.join(validation_folds)})").df()

    # train_df = pd.read_csv(train_df_path)
    # val_df = pd.read_csv(val_df_path)
    train_dataset = LabeledSmilesDataset(
        df=train_df,
        property_columns=property_columns,
        smiles_column=smiles_column,
    )
    val_dataset = LabeledSmilesDataset(
        df=val_df,
        property_columns=property_columns,
        smiles_column=smiles_column,
    )

    # Instantiate the encoder
    if encoder_weights_path is None:
        config_encoder = config_all.get("encoder")
        encoder = LigandEncoder(**config_encoder).to(device)
        probe = LinearProbe(
            input_dim=encoder.embed_dim,
            output_names=property_columns,
        ).to(device)
    else:
        encoder = LigandEncoder.load(encoder_weights_path).to(device)
        if probe_weights_path is None:
            probe = LinearProbe(
                input_dim=encoder.embed_dim,
                output_names=property_columns,
            ).to(device)
        else:
            probe = LinearProbe.load(probe_weights_path).to(device)
            assert probe.output_names == property_columns

    num_encoder_params = sum([p.numel() for p in encoder.parameters()])
    logging.info(f"Number of encoder parameters: {num_encoder_params:,}.")

    # Train the encoder and probe (which combine to form the surrogate) on
    # labeled data
    train(
        encoder=encoder,
        probe=probe,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config_train_surrogate,
        outdir=outdir,
    )

    logging.info(f"Elapsed time: {datetime.now() - start_time}.")


if __name__ == "__main__":
    args = parse_arguments()
    main(
        args.config,
        args.parquet_path,
        args.training_folds,
        args.validation_folds,
        args.property_columns,
        args.encoder_weights_path,
        args.probe_weights_path,
        args.smiles_column,
        args.run_id,
        args.output_dir,
    )
