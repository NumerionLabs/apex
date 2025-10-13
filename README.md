# APEX
Code for [APEX: Approximate-but-exhaustive search for ultra-large combinatorial synthesis libraries]()


### Setup

First install the [apex_topk](https://github.com/AtomwiseInc/apex_topk) extension module. Then install [apex](https://github.com/AtomwiseInc/apex) as follows:

```
git clone https://github.com/AtomwiseInc/apex.git
cd apex
pip install .
```

### Datasets

The data used in the paper can be downloaded [here]() and includes computationally labeled combinatorial synthesis libraries (CSLs) with more than 1M (training/validation) and 12M (testing) compounds, along with an unlabeled CSL with more than 10B compounds that can be searched with APEX. The datasets are structued as follows:
```
.
├── brics_csl_10B
│   ├── reactions.parquet
│   └── synthons.parquet
├── brics_csl_12M
│   ├── enumerated_and_scored.parquet
│   ├── reactions.parquet
│   └── synthons.parquet
└── brics_csl_1M
    ├── enumerated_and_scored.parquet
    ├── reactions.parquet
    └── synthons.parquet
```

### Usage

#### Train surrogate

```
--config configs/apex.yaml \
--parquet_path ../brics_csl_1M/enumerated_and_scored.parquet \
--training_folds 0 \
--validation_folds 1 \
--property_columns fraction_cs_p3 logp max_ring_size mol_wt mr n_aliphatic_rings n_aromatic_ring n_atoms n_bridgehead_atoms n_hba n_hbd n_heavy_atom n_heterocycles n_ring n_rotatable_bonds n_stereocenter n_unspecified_stereocenter qed qeppi score_DRD2 score_ESR1 score_F10 score_MET score_PARP1 sps synthetic_accessibility_score tpsa weird_fragment_score \
--output_dir $OUTPUT_DIR \
--run_id $SURROGATE_RUN_ID
```

#### Train factorizer

```
train_factorizer \
--config configs/apex.yaml \
--encoder_weights_path $OUTPUT_DIR/$SURROGATE_RUN_ID/checkpoints/encoder.pt \
--probe_weights_path $OUTPUT_DIR/$SURROGATE_RUN_ID/checkpoints/probe.pt \
--library_path ../brics_csl_12M/ \
--output_dir $OUTPUT_DIR \
--run_id $FACTORIZER_RUN_ID
```
