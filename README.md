# ON THE ALIGNMENT BETWEEN SELF-SUPERVISED AND SUPERVISED LEARNING

## Installation

The packages that we use are straightforward to install. Please run the following command:

```bash
conda env create -f requirements.yml
conda activate contrastive
```

## Pretraining models parallelly

To pretrain models with shared randomness, you can run the following command:

```bash
torchrun --nproc_per_node=1 --standalone scripts/parallel_train_simclr.py \
--config <path-to-config-file>
```

Note: We provide config files for all the datasets that we used in our work. Please locate them in `configs/` directory.

## Figure 2: RSA/CKA alignment during training

To get RSA/CKA values for train and test dataset with all four models (CL, NSCL, SCL, and CE), run:

```bash
python scripts/alignment_eval.py \
    --config <path-to-config-file> \
    --ckpt_path <path-to-all-checkpoints-directory> \
    --output_path <path-to-save-metrics>
```

## Figure 3: N-way RSA/CKA analysis

First, we need to pre-train models with shared randomness on the desired dataset. After that, to get N-way RSA/CKA values for train and test dataset with all four models (CL, NSCL, SCL, and CE), run:

```bash
bash bash/run_n_way_training.sh <config-path> <dataset-name>
```

After pretraining, to evaluate the N-way alignment, run:

```bash
bash bash/run_n_way_alignment_eval.sh <config-path> <ckpt-path> <output-path> <classes-per-dataset> <dataset-name>
```

## Figure 4: Varying temperature

To pretrain models with different temperature values, repeat the steps for [pretraining-models-parallelly]() and set different temperature values in the config file.

## Figure 5: Varying batch-size and learning rate

To pretrain models with different batch-size and learning rate values, repeat the steps for [pretraining-models-parallelly]() and set different batch-size and learning rate values in the config file.

## Figure 6: Weight-space coupling

To get average weight gap between different models, run:

```bash
python scripts/weight_space_coupling.py \
    --ckpt_path <path-to-all-checkpoints-directory> \
    --compare <chose between 'cl', 'nscl', 'scl', 'ce> \
    --output_file <path-to-save-metrics>
```
