# Study on various factors affecting alignment during training

This document helps to reproduce Figures 3-5 of the paper.

## Effect of Classes ($C$)

To reproduce the N-way RSA/CKA results across CL, NSCL, SCL, and CE models:

### 1. Pretraining

Pretrain models with **shared randomness** on the target dataset:

```bash
bash bash/run_n_way_training.sh <config-path> <dataset-name>
```

### 2. Evaluation

After pretraining, evaluate N-way alignment on both train and test sets:

```bash
bash bash/run_n_way_alignment_eval.sh <config-path> <ckpt-path> <output-path> <classes-path> <dataset-name>
```

**Notes:**

- `<ckpt-path>` should be a directory containing checkpoints for all four models.
- `<classes-path>` should be a .txt file in the format described below.
- The script logs RSA and CKA metrics to `<output-path>` and automatically associates them with figure labels.

Example format for classes.txt

```text
N = 2; Classes = [7 8]
N = 4; Classes = [6 5 4 9]
N = 6; Classes = [5 6 1 0 2 7]
N = 8; Classes = [1 9 3 7 8 0 6 5]
```

## Effect of Temperature ($\tau$)

To study the effect of temperature on alignment:

1. Follow the [pretraining models in parallel](https://github.com/DLFundamentals/understanding_ssl_v2/tree/main?tab=readme-ov-file#pretraining-models-with-shared-randomness) procedure.
2. Modify the `temperature` field in your config file to one of the desired values (e.g., `0.1`, `0.5`, `1.0`).
3. Re-run pretraining for each temperature setting.

After training, evaluate alignment using the same command as earlier.

**Notes:**

- The resulting RSA/CKA logs correspond to Figure 4 in the paper.

## Effect of Batch Size ($B$)

To analyze how alignment changes with batch size ($B$) and learning rate ($\eta$):

1. Follow the [pretraining models in parallel](https://github.com/DLFundamentals/understanding_ssl_v2/tree/main?tab=readme-ov-file#pretraining-models-with-shared-randomness) procedure.
2. Modify the `batch_size` and `lr_rate` field in your config file.
3. Train and evaluate using the same commands as above..

**Notes:**

- Recommended scaling rules:

  - Linear: `$\eta \propto B$
  - Square-root: `$\eta \propto \sqrt{B}$
  - Fourth-root: `$\eta \propto B^{1/4}$
- Results correspond to Figure 5 of the paper.
