# ON THE ALIGNMENT BETWEEN SELF-SUPERVISED AND SUPERVISED LEARNING

## Installation

To get started, follow these steps:

```bash
git clone https://github.com/DLFundamentals/understanding_ssl_v2.git
cd understanding_ssl_v2
```

The packages that we use are straightforward to install. Please run the following command:

```bash
conda env create -f requirements.yml
conda activate contrastive
```

## Pretraining models parallelly

### Distributed Training on Multiple GPUs

Run the following command to train SimCLR on multiple GPUs.
> **NOTE:** In our experiments, we used 2 GPUs for training. You can adjust the number of GPUs based on your hadrware setup.

```bash
torchrun --nproc_per_node=N_GPUs --standalone scripts/parallel_dcl_nscl_train_simclr.py --config <path-to-yaml-config>
```

Replace `N_GPUs` with the number of GPUs you want to use and `<path-to-yaml-config>` with the path to your configuration file.

Please refer to [docs/pretraining](https://github.com/DLFundamentals/understanding-ssl/blob/main/docs/pretraining.md) for more details.
