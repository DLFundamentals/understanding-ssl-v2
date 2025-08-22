# Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning

## Installation

To get started, follow these steps:

```bash
git clone https://github.com/DLFundamentals/understanding-ssl.git
cd understanding-ssl
```

The packages that we use are straightforward to install. Please run the following command:

```bash
conda env create -f requirements.yml
conda activate contrastive
```

## Pretraining SSL models

### Training on Single GPU

Run the following command to train SimCLR on single GPU.

```bash
python scripts/train.py --config <path-to-yaml-config>
```

### Distributed Training on Multiple GPUs

Run the following command to train SimCLR on multiple GPUs.
> **NOTE:** In our experiments, we used 2 GPUs for training. You can adjust the number of GPUs based on your hadrware setup.

```bash
torchrun --nproc_per_node=N_GPUs --standalone scripts/multigpu_train_simclr.py --config <path-to-yaml-config>
```

Replace `N_GPUs` with the number of GPUs you want to use and `<path-to-yaml-config>` with the path to your configuration file.

Please refer to [docs/pretraining](https://github.com/DLFundamentals/understanding-ssl/blob/main/docs/pretraining.md) for more details.

## Linear Probing

To evaluate pretrained encoders via linear probing, you can run:

```bash
python scripts/linear_probe.py --config <path-to-config-file> --ckpt_path <path-to-ckpt-dir> --output_path <path-to-save-logs> --N <n_samples>
```

For example,

```bash
python scripts/linear_probe.py --config configs/simclr_DCL_cifar10_b1024.yaml --ckpt_path experiments/simclr/cifar10_dcl/checkpoints/ --output_path logs/cifar10/ --N 500
```

## Evaluation

To validate our Theorem [1], you can run:

```bash
python scripts/losses_eval.py --config <path-to-config-file> --ckpt_path <path-to-ckpt-dir> --output_path <path-to-save-logs>
```

For example,

```bash
python scripts/losses_eval.py --config configs/simclr_DCL_cifar10_b1024.yaml --ckpt_path experiments/simclr/cifar10_dcl/checkpoints/ --output_path logs/cifar10/simclr/exp1/
```

This will log `losses.csv` file to your `output_path` directory. You can analyse losses as a function of epochs and verify our proposed bound.

Please refer to [docs/evaluation](https://github.com/DLFundamentals/understanding-ssl/blob/main/docs/evaluation.md) scripts for reproducing additional experiments shown in our paper.

## 📚 Citation

If you find our work useful in your research or applications, please cite us using the following BibTeX:

```bibtex
@misc{clVSnscl,
      title={Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning}, 
      author={Achleshwar Luthra and Tianbao Yang and Tomer Galanti},
      year={2025},
      eprint={2506.04411},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.04411}, 
}
```
