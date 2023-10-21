# Uncovering shape signatures of resting-state functional connectivity by geometric deep learning on Riemannian manifold

This repo contains the code of our HBM 2022 paper [Uncovering shape signatures of resting-state functional connectivity by geometric deep learning on Riemannian manifold](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.25897).

## Usage

1. create an environment

```bash
conda env create -f requirements.yml
```

2. activate the environment

```bash
conda activate HBM-2022
```

3. run train.py

```bash
python train.py
```

## Reference

If you find our work useful in your research, please consider citing:

```bibtex
@article{https://doi.org/10.1002/hbm.25897,
    author = {Dan, Tingting and Huang, Zhuobin and Cai, Hongmin and Lyday, Robert G. and Laurienti, Paul J. and Wu, Guorong},
    title = {Uncovering shape signatures of resting-state functional connectivity by geometric deep learning on Riemannian manifold},
    journal = {Human Brain Mapping},
    volume = {43},
    number = {13},
    pages = {3970-3986},
    keywords = {deep learning, functional brain network, functional dynamics, Riemannian geometry, symmetric positive definite matrix},
    doi = {https://doi.org/10.1002/hbm.25897},
    url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/hbm.25897},
    eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/hbm.25897},
    year = {2022}
}
```