# Experiments for Einsum Benchmark Dataset

The full dataset is stored at [zenodo](https://zenodo.org/records/11477304).
You can conveniently access it using our [wrapper package](https://github.com/ti2-group/einsum_benchmark).

## Experiments

This repository contains four different experiments. The explaination and code for each experiment is contained in one sub folder:

- [Experiment 1](exp1) - Scalability of processing large contraction paths
- [Experiment 2](exp2) - Asymmetric tensor sizes in pairwise contractions
- [Experiment 3](exp3) - Sparse intermediate tensors
- [Experiment 4](exp4) - Data types

## Prerequisites

The experiments were executed on the follwing environment:

- python (3.10.9)
- numpy (1.26.4)
- opt_einsum (3.3.0)
- cotengra (0.6.2)
- kahypar (1.3.5)
- tensorflow (2.16.1)
- torch (1.12.1)
- jax (0.4.28)
- jaxlib (0.4.28)
- pyinstrument (4.6.2)
- tqdm (4.66.4)
- sqlite (3.45.3)

You can use the `env.yml` file to create the conda environment `einsumbm` using

```
conda env create -f env.yml
```

To remove the environment first deactivate it

```
conda deactivate
```

and then remove it

```
conda env remove -n einsumbm
```
