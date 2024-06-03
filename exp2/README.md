# Asymmetric tensor sizes in pairwise contractions

Start the experiment by running `python exp_asym.py`.

The experiment demonstrates a significant discrepancy between the theoretical number of operations and actual execution times of einsum expressions due to asymmetric tensor sizes during pairwise contractions. Notably, paths computed to minimize the theoretical number of operations are susceptible to these asymmetric contractions. For the experiment we use `cotengra` and `kahypar` to compute contraction paths, optimizing for intermediate size and the number of operations on a quantum circuit instance from the dataset. The experiment repeatedly measures and compares the execution times of einsum expressions against the optimization method used for computing the path.