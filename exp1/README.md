# Scalability of processing large contraction paths

Start the experiment by running `python exp_path.py`.

The experiment uses instances of the einsum benchmark dataset to show the inefficiencies of contraction path compuatations for an einsum expression with `opt_einsum`. Comparing `greedy ssa` (the actual path compuatation), `ssa to linear` (the transformation into another format) and `generate out strings` (generation strings of the pairwise computation). The latter two times can be seen as overhead, as the path is already present after performing `greedy ssa`.

