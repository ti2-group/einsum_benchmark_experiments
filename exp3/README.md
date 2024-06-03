# Sparse intermediate tensors

Start the experiment by running `python exp_sparse.py`.

The experiment uses instances of the einsum benchmark dataset to demonstrate inefficiencies in handling sparse intermediate tensors. To highlight these inefficiencies, the same dataset instance is executed using both PyTorch, which does not specialize in sparse formats, and SQLite, which processes the data in a sparse format. The SQL query for computing the einsum expression is generated in `sql_commands.py`. 