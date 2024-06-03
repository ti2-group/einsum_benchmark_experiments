# Data types

Start the experiment by running `python exp_data_type.py`.

The experiment uses an instance of the einsum benchmark dataset to show how the data type effects the computation of the einsum expression. Therefore, before the einsum expression is computated, all tensors are cast into the same data type. We explore the date types `int16`, `int32`, `int64`, `float32`, `float64`, `complex64` and `complex128`. After the cast we run the compuatation with different backends (`PyTorch`, `NumPy`, `TensorFlow`, `JAX`). The experiment reports the compuatation time for each data type / backend.