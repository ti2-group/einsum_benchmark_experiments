import pickle
from timeit import default_timer as timer

import opt_einsum as oe
import numpy as np
import torch as pt
import tensorflow as tf
import jax


def cast_to_backend(tensors, backend="pytorch", dtype="float32"):
    backends = ["torch", "tensorflow", "numpy", "jax"]
    if not backend in backends:
        raise Exception(f"Backend '{backend}' not supported. Please use one of the following backends: {backends}")

    # use the correct data format
    _t = [t.astype(dtype) for t in tensors]

    # convert into the correct backend
    if backend == "torch":
        return [pt.from_numpy(t) for t in _t]
    if backend == "numpy":
        return _t
    if backend == "tensorflow":
        return [tf.convert_to_tensor(t) for t in _t]
    if backend == "jax":
        return [jax.numpy.asarray(t).astype(dtype) for t in _t]


def exec_data_type_experiment(instance: str, iterations=10, backend="torch"):
    """
    instance (str): name of the benchmark instance
    iterations (int): number of iterations for burn in and experiment
    backend (str): backend you want to use
    """
    print(instance)
    times = {}

    # read the data set
    with open(f"./data/{instance}", 'rb') as file:
        format_string, l, path_info, sum_output = pickle.load(file)
        path, size_log2, flops_log10, min_density, avg_density = path_info[0]  # size optimized path

    for data_type in data_types:
        print("start experiment for data type", data_type)
        # first cast the data
        cast_l = cast_to_backend(l, backend, data_type)
        # first burn in the computation iterations time
        print("start burn in:", data_type)
        try:
            result = oe.contract(format_string, *cast_l, optimize=path, backend=backend)
        except Exception as e:
            pass

        # run the experiment iterations times and track all times
        times[data_type] = []
        invalid = False
        result = "N/A"
        for i in range(iterations):
            print(f"{i + 1} / {iterations}", end='\r', flush=True)
            tic = timer()
            try:
                result = oe.contract(format_string, *cast_l, optimize=path, backend=backend)
            except Exception as e:
                print(e)
                invalid = True
            toc = timer()
            if invalid:
                times[data_type].append(np.inf)
                invalid = False
            else:
                times[data_type].append(toc - tic)

    print(times, "\n")

    for i in range(iterations):
        line = backend + "," + ",".join([str(times[dt][i]) for dt in data_types]) + "\n"
        f.write(line)
    f.flush()


if __name__ == "__main__":

    # save results as csv
    header_csv = "backend,"
    # data types we want to test
    data_types = ["int16", "int32", "int64", "float32", "float64", "csingle", "cdouble"]
    header_csv += ",".join(data_types) + "\n"
    print(header_csv)
    f = open("dtype_results.csv", "w")
    f.write(header_csv)
    f.flush()

    instance = "mc_rw_blockmap_05_01.net.pkl"
    iterations = 10

    # start the experiment for each backend
    print("\nJAX")
    exec_data_type_experiment(instance, iterations=iterations, backend="jax")
    print("TensorFlow")
    exec_data_type_experiment(instance, iterations=iterations, backend="tensorflow")
    print("\nPyTorch")
    exec_data_type_experiment(instance, iterations=iterations, backend="torch")
    print("\nNumPy")
    exec_data_type_experiment(instance, iterations=iterations, backend="numpy")
