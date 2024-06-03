import opt_einsum as oe
import pickle
import numpy as np
import torch
import time
import sql_commands
import sqlite3
import tensorflow as tf

if __name__ == '__main__':
    # save results as csv
    header_csv = "problem,tensors,backend,flops,size,size_log2,flops_log10,min_density,avg_density,time_in_seconds\n"
    f = open("sparse_results.csv", "w")
    f.write(header_csv)
    f.flush()

    instances = ["qc_variational_29.pkl", "mc_2021_036.pkl", "mc_2022_087.pkl"]
    iterations = 10
    warmup_iterations = 1
    path_to_instances = "./data/"
    for instance in instances:
        with open(path_to_instances + instance, 'rb') as file:
            format_string, l, path_info, sum_output = pickle.load(file)
            path, size_log2, flops_log10, min_density, avg_density = path_info[0]  # size optimized path

        ### SQL
        print("SQL:", instance)
        paramater_names = ["T" + str(i) for i in range(len(l))]
        tensors = dict()
        for i in range(len(l)):
            tensors[paramater_names[i]] = l[i]
        einstein_notation = format_string
        pi = oe.contract_path(einstein_notation, *l, optimize=path)[1]
        is_complex = l[0].dtype == np.complex64 or l[0].dtype == np.complex128
        query = sql_commands.sql_einsum_query(einstein_notation, paramater_names, evidence=tensors, path_info=pi,
                                              complex=is_complex)

        # connect to database
        con = sqlite3.connect(':memory:')
        cur = con.cursor()

        # warmup sql
        for _ in range(warmup_iterations):
            sqlite_result = cur.execute(query)
            con.commit()
            sqlite_result = sqlite_result.fetchall()

        for _ in range(iterations):
            tic = time.time()
            sqlite_result = cur.execute(query)
            con.commit()
            sqlite_result = sqlite_result.fetchall()
            toc = time.time()
            sql_time = toc - tic
            print("sql_time:", sql_time, "s")
            print(sqlite_result)

            entry = f"{instance},{len(l)},SQL,{10 ** flops_log10},{2 ** size_log2},{size_log2},{flops_log10},{min_density},{avg_density},{sql_time}\n"
            f.write(entry)
        f.flush()

        ### PyTorch
        print("\nPyTorch:", instance)
        cast_l = [torch.from_numpy(t) for t in l]

        # warmup torch
        for _ in range(warmup_iterations):
            result_torch = oe.contract(format_string, *cast_l, optimize=path, backend="torch")

        for _ in range(iterations):
            tic = time.time()
            result_torch = oe.contract(format_string, *cast_l, optimize=path, backend="torch")
            toc = time.time()
            torch_time = toc - tic
            print("torch_time:", torch_time, "s")
            print(np.sum(result_torch.detach().numpy()))

            entry = f"{instance},{len(l)},PyTorch,{10 ** flops_log10},{2 ** size_log2},{size_log2},{flops_log10},{min_density},{avg_density},{torch_time}\n"
            f.write(entry)
        f.flush()

        ### TensorFlow
        print("\nTensorFlow:", instance)
        cast_l = [tf.convert_to_tensor(t) for t in l]

        # warmup tensorflow
        for _ in range(warmup_iterations):
            result_tensorflow = oe.contract(format_string, *cast_l, optimize=path, backend="tensorflow")

        for _ in range(iterations):
            tic = time.time()
            result_tensorflow = oe.contract(format_string, *cast_l, optimize=path, backend="tensorflow")
            toc = time.time()
            tensorflow_time = toc - tic
            print("tensorflow_time:", tensorflow_time, "s")
            print(np.sum(result_tensorflow.numpy()))

            entry = f"{instance},{len(l)},TensorFlow,{10 ** flops_log10},{2 ** size_log2},{size_log2},{flops_log10},{min_density},{avg_density},{tensorflow_time}\n"
            f.write(entry)
        f.flush()

        try:
            ### NumPy
            print("\nNumPy:", instance)
            # warmup numpy
            for _ in range(warmup_iterations):
                result_numpy = oe.contract(format_string, *l, optimize=path, backend="numpy")

            for _ in range(iterations):
                tic = time.time()
                result_numpy = oe.contract(format_string, *l, optimize=path, backend="numpy")
                toc = time.time()
                numpy_time = toc - tic
                print("numpy_time:", numpy_time, "s")
                print(np.sum(result_numpy))

                entry = f"{instance},{len(l)},NumPy,{10 ** flops_log10},{2 ** size_log2},{size_log2},{flops_log10},{min_density},{avg_density},{numpy_time}\n"
                f.write(entry)
            f.flush()
        except:
            print("Numpy failed:", instance)
            for _ in range(iterations):
                entry = f"{instance},{len(l)},NumPy,{10 ** flops_log10},{2 ** size_log2},{size_log2},{flops_log10},{min_density},{avg_density},N/A\n"
                f.write(entry)
            f.flush()
    f.close()
