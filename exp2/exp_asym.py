import pickle
from timeit import default_timer as timer
import opt_einsum as oe
import cotengra as ctg
import torch as pt
import math

if __name__ == "__main__":
    # load an instance from the einsum benchmark, uncomment the file you want to use
    instance = "qc_circuit_n49_m14_s9_e6_pEFGH_simplified.pkl"
    with open(f"./data/{instance}", 'rb') as file:
        format_string, l, path_info, sum_output = pickle.load(file)

    # optimize paths for size and for flops
    max_repeats = 4096
    max_time = 100.0
    minimize = ["flops", "size"]
    header_csv = "problem,path_optimized_for,flops,size,flops_log10,size_log2,execution_time,path\n"

    # save results
    f = open("asym_flops_size.csv", "w")
    f.write(header_csv)
    f.flush()

    num_runs = 100
    for _ in range(num_runs):
        for minim in minimize:
            # compute a path using cotengra for either flops or size
            optimizer = ctg.HyperOptimizer(minimize=minim, max_repeats=max_repeats, max_time=max_time, progbar=True)
            path, path_info = oe.contract_path(format_string, *l, optimize=optimizer)
            flops_log10 = round(math.log10(path_info.opt_cost), 2)
            size_log2 = round(math.log2(path_info.largest_intermediate), 2)

            cast_l = [pt.from_numpy(t) for t in l]
            results = []

            # track the time for each path
            tic = timer()
            result = oe.contract(format_string, *cast_l, optimize=path, backend="torch")
            toc = timer()
            pytorch_time = toc - tic

            # save results
            entry = f"{instance},{minim},{path_info.opt_cost},{path_info.largest_intermediate},{flops_log10},{size_log2},{pytorch_time},\"{list(path)}\"\n"
            f.write(entry)
            f.flush()
            print(result)
