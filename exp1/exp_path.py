import opt_einsum as oe
import pickle
import math
from pyinstrument import Profiler
import os


def sorted_pkl_files(directory):
    items = []
    for file in os.listdir(directory):
        if file.endswith(".pkl"):
            items.append(file)
    sorted_items = sorted(items)
    return sorted_items


def profile_problem(format_string, l):
    # profile the problem
    profiler = Profiler()
    profiler.start()
    path, desc = oe.contract_path(format_string, *l, optimize="greedy")
    profiler.stop()
    out = profiler.output_text(unicode=True, color=False, show_all=True)

    # collect information
    time_find_contraction = 0
    if " find_contraction" in out:
        time_find_contraction = float(out.split(" find_contraction", 1)[0].rsplit(" ", 1)[1])
    greedy_time_contraction = 0
    greedy_ssa_time_contraction = 0
    if " greedy  " in out and " ssa_greedy_optimize  " in out:
        greedy_time_contraction = float(out.split(" greedy  ", 1)[0].rsplit(" ", 1)[1])
        greedy_ssa_time_contraction = float(out.split(" ssa_greedy_optimize  ", 1)[0].rsplit(" ", 1)[1])
    ssa_to_linear = round(greedy_time_contraction - greedy_ssa_time_contraction, 6)
    total = round(time_find_contraction + greedy_ssa_time_contraction + ssa_to_linear, 6)
    flops_log10 = round(math.log10(desc.opt_cost), 2)
    size_log2 = round(math.log2(desc.largest_intermediate), 2)
    return greedy_ssa_time_contraction, ssa_to_linear, time_find_contraction, total, flops_log10, size_log2


if __name__ == '__main__':
    path_to_problems_folder = "./data/"

    i = 10001
    header_csv = "problem,input_tensors,greedy_ssa_time_contraction,ssa_to_linear,time_find_contraction,total," \
                 "flops_log10,size_log2\n"
    f = open("oe_one_path_results.csv", "w")
    f.write(header_csv)
    f.flush()

    # run the experiments each time in the same order
    problems = sorted_pkl_files(path_to_problems_folder)
    problems_count_str = str(len(problems))

    for problem in problems:
        print(problem)
        with open(path_to_problems_folder + problem, 'rb') as file:
            # load an instance from the einsum benchmark
            format_string, l, path_info, sum_output = pickle.load(file)

            # start the profiling
            greedy_ssa_time_contraction, ssa_to_linear, time_find_contraction, total, flops_log10, size_log2 = profile_problem(format_string, l)

            input_tensors = len(l)
            print(str(i)[-len(problems_count_str):] + " of " + problems_count_str + ":", problem)
            print("input_tensors:", input_tensors)
            print("greedy ssa:", greedy_ssa_time_contraction, "s")
            print("ssa to linear:", ssa_to_linear, "s")
            print("generate output strings:", time_find_contraction, "s")
            print("total:", total, "s")
            print("log10[FLOPS]:", flops_log10)
            print("log2[SIZE]:", size_log2)
            print()
            entry = f"{problem},{input_tensors},{greedy_ssa_time_contraction},{ssa_to_linear},{time_find_contraction},{total},{flops_log10},{size_log2}\n"
            f.write(entry)
            f.flush()
            i += 1

    f.close()
