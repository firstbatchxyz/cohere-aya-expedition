import json
from dria import Dria
from metrics import compute_diversity


def main():
    base_instruction = "Write a math problem."
    # num_nodes = 5
    # iters = 5
    # dria = generate_dria_dataset(base_instruction, num_nodes, iters, "dria")
    # get_metrics(dria)

    bl = generate_bl_dataset(base_instruction, 30, "bl")
    get_metrics(bl)


def generate_dria_dataset(base_instruction, num_nodes, iters, out):

    dria = Dria(num_nodes=num_nodes, base_instruction=base_instruction)
    dria.run(iters)

    for instruction in dria.data:
        print(instruction)

    with open(out + ".json", "w") as f:
        json.dump(dria.data, f)

    return dria


def generate_bl_dataset(base_instruction, num, out):
    dria = Dria(num_nodes=num, base_instruction=base_instruction)
    dria.bootstrap()

    for instruction in dria.data:
        print(instruction)

    with open(out + ".json", "w") as f:
        json.dump(dria.data, f)

    return dria


def get_metrics(dria):
    metrics = compute_diversity(dria.controller.vdb.vecs)
    print(metrics)
    return metrics


main()
