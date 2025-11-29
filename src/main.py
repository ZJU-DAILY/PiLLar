import argparse
import csv
import json
import os
import sys
import time

import pandas as pd

from . import pillar_globals as g
from .logging_utils import print_log
from .similarity import similarity_score
from .mcts import Node, MCTS, predefined_messages, prompt_unpivot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, help="dataset name")
    parser.add_argument(
        "--max_iteration",
        "-i",
        type=int,
        default=2,
        help="number of iterations for MCTS",
    )
    parser.add_argument(
        "--max_children",
        "-c",
        type=int,
        default=3,
        help="maximum number of children for each node",
    )
    parser.add_argument(
        "--top_k",
        "-k",
        type=int,
        default=3,
        help="number of top results to return from MCTS",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="qwen3-235b-a22b",
        help="model to use for LLM completion",
    )
    parser.add_argument(
        "--epsilon",
        "-e",
        type=float,
        default=0.05,
        help="epsilon for epsilon-random expansion",
    )
    g.args = parser.parse_args()

    if not os.path.exists("./output"):
        os.makedirs("./output")
    if not os.path.exists(f"./output/{g.args.dataset}"):
        os.makedirs(f"./output/{g.args.dataset}")
    log_dir = f"./output/{g.args.dataset}/PiLLar_{g.args.model}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    start_t = time.localtime()
    g.start_time = start_t
    log_path = os.path.join(
        log_dir, "log" + time.strftime("%Y%m%d%H%M%S", start_t) + ".txt"
    )

    with open(log_path, "w") as f:
        sys.stdout = f

        print("Dataset: ", g.args.dataset)
        print("Max iteration: ", g.args.max_iteration)
        print("Max children: ", g.args.max_children)
        print("epsilon: ", g.args.epsilon)

        dataset_dir = "./dataset/"
        source_dir = os.path.join(dataset_dir, g.args.dataset, "source.csv")
        target_dir = os.path.join(dataset_dir, g.args.dataset, "target.csv")
        column_explanation_dir = os.path.join(
            dataset_dir, g.args.dataset, "column_explanations.json"
        )
        ground_truth_dir = os.path.join(
            dataset_dir, g.args.dataset, "ground_truth.json"
        )

        with open(source_dir, "r") as sf:
            g.source_attributes = next(csv.reader(sf))
        with open(target_dir, "r") as tf:
            g.target_attributes = next(csv.reader(tf))
        with open(column_explanation_dir, "r") as cef:
            g.column_explanations = json.load(cef)
        with open(ground_truth_dir, "r") as gf:
            ground_truth = json.load(gf)

        g.df_source = pd.read_csv(source_dir)
        g.df_target = pd.read_csv(target_dir)
        g.source_types = {col: g.df_source[col].dtype for col in g.source_attributes}
        g.target_types = {col: g.df_target[col].dtype for col in g.target_attributes}

        messages = predefined_messages.copy()

        prompt_source = "# " + "\n# ".join(g.source_attributes)
        prompt_target = "# " + "\n# ".join(g.target_attributes)
        prompt_description = ""
        for attribute in g.source_attributes:
            prompt_description += (
                "# " + attribute + ": " + g.column_explanations[attribute] + "\n"
            )

        messages.append(
            {
                "role": "user",
                "content": prompt_unpivot.format(
                    attributes=prompt_source,
                    description=prompt_description,
                    target_attributes=prompt_target,
                ),
            }
        )

        from .llm_utils import completion_create

        get_response = False
        while not get_response:
            query_time = time.time()
            _, response, create_time = completion_create(
                model=g.args.model,
                messages=messages,
                extra_body={"enable_thinking": False},
                Stream=True,
            )
            g.total_time_wasted += create_time - query_time
            try:
                answer = json.loads(response)
                get_response = True
            except json.JSONDecodeError:
                print_log(
                    "Error: Invalid JSON format in response. Response content: "
                    + response
                )
                print_log("Retrying...")

        unpivot_columns = answer["unpivot_columns"]
        if len(unpivot_columns) == 0:
            init_C: list[int] = []
        else:
            init_C = [g.source_attributes.index(col) for col in unpivot_columns]

        messages.append({"role": "assistant", "content": response})

        matching, reward = similarity_score(init_C)
        root = Node(parent=None, Q=reward, reward=reward, C=init_C, matching=matching)
        best = MCTS(root)

        final_matching = best.matching
        final_unpivot_subset = best.C

        print(
            "Final unpivot subset:",
            [g.source_attributes[i] for i in final_unpivot_subset],
        )
        print("Final matching:")
        for key, value in final_matching.items():
            print(list(value), "->", key)

        MAcc = 0
        AAcc = 0
        unpivot_subset_gt = None
        subset_correct = False
        value_match = False
        source_attributes_matched = set()
        source_attributes_final = set()

        for attrs in final_matching.values():
            source_attributes_final.update(attrs)

        for key, value in ground_truth.items():
            source_attributes_matched.update(value)
            if len(value) > 1:
                unpivot_subset_gt = frozenset(value)
            if key in final_matching:
                if final_matching[key] == frozenset(value):
                    if len(value) == 0:
                        value_match = True
                        AAcc += 1
                    else:
                        MAcc += 1
                        if len(value) == 1:
                            AAcc += 2
                        else:
                            subset_correct = True
                            AAcc += 1
                else:
                    if len(value) > 1 and len(final_matching[key]) > 1:
                        AAcc += 1

        if value_match and subset_correct:
            MAcc += 1

        for src_attr in g.source_attributes:
            if src_attr not in source_attributes_matched:
                if src_attr not in source_attributes_final:
                    AAcc += 1

        for tgt_attr in g.target_attributes:
            if final_matching.get(tgt_attr) is None:
                if ground_truth.get(tgt_attr) is None:
                    AAcc += 1

        if unpivot_subset_gt is not None:
            for idx in final_unpivot_subset:
                if g.source_attributes[idx] in unpivot_subset_gt:
                    AAcc += 1

        print("E2E Accuracy (Acc_E2E):", MAcc)
        print("Per Attribute Accuracy (Acc_per_attr.):", AAcc)

        end_t = time.localtime()
        g.end_time = end_t
        print("Start time: ", time.strftime("%Y-%m-%d %H:%M:%S", start_t))
        print("End time: ", time.strftime("%Y-%m-%d %H:%M:%S", end_t))
        print(
            "Total time: ",
            time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    time.mktime(end_t) - time.mktime(start_t) - g.total_time_wasted
                ),
            ),
        )


if __name__ == "__main__":
    main()
