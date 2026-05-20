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
from .mcts import Node, MCTS, predefined_messages, prompt_unpivot, prompt_unpivot_without_description


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
        "--mode",
        type=str,
        default="average",
        choices=["average", "llm_only", "llm_weight_only", "score_and_weight"],
        help="similarity mode: average | llm_only | llm_weight_only | score_and_weight",
    )
    parser.add_argument(
        "--epsilon",
        "-e",
        type=float,
        default=0.05,
        help="epsilon for epsilon-random expansion",
    )
    parser.add_argument(
        "--without_description",
        action="store_true",
        help="whether to exclude column descriptions in the prompt",
        default=False,
    )
    parser.add_argument(
        "--root_generation",
        "-r",
        type=str,
        default="query",
        choices=["query", "all", "random"],
        help="method to generate the root node: query | all | random",
    )
    parser.add_argument(
        "--no_parallel",
        action="store_true",
        help="whether to disable parallel processing",
        default=False,
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
        print("mode: ", g.args.mode)
        print("Model: ", g.args.model)
        print("Without description: ", g.args.without_description)
        print("Root generation method: ", g.args.root_generation)

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
        if g.args.without_description:
            g.column_explanations = None
        else:
            with open(column_explanation_dir, "r") as cef:
                g.column_explanations = json.load(cef)
        with open(ground_truth_dir, "r") as gf:
            ground_truth = json.load(gf)

        g.df_source = pd.read_csv(source_dir)
        g.df_target = pd.read_csv(target_dir)
        g.source_types = {col: g.df_source[col].dtype for col in g.source_attributes}
        g.target_types = {col: g.df_target[col].dtype for col in g.target_attributes}

        if g.args.root_generation == "all":
            init_C = list(range(len(g.source_attributes)))
        elif g.args.root_generation == "random":
            import random
            init_len = random.randint(0, len(g.source_attributes))
            init_C = random.sample(range(len(g.source_attributes)), init_len)
        else:

            messages = predefined_messages.copy()

            prompt_source = "# " + "\n# ".join(g.source_attributes)
            prompt_target = "# " + "\n# ".join(g.target_attributes)

            if not g.args.without_description:
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
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": prompt_unpivot_without_description.format(
                            attributes=prompt_source,
                            target_attributes=prompt_target,
                        ),
                    }
                )

            from .llm_utils import completion_create

            retry_count = 0
            while retry_count < 3:
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
                    retry_count = 3
                except json.JSONDecodeError:
                    print_log(
                        "Error: Invalid JSON format in response. Response content: "
                        + response
                    )
                    print_log("Retrying...")
                    retry_count += 1

            unpivot_columns = answer["unpivot_columns"]
            if len(unpivot_columns) == 0:
                init_C: list[int] = []
            else:
                init_C = [g.source_attributes.index(col) for col in unpivot_columns]

            messages.append({"role": "assistant", "content": response})

        matching, reward = similarity_score(init_C, g.args.without_description)
        root = Node(parent=None, Q=reward, reward=reward, C=init_C, matching=matching)
        best = MCTS(root, g.args.without_description)

        final_matching = best.matching
        final_unpivot_subset = list(set(best.C))

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
                print("Ground truth unpivot subset:", unpivot_subset_gt)
            if key in final_matching:
                if final_matching[key] == frozenset(value):
                    print("Correct match for target attribute:", key)
                    if len(value) == 0:
                        value_match = True
                        AAcc += 1
                        print("AAcc increased for correct match of target attribute:", key)
                    else:
                        MAcc += 1
                        print("MAcc increased for correct match of target attribute:", key)
                        if len(value) == 1:
                            AAcc += 2
                            print("AAcc increased by 2 for correct single attribute match of target attribute:", key)
                        else:
                            subset_correct = True
                            AAcc += 1
                            print("AAcc increased for correct subset match of target attribute:", key)
                else:
                    print("Incorrect match for target attribute:", key)
                    if len(value) > 1 and len(final_matching[key]) > 1:
                        AAcc += 1
                        print("AAcc increased for incorrect match of target attribute with multiple attributes:", key)

        if value_match and subset_correct:
            MAcc += 1

        for src_attr in g.source_attributes:
            if src_attr not in source_attributes_matched:
                if src_attr not in source_attributes_final:
                    AAcc += 1
                    print("AAcc increased for unmatched source attribute not in final matching:", src_attr)

        for tgt_attr in g.target_attributes:
            if final_matching.get(tgt_attr) is None:
                if ground_truth.get(tgt_attr) is None:
                    AAcc += 1
                    print("AAcc increased for unmatched target attribute not in ground truth:", tgt_attr)

        if unpivot_subset_gt is not None:
            for idx in final_unpivot_subset:
                if g.source_attributes[idx] in unpivot_subset_gt:
                    AAcc += 1
                    print("AAcc increased for correctly identified unpivot attribute:", g.source_attributes[idx])

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
