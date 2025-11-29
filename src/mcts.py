import math
import random
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, FrozenSet, Optional

import pandas as pd

from . import pillar_globals as g
from .logging_utils import print_log
from .llm_utils import completion_create
from .similarity import similarity_score


EPSILON = 1e-6
PARALLEL_NUM = 5


predefined_messages = [
    {
        "role": "system",
        "content": "You are now an expert in data governance, first I'll give you a definition, a requirement \
and some examples, and I need you to remember them for the following request.",
    },
    {
        "role": "user",
        "content": 'Definition:\nUnpivot: Transforming multiple horizontally arranged numeric columns into vertical \
attribute-value pairs, preserving identifier columns, where original column names become values in a new attribute column \
and their corresponding data is consolidated into a unified value column.\n\
Requirement:\nYour task is to detect the attributes that can be unpivoted in the source table. You can select more attributes within reason.\
A source table and a target table for reference will be provided. Your answer should be in JSON format and no explanation \
is needed. For example, if the attributes to be unpivoted is [A, B, C, D], your answer should be \
{"unpivot_columns": ["A", "B", "C", "D"]}. If no attribute is in the unpivot subset, answer with an empty unpivot_columns \
array, that is, {"unpivot_columns": []}. And remember that your JSON string should be pure text, do not put it in a code block.\n\
Example:\nFor input attributes [Product, Jan_Sales, Feb_Sales], the corresponding output attribute is \
[Jan_Sales, Feb_Sales], and the output answer should be {"unpivot_columns": ["Jan_Sales", "Feb_Sales"]}.\n\
Example:\nFor input attributes [Trade, Date, Quantity], the corresponding output attribute is [], \
because there is no attribute to be unpivoted, and the output answer should be {"unpivot_columns": []}.',
    },
    {
        "role": "assistant",
        "content": "Got it! Please provide the source and target tables so I can determine the unpivot columns and provide the JSON output.",
    },
]

prompt_unpivot = "### Identify the columns that can be unpivoted in a list of column names and with no explanation.\n\
### Source column names:\n#\n{attributes}\n#\n\
### Description:\n#\n{description}#\n\
### Target column names for reference:\n#\n{target_attributes}\n"

prompt_feedback = "### Evaluate the columns selected to be unpivoted from the source table. The selection aims to transfer the \
source table to the target table. You should focus on the transformation between the source and target table structure rather than \
the meaning of unpivot. The provided sample data may have been anonymised. Analyze this answer strictly and critically, point out \
every flaw for ervery possible imperfection about the selection. You only need to evaluate the selection of unpivot subset itself. \
Note that the selected subset is under loose limits, your task is to reduce the size of the subset if there exists redundant \
attributes in the subset. Remember the attributes should be selected from the source attributes, do not use names that do not exist.\n\
### Source column names:\n#\n{attributes}\n#\n\
### Description:\n#\n{description}#\n\
### Target column names for reference:\n#\n{target_attributes}\n#\n\
### Sample data from source table:\n#\n{source_sample_data}#\n\
### Sample data from target table:\n#\n{target_sample_data}#\n\
### Selected columns for unpivot:\n#\n# {unpivot_subset}\n#\n"

prompt_refine = "### Refine your selection based on the feedback. If the feedback indicates that the selection is ideal, then you can remain \
the selection unchanged. Note that the suggested subset provided in the feedback may contain attributes that are not in the source table, \
you should not totally rely on it, but rather use it as a reference and strictly select from source attributes.\n\
### Feedback:\n#\n# {feedback}\n#\n"


@dataclass
class Node:
    parent: Optional["Node"] = None
    Q: float = 0.0
    reward: float = 0.0
    C: List[int] = field(default_factory=list)
    matching: Dict[str, FrozenSet[str]] = field(default_factory=dict)
    children: List["Node"] = field(default_factory=list)
    children_num: int = 0
    N: int = 0


def fully_expanded(node: Node) -> bool:
    for child in node.children:
        if child.Q > node.Q:
            return True
    if node.children_num >= g.args.max_children:
        return True
    return False


def best_utc(node: Node) -> Node:
    best = None
    max_utc = -1e18
    for child in node.children:
        utc = child.Q + 2 * math.sqrt(math.log(node.N + 1) / (child.N + EPSILON))
        if best is None or utc > max_utc:
            best = child
            max_utc = utc
    return best


def traverse(node: Node) -> Node:
    while fully_expanded(node) and len(node.children) > 0:
        node = best_utc(node)
    return node


def self_refine(node: Node) -> Optional[Node]:
    unpivot_subset = [g.source_attributes[i] for i in node.C]
    prompt_source = "# " + "\n# ".join(g.source_attributes)
    prompt_target = "# " + "\n# ".join(g.target_attributes)

    prompt_description = ""
    for attribute in g.source_attributes:
        prompt_description += (
            "# " + attribute + ": " + g.column_explanations[attribute] + "\n"
        )

    source_attributes_unpivot = [
        g.source_attributes[i]
        for i in range(len(g.source_attributes))
        if i not in node.C
    ]

    df_tmp = g.df_source.copy()
    if len(node.C) != 0:
        df_tmp = pd.melt(
            df_tmp,
            id_vars=source_attributes_unpivot,
            value_vars=unpivot_subset,
            var_name="generated_attributes",
            value_name="generated_value",
        )

    prompt_source_sample = ""
    for column in df_tmp.columns:
        prompt_source_sample += (
            "# "
            + column
            + ": "
            + "["
            + ", ".join([str(x) for x in df_tmp[column].values])
            + "]\n"
        )

    prompt_target_sample = ""
    for column in g.df_target.columns:
        prompt_target_sample += (
            "# "
            + column
            + ": "
            + "["
            + ", ".join([str(x) for x in g.df_target[column].values])
            + "]\n"
        )

    prompt = prompt_feedback.format(
        attributes=prompt_source,
        description=prompt_description,
        source_sample_data=prompt_source_sample,
        target_sample_data=prompt_target_sample,
        unpivot_subset=unpivot_subset,
        target_attributes=prompt_target,
    )

    messages = [
        {
            "role": "system",
            "content": "You are now an expert in data governance and schema matching, "
            "and provides feedback on the quality of unpivot detection.",
        },
        {"role": "user", "content": prompt},
    ]

    query_time = time.time()
    _, response, create_time = completion_create(
        model=g.args.model,
        messages=messages,
        extra_body={"enable_thinking": False},
        Stream=True,
    )
    feedback = response
    g.total_time_wasted += create_time - query_time

    messages = predefined_messages.copy()
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
    response_init = {"unpivot_columns": unpivot_subset}
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps(response_init),
        }
    )
    messages.append(
        {
            "role": "user",
            "content": prompt_refine.format(feedback=feedback),
        }
    )

    get_response = False
    while not get_response:
        query_time = time.time()
        _, response, create_time = completion_create(
            model=g.args.model,
            messages=messages,
            extra_body={"enable_thinking": False},
            Stream=True,
        )
        get_response = True
        g.total_time_wasted += create_time - query_time
        try:
            answer = json.loads(response)
        except json.JSONDecodeError:
            print_log(
                "Error: Invalid JSON format in response. Response content: " + response
            )
            print_log("Retrying...")
            get_response = False
            continue

    unpivot_columns = answer["unpivot_columns"]
    if len(unpivot_columns) == 0:
        new_C: List[int] = []
    else:
        try:
            new_C = [g.source_attributes.index(col) for col in unpivot_columns]
        except ValueError:
            return None

    matching, reward = similarity_score(new_C)
    child = Node(parent=node, Q=reward, reward=reward, C=new_C, matching=matching)
    child.N = 1
    node.children.append(child)
    return child


def random_add(node: Node) -> Node:
    possible_additions = [i for i in range(len(g.source_attributes)) if i not in node.C]
    addition = random.choice(possible_additions)
    new_C = node.C.copy()
    new_C.append(addition)
    matching, reward = similarity_score(new_C)
    child = Node(parent=node, Q=reward, reward=reward, C=new_C, matching=matching)
    child.N = 1
    node.children.append(child)
    return child


def random_remove(node: Node) -> Node:
    removal = random.choice(list(node.C))
    new_C = node.C.copy()
    new_C.remove(removal)
    matching, reward = similarity_score(new_C)
    child = Node(parent=node, Q=reward, reward=reward, C=new_C, matching=matching)
    child.N = 1
    node.children.append(child)
    return child


def random_swap(node: Node) -> Node:
    removal = random.choice(list(node.C))
    possible_additions = [i for i in range(len(g.source_attributes)) if i not in node.C]
    addition = random.choice(possible_additions)
    new_C = node.C.copy()
    new_C.remove(removal)
    new_C.append(addition)
    matching, reward = similarity_score(new_C)
    child = Node(parent=node, Q=reward, reward=reward, C=new_C, matching=matching)
    child.N = 1
    node.children.append(child)
    return child


def pick_unvisited(node: Node) -> Optional[Node]:
    if random.random() > g.args.epsilon:
        return self_refine(node)
    else:
        possible_ops = []
        if len(node.C) < len(g.source_attributes):
            possible_ops.append("add")
        if len(node.C) > 0:
            possible_ops.extend(["remove", "swap"])

        op = random.choice(possible_ops)
        if op == "add":
            return random_add(node)
        elif op == "remove":
            return random_remove(node)
        else:
            return random_swap(node)


def backpropagate(node: Node) -> None:
    while node is not None:
        max_Q = node.reward
        for child in node.children:
            if child.Q > max_Q:
                max_Q = child.Q
        node.Q = 0.5 * (node.Q + max_Q)
        node.N += 1
        node = node.parent


def MCTS(root: Node) -> Node:
    import concurrent.futures

    for _ in range(g.args.max_iteration):
        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for _ in range(PARALLEL_NUM):
                node = traverse(root)
                node.N += 10
                node.children_num += 1
                futures.append(executor.submit(pick_unvisited, node))

        for future in concurrent.futures.as_completed(futures):
            leaf = future.result()
            if leaf is not None:
                leaf.parent.N -= 10
                backpropagate(leaf)

    queue = [root]
    best = root
    while queue:
        node = queue.pop(0)
        if node.reward > best.reward:
            best = node
        queue.extend(node.children)
    return best
