import time
import json
import random
from typing import Dict, List, Tuple, FrozenSet

import numpy as np
import pandas as pd
import textdistance as td
from scipy import stats, optimize
from sklearn.metrics.pairwise import cosine_similarity

from . import pillar_globals as g
from .logging_utils import print_log
from .llm_utils import completion_create


def KL_divergence(p, q):
    return stats.entropy(p, q, base=2)


def JS_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * KL_divergence(p, m) + 0.5 * KL_divergence(q, m)


def distribution_similarity(s1, s2, v1, v2) -> float:
    scores: Dict[str, float] = {}

    scores["edit_distance_similarity"] = td.levenshtein.normalized_similarity(s1, s2)

    if (s1, s2) in g.bert_similarity_cache:
        scores["name_bert_similarity"] = g.bert_similarity_cache[(s1, s2)]
    else:
        scores["name_bert_similarity"] = cosine_similarity(
            [g.model.encode(s1)], [g.model.encode(s2)]
        )[0][0]
        g.bert_similarity_cache[(s1, s2)] = scores["name_bert_similarity"]

    try:
        all_values = np.union1d(v1, v2)
        min_value = min(all_values)
        max_value = max(all_values)
        if min_value == max_value:
            scores["JS_divergence_similarity"] = 1.0
        else:
            bin_num = int((len(v1) + len(v2)) / 2)
            bins = np.linspace(min_value, max_value, bin_num + 1)
            prob1 = np.histogram(v1, bins=bins, density=True)[0]
            prob2 = np.histogram(v2, bins=bins, density=True)[0]
            scores["JS_divergence_similarity"] = 1 - JS_divergence(prob1, prob2)
    except Exception:
        print_log(
            f"Error: Invalid values for distribution similarity calculation. "
            f"Error with s1: {s1}, s2: {s2}"
        )

    score_sum = sum(scores.values())
    score_cnt = len(scores)
    return score_sum / max(score_cnt, 1)


def string_similarity(s1, s2) -> float:
    scores: Dict[str, float] = {}
    scores["edit_distance_similarity"] = td.levenshtein.normalized_similarity(s1, s2)

    if (s1, s2) in g.bert_similarity_cache:
        scores["name_bert_similarity"] = g.bert_similarity_cache[(s1, s2)]
    else:
        scores["name_bert_similarity"] = cosine_similarity(
            [g.model.encode(s1)], [g.model.encode(s2)]
        )[0][0]
        g.bert_similarity_cache[(s1, s2)] = scores["name_bert_similarity"]

    score_sum = sum(scores.values())
    score_cnt = len(scores)
    return score_sum / max(score_cnt, 1)


prompt_ner = 'Given the attribute subset for unpivot from a source table and the target attributes, please \
identify the column names that will be generated after unpivot, as well as their column description. Please \
answer in JSON string format {{"column_attribute": {{"name": "xxx", "description": "xxx"}}, "column_value":\
 {{"name": "xxx", "description": "xxx"}}}}, where "column_attribute" refers to the column that \
contains the attributes from source table after unpivot, and "column_value" refers to the column that \
contains the values. Your answer should be pure text rather than a code block.\n\
### Values:\n#\n{subset}\n#\n\
### Descriptions:\n#\n{description}#\n\
### Target attributes:\n#\n{target_attributes}\n'


def query_ner(C: List[int]) -> Tuple[str, str]:
    unpivot_subset = [g.source_attributes[i] for i in C]
    prompt_subset = "# " + "\n# ".join(unpivot_subset)
    prompt_target = "# " + "\n# ".join(g.target_attributes)

    prompt_description = ""
    for attribute in unpivot_subset:
        prompt_description += (
            "# " + attribute + ": " + g.column_explanations[attribute] + "\n"
        )

    prompt = prompt_ner.format(
        subset=prompt_subset,
        description=prompt_description,
        target_attributes=prompt_target,
    )

    get_response = False
    while not get_response:
        query_time = time.time()
        _, response, create_time = completion_create(
            model=g.args.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are now an expert in data governance and schema matching.",
                },
                {"role": "user", "content": prompt},
            ],
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

    column_attribute_name = answer["column_attribute"]["name"]
    column_value_name = answer["column_value"]["name"]
    return column_attribute_name, column_value_name


def similarity_score(C: List[int]) -> Tuple[Dict[str, FrozenSet[str]], float]:
    frozen = frozenset(C)
    if frozen in g.similarity_sum_cache:
        return g.similarity_sum_cache[frozen]
    if frozen in g.calculating:
        while frozen in g.calculating:
            time.sleep(1)
        return g.similarity_sum_cache[frozen]

    g.calculating[frozen] = True

    if len(C) == 1:
        g.calculating.pop(frozen, None)
        return {}, 0.0

    unpivot_subset = [g.source_attributes[i] for i in C]

    for a in unpivot_subset:
        for b in unpivot_subset:
            if g.source_types[a] != g.source_types[b]:
                g.calculating.pop(frozen, None)
                return {}, 0.0

    source_attributes_unpivot = [
        g.source_attributes[i] for i in range(len(g.source_attributes)) if i not in C
    ]

    if len(C) != 0:
        df_tmp = g.df_source.copy()
        df_tmp = pd.melt(
            df_tmp,
            id_vars=source_attributes_unpivot,
            value_vars=unpivot_subset,
            var_name="attributes",
            value_name="value",
        )
        combined = df_tmp["value"].values
        attributes = df_tmp["attributes"].values
    else:
        df_tmp = g.df_source
        combined = None
        attributes = None

    column_attribute_name = ""
    column_value_name = ""

    if len(C) != 0:
        column_attribute_name, column_value_name = query_ner(C)
        source_attributes_unpivot.append(column_attribute_name)
        source_attributes_unpivot.append(column_value_name)

    n = len(source_attributes_unpivot)
    m = len(g.target_attributes)

    similarity_matrix = [[0.0 for _ in range(n)] for _ in range(m)]

    for i in range(m):
        t_attr = g.target_attributes[i]
        for j in range(n):
            if len(C) != 0:
                if j == n - 2:
                    similarity_matrix[i][j] = string_similarity(
                        t_attr,
                        source_attributes_unpivot[j],
                    )
                    continue
                elif j == n - 1:
                    if g.target_types[t_attr] != g.source_types[unpivot_subset[0]]:
                        similarity_matrix[i][j] = 0.0
                    else:
                        if g.source_types[unpivot_subset[0]] == np.int64:
                            similarity_matrix[i][j] = distribution_similarity(
                                t_attr,
                                source_attributes_unpivot[j],
                                g.df_target[t_attr].values,
                                combined,
                            )
                        else:
                            similarity_matrix[i][j] = string_similarity(
                                t_attr,
                                source_attributes_unpivot[j],
                            )
                    continue

            s_attr = source_attributes_unpivot[j]
            if g.target_types[t_attr] != g.source_types[s_attr]:
                similarity_matrix[i][j] = 0.0
            else:
                if g.target_types[t_attr] == np.int64:
                    similarity_matrix[i][j] = distribution_similarity(
                        t_attr,
                        s_attr,
                        g.df_target[t_attr].values,
                        g.df_source[s_attr].values,
                    )
                else:
                    similarity_matrix[i][j] = string_similarity(
                        t_attr,
                        s_attr,
                    )

    threshold_min = 0.2
    threshold_max = 0.95
    for i in range(m):
        for j in range(n):
            v = similarity_matrix[i][j]
            if v < threshold_min:
                similarity_matrix[i][j] = 0.0
            elif v > threshold_max:
                similarity_matrix[i][j] = 1.0

    matching: Dict[str, FrozenSet[str]] = {}
    row, col = optimize.linear_sum_assignment(similarity_matrix, maximize=True)
    sim_sum = 0.0

    for k in range(len(row)):
        r = row[k]
        c = col[k]
        sim_sum += similarity_matrix[r][c]
        if similarity_matrix[r][c] > 0:
            if len(C) != 0 and c == n - 2:
                matching[g.target_attributes[r]] = frozenset(unpivot_subset)
            elif len(C) != 0 and c == n - 1:
                matching[g.target_attributes[r]] = frozenset()
            else:
                matching[g.target_attributes[r]] = frozenset(
                    [source_attributes_unpivot[c]]
                )

    g.similarity_sum_cache[frozen] = (matching, sim_sum)
    g.calculating.pop(frozen, None)

    return matching, sim_sum
