import time
import json
import random
import re
import concurrent.futures
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

LLM_MODES = {"llm_only", "llm_weight_only", "score_and_weight"}

prompt_similarity_llm_only = "Evaluate the similarity between the two given columns according to the source/target table \
schema provided. The similarity score should be a float number between 0 and 1. Answer with only one final float \
score, no any other character should be added.\n\
### Column 1: {column1}\n\
### Column 2: {column2}\n\
### Sample data for column 1:\n#\n#{sample1}\n#\n\
### Sample data for column 2:\n#\n#{sample2}\n#\n\
### Source column names before unpivot:\n#\n{source_attributes_before}\n#\n\
### Source column names after unpivot:\n#\n{source_attributes_after}\n#\n\
### Target column names:\n#\n{target_attributes}\n#\n\
### Similarity scores:\n#\n{scores}\n"

prompt_similarity_llm_weight_only = "Evaluate the similarity between the two given columns by combining the provided dimensional \
scores. The similarity score should be a float number between 0 and 1. Answer with only one final float score, \
no any other character should be added.\n\
### Column 1: {column1}\n\
### Column 2: {column2}\n\
### Sample data for column 1:\n#\n#{sample1}\n#\n\
### Sample data for column 2:\n#\n#{sample2}\n#\n\
### Source column names before unpivot:\n#\n{source_attributes_before}\n#\n\
### Source column names after unpivot:\n#\n{source_attributes_after}\n#\n\
### Target column names:\n#\n{target_attributes}\n#\n\
### Similarity scores:\n#\n{scores}\n"

prompt_similarity_llm_weight_only_unpivot = "Evaluate the similarity between the two given columns by combining the provided dimensional \
scores. If the JS distribution similarity is provided, assign it a higher weight than the others. The similarity \
score should be a float number between 0 and 1. Answer with only one final float score, no any other character \
should be added.\n\
### Column 1: {column1}\n\
### Column 2: {column2}\n\
### Sample data for column 1:\n#\n#{sample1}\n#\n\
### Sample data for column 2:\n#\n#{sample2}\n#\n\
### Source column names before unpivot:\n#\n{source_attributes_before}\n#\n\
### Source column names after unpivot:\n#\n{source_attributes_after}\n#\n\
### Target column names:\n#\n{target_attributes}\n#\n\
### Similarity scores:\n#\n{scores}\n"

prompt_similarity_score_and_weight = "Evaluate the similarity between the two given columns. The similarity score should be a \
float number between 0 and 1. Answer with only one final float score, no any other character should be added. \
Do it step by step:\n\
1. Give a similarity score between the two attributes according to the source/target table schema provided.\n\
2. Combine this score with other provided dimensional scores (0-1) using weighted averaging.\n\
### Column 1: {column1}\n\
### Column 2: {column2}\n\
### Sample data for column 1:\n#\n#{sample1}\n#\n\
### Sample data for column 2:\n#\n#{sample2}\n#\n\
### Source column names before unpivot:\n#\n{source_attributes_before}\n#\n\
### Source column names after unpivot:\n#\n{source_attributes_after}\n#\n\
### Target column names:\n#\n{target_attributes}\n#\n\
### Similarity scores:\n#\n{scores}\n"

prompt_similarity_score_and_weight_unpivot = "Evaluate the similarity between the two given columns. The similarity score should be a \
float number between 0 and 1. Answer with only one final float score, no any other character should be added. \
Do it step by step:\n\
1. Give a similarity score between the two attributes according to the source/target table schema provided.\n\
2. Combine this score with other provided dimensional scores (0-1) using weighted averaging. If the JS distribution \
similarity is provided, assign it a higher weight than the others.\n\
### Column 1: {column1}\n\
### Column 2: {column2}\n\
### Sample data for column 1:\n#\n#{sample1}\n#\n\
### Sample data for column 2:\n#\n#{sample2}\n#\n\
### Source column names before unpivot:\n#\n{source_attributes_before}\n#\n\
### Source column names after unpivot:\n#\n{source_attributes_after}\n#\n\
### Target column names:\n#\n{target_attributes}\n#\n\
### Similarity scores:\n#\n{scores}\n"

similarity_definition = {
    "edit_distance_similarity": "Normalized similarity based on edit distance between attribute names, i.e. the number of insertions, deletions, or substitutions required to change one string into the other",
    "name_bert_similarity": "Cosine similarity between BERT embeddings for attribute names",
    "description_bert_similarity": "Cosine similarity between BERT embeddings for attribute descriptions",
    "JS_divergence_similarity": "Jensen-Shannon divergence similarity between distributions of sample values, it has been preprocessed to be a float number between 0 and 1, the bigger the value, the more similar the two distributions are",
}

def llm_combine(
    s1: str,
    s2: str,
    scores: Dict[str, float],
    v1,
    v2,
    prompt_strs: List[str],
    mode: str,
    is_unpivot: bool,
) -> float:
    cache_key = (mode, s1, s2, is_unpivot, prompt_strs[1])
    if cache_key in g.similarity_cache:
        return g.similarity_cache[cache_key]

    scores_str = ""
    for dimension, score in scores.items():
        scores_str += (
            "# "
            + dimension
            + ": "
            + "{:.10f}".format(score)
            + "\n"
            + "# definition: "
            + similarity_definition[dimension]
            + "\n"
        )

    v1_str = "[" + ", ".join([str(x) for x in v1]) + "]"
    v2_str = "[" + ", ".join([str(x) for x in v2]) + "]"

    if mode == "llm_only":
        prompt = prompt_similarity_llm_only.format(
            column1=s1,
            column2=s2,
            sample1=v1_str,
            sample2=v2_str,
            source_attributes_before=prompt_strs[0],
            source_attributes_after=prompt_strs[1],
            target_attributes=prompt_strs[2],
            scores=scores_str,
        )
    elif mode == "llm_weight_only":
        if is_unpivot:
            prompt = prompt_similarity_llm_weight_only_unpivot.format(
                column1=s1,
                column2=s2,
                sample1=v1_str,
                sample2=v2_str,
                source_attributes_before=prompt_strs[0],
                source_attributes_after=prompt_strs[1],
                target_attributes=prompt_strs[2],
                scores=scores_str,
            )
        else:
            prompt = prompt_similarity_llm_weight_only.format(
                column1=s1,
                column2=s2,
                sample1=v1_str,
                sample2=v2_str,
                source_attributes_before=prompt_strs[0],
                source_attributes_after=prompt_strs[1],
                target_attributes=prompt_strs[2],
                scores=scores_str,
            )
    elif mode == "score_and_weight":
        if is_unpivot:
            prompt = prompt_similarity_score_and_weight_unpivot.format(
                column1=s1,
                column2=s2,
                sample1=v1_str,
                sample2=v2_str,
                source_attributes_before=prompt_strs[0],
                source_attributes_after=prompt_strs[1],
                target_attributes=prompt_strs[2],
                scores=scores_str,
            )
        else:
            prompt = prompt_similarity_score_and_weight.format(
                column1=s1,
                column2=s2,
                sample1=v1_str,
                sample2=v2_str,
                source_attributes_before=prompt_strs[0],
                source_attributes_after=prompt_strs[1],
                target_attributes=prompt_strs[2],
                scores=scores_str,
            )
    else:
        return 0.0

    futures = []
    scores_list: List[float] = []
    query_time = time.time()
    max_create_time = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for _ in range(3):
            futures.append(
                executor.submit(
                    completion_create,
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
            )
        for future in concurrent.futures.as_completed(futures):
            _, response, create_time = future.result()
            max_create_time = max(max_create_time, create_time)
            try:
                response_score = float(re.findall(r"[-+]?\d*\.\d+|\d+", response)[0])
                scores_list.append(response_score)
            except Exception:
                print_log(
                    "Error: Invalid response format. Response content: " + response
                )

    if len(scores_list) == 0:
        score = 0.0
    else:
        score = sum(scores_list) / len(scores_list)

    g.total_time_wasted += max_create_time - query_time
    g.similarity_cache[cache_key] = score
    return score


def distribution_similarity(
    s1,
    s2,
    v1,
    v2,
    prompt_strs: List[str],
    mode: str,
    is_unpivot: bool,
) -> float:
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

    if mode in LLM_MODES:
        return llm_combine(s1, s2, scores, v1, v2, prompt_strs, mode, is_unpivot)

    score_sum = sum(scores.values())
    score_cnt = len(scores)
    return score_sum / max(score_cnt, 1)


def string_similarity(
    s1,
    s2,
    v1,
    v2,
    prompt_strs: List[str],
    mode: str,
    is_unpivot: bool,
) -> float:
    scores: Dict[str, float] = {}
    scores["edit_distance_similarity"] = td.levenshtein.normalized_similarity(s1, s2)

    if (s1, s2) in g.bert_similarity_cache:
        scores["name_bert_similarity"] = g.bert_similarity_cache[(s1, s2)]
    else:
        scores["name_bert_similarity"] = cosine_similarity(
            [g.model.encode(s1)], [g.model.encode(s2)]
        )[0][0]
        g.bert_similarity_cache[(s1, s2)] = scores["name_bert_similarity"]

    if mode in LLM_MODES:
        return llm_combine(s1, s2, scores, v1, v2, prompt_strs, mode, is_unpivot)

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

prompt_ner_without_description = 'Given the attribute subset for unpivot from a source table and the target attributes, please \
identify the column names that will be generated after unpivot, as well as their column description. Please \
answer in JSON string format {{"column_attribute": {{"name": "xxx", "description": "xxx"}}, "column_value":\
 {{"name": "xxx", "description": "xxx"}}}}, where "column_attribute" refers to the column that \
contains the attributes from source table after unpivot, and "column_value" refers to the column that \
contains the values. Your answer should be pure text rather than a code block.\n\
### Values:\n#\n{subset}\n#\n\
### Target attributes:\n#\n{target_attributes}\n'

def query_ner(C: List[int], without_description: bool) -> Tuple[str, str]:
    unpivot_subset = [g.source_attributes[i] for i in C]
    prompt_subset = "# " + "\n# ".join(unpivot_subset)
    prompt_target = "# " + "\n# ".join(g.target_attributes)

    if not without_description:
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
    else:
        prompt = prompt_ner_without_description.format(
            subset=prompt_subset,
            target_attributes=prompt_target,
        )

    retry_count = 0
    while retry_count < 3:
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
        g.total_time_wasted += create_time - query_time
        try:
            answer = json.loads(response)
            retry_count = 3
        except json.JSONDecodeError:
            print_log(
                "Error: Invalid JSON format in response. Response content: " + response
            )
            print_log("Retrying...")
            retry_count += 1

    column_attribute_name = answer["column_attribute"]["name"]
    column_value_name = answer["column_value"]["name"]
    return column_attribute_name, column_value_name


def similarity_score(
    C: List[int], without_description: bool
) -> Tuple[Dict[str, FrozenSet[str]], float]:
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

    # for a in unpivot_subset:
    #     for b in unpivot_subset:
    #         if g.source_types[a] != g.source_types[b]:
    #             g.calculating.pop(frozen, None)
    #             return {}, 0.0

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
        column_attribute_name, column_value_name = query_ner(C, without_description)
        source_attributes_unpivot.append(column_attribute_name)
        source_attributes_unpivot.append(column_value_name)

    prompt_strs = [
        "# " + "\n# ".join(g.source_attributes),
        "# " + "\n# ".join(source_attributes_unpivot),
        "# " + "\n# ".join(g.target_attributes),
    ]

    n = len(source_attributes_unpivot)
    m = len(g.target_attributes)

    similarity_matrix = [[0.0 for _ in range(n)] for _ in range(m)]
    mode = g.args.mode
    if mode not in LLM_MODES:
        mode = "average"

    for i in range(m):
        t_attr = g.target_attributes[i]
        for j in range(n):
            if len(C) != 0:
                if j == n - 2:
                    similarity_matrix[i][j] = string_similarity(
                        t_attr,
                        source_attributes_unpivot[j],
                        g.df_target[t_attr].values,
                        attributes,
                        prompt_strs,
                        mode,
                        True,
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
                                prompt_strs,
                                mode,
                                True,
                            )
                        else:
                            similarity_matrix[i][j] = string_similarity(
                                t_attr,
                                source_attributes_unpivot[j],
                                g.df_target[t_attr].values,
                                combined,
                                prompt_strs,
                                mode,
                                True,
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
                        prompt_strs,
                        mode,
                        False,
                    )
                else:
                    similarity_matrix[i][j] = string_similarity(
                        t_attr,
                        s_attr,
                        g.df_target[t_attr].values,
                        g.df_source[s_attr].values,
                        prompt_strs,
                        mode,
                        False,
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
