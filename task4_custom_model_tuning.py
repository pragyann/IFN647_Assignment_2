import os
from contextlib import redirect_stdout

from task3_custom_model import modelC
from task4_eval import (
    average_precision,
    load_ranking,
    load_relevance_judgements,
    mean_score,
    precision_at_rank_10,
)
from topics import load_topics

TOPICS_FILE_PATH = "Topics.txt"
DOC_COLLECTION_PATH = "Doc_Collection"

TOP_R_VALUES = [10, 20]
BOTTOM_NR_VALUES = [0, 10]
ALPHA_VALUES = [8]
BETA_VALUES = [16]
GAMMA_VALUES = [0, 4]
LAMBDA_MODEL_VALUES = [0.8, 0.9]
THETA_VALUES = [3.0, 3.5]

def evaluate_modelC():
    """
    Function that evaluates Model_C ranking files using the two task4_eval metrics.\n
    Return value: tuple `(map_score, average_p10_score)`
    """
    ranking_dir = "ModelOutputs"
    judgement_dir = "Relevant_Judgements"
    Topics = load_topics(TOPICS_FILE_PATH)
    ap_scores = {}
    p10_scores = {}

    for Ti in Topics:
        Ri = Ti.topic_id
        dataset_id = Ri[1:]

        judgement_path = os.path.join(judgement_dir, f"Dataset{dataset_id}.txt")
        ranking_path = os.path.join(ranking_dir, f"ModelC_{Ri}_Ranking.dat")

        relevance = load_relevance_judgements(judgement_path)
        ranking = load_ranking(ranking_path)

        ap_scores[Ri] = average_precision(ranking, relevance)
        p10_scores[Ri] = precision_at_rank_10(ranking, relevance)

    map_score = mean_score(ap_scores, Topics)
    average_p10_score = mean_score(p10_scores, Topics)

    return map_score, average_p10_score

def run_modelC_with_parameters(
    top_r,
    bottom_nr,
    alpha,
    beta,
    gamma,
    lambda_model,
    theta,
):
    """
    Function that runs Model_C for one set of tuning parameters.\n
    Parameters are the same tuning values passed to `modelC`.\n
    Return value: none
    """

    modelC(
        TOPICS_FILE_PATH,
        DOC_COLLECTION_PATH,
        top_r,
        bottom_nr,
        alpha,
        beta,
        gamma,
        lambda_model,
        theta,
    )

def print_best_result(title, result):
    """
    Function that prints the best parameter set for one evaluation metric.\n
    Parameters:
        `title` - name of the evaluation metric
        `result` - dictionary containing parameters and evaluation scores\n
    Return value: none
    """
    print(title)
    print(f"MAP = {result['map_score']:.3f}")
    print(f"Average Precision@10 = {result['average_p10_score']:.3f}")
    print("Parameters:")

    for parameter_name, parameter_value in result["parameters"].items():
        print(f"{parameter_name} = {parameter_value}")

    print()

def grid_search_modelC():
    """
    Function that performs a simple for-loop grid search over Model_C parameters.\n
    Return value: tuple `(best_map_result, best_p10_result)`
    """
    total_runs = (
        len(TOP_R_VALUES)
        * len(BOTTOM_NR_VALUES)
        * len(ALPHA_VALUES)
        * len(BETA_VALUES)
        * len(GAMMA_VALUES)
        * len(LAMBDA_MODEL_VALUES)
        * len(THETA_VALUES)
    )
    run_count = 0
    best_map_result = None
    best_p10_result = None

    for top_r in TOP_R_VALUES:
        for bottom_nr in BOTTOM_NR_VALUES:
            for alpha in ALPHA_VALUES:
                for beta in BETA_VALUES:
                    for gamma in GAMMA_VALUES:
                        for lambda_model in LAMBDA_MODEL_VALUES:
                            for theta in THETA_VALUES:
                                run_count = run_count + 1
                                parameters = {
                                    "top_r": top_r,
                                    "bottom_nr": bottom_nr,
                                    "alpha": alpha,
                                    "beta": beta,
                                    "gamma": gamma,
                                    "lambda_model": lambda_model,
                                    "theta": theta,
                                }

                                print(f"Grid search run {run_count}/{total_runs}")
                                run_modelC_with_parameters(
                                    top_r,
                                    bottom_nr,
                                    alpha,
                                    beta,
                                    gamma,
                                    lambda_model,
                                    theta,
                                )

                                map_score, average_p10_score = evaluate_modelC()

                                result = {
                                    "parameters": parameters,
                                    "map_score": map_score,
                                    "average_p10_score": average_p10_score,
                                }

                                print(f"MAP = {map_score:.3f}")
                                print(f"Average Precision@10 = {average_p10_score:.3f}")
                                print()

                                if (
                                    best_map_result is None
                                    or map_score > best_map_result["map_score"]
                                ):
                                    best_map_result = result

                                if (
                                    best_p10_result is None
                                    or average_p10_score
                                    > best_p10_result["average_p10_score"]
                                ):
                                    best_p10_result = result

    print_best_result("Best parameters by MAP", best_map_result)
    print_best_result("Best parameters by Average Precision@10", best_p10_result)

    return best_map_result, best_p10_result


def main():
    grid_search_modelC()


if __name__ == "__main__":
    main()
