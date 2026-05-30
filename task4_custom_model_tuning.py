"""
Running this file will change values in the output files of the custom model in ModelOutputs/ModelC_R{topic_number}_Ranking.dat
"""
import os

from task3_custom_model import modelC
from task4_eval import (
    average_precision,
    discounted_cumulative_gain_at_rank_10,
    load_ranking,
    load_relevance_judgements,
    mean_score,
    precision_at_rank_10,
)
from topics import load_topics

TOPICS_FILE_PATH = "Topics.txt"
DOC_COLLECTION_PATH = "Doc_Collection"

TOP_R_VALUES = [10, 15, 20]
BOTTOM_NR_VALUES = [0, 10]
ALPHA_VALUES = [8, 10, 20]
BETA_VALUES = [16, 20]
GAMMA_VALUES = [0, 4]
LAMBDA_MODEL_VALUES = [0.8, 0.9]
THETA_VALUES = [2.5, 3.0, 3.5]
TITLE_WEIGHT_VALUES = [1.0, 1.2]
DESC_WEIGHT_VALUES = [0.1, 0.2, 0.5]


def print_and_write(output_file, text=""):
    """
    Function that prints one line and writes the same line to the tuning result file.\n
    Parameters:
        `output_file` - open file object for tuning results
        `text` - text to print and write\n
    Return value: none
    """
    print(text)
    output_file.write(text + "\n")


def evaluate_modelC():
    """
    Function that evaluates Model_C ranking files using the three task4_eval metrics.\n
    Return value: tuple `(map_score, average_p10_score, average_dcg10_score)`
    """
    ranking_dir = "ModelOutputs"
    judgement_dir = "Relevant_Judgements"
    Topics = load_topics(TOPICS_FILE_PATH)
    ap_scores = {}
    p10_scores = {}
    dcg10_scores = {}

    for Ti in Topics:
        Ri = Ti.topic_id
        dataset_id = Ri[1:]

        judgement_path = os.path.join(judgement_dir, f"Dataset{dataset_id}.txt")
        ranking_path = os.path.join(ranking_dir, f"ModelC_{Ri}_Ranking.dat")

        relevance = load_relevance_judgements(judgement_path)
        ranking = load_ranking(ranking_path)

        ap_scores[Ri] = average_precision(ranking, relevance)
        p10_scores[Ri] = precision_at_rank_10(ranking, relevance)
        dcg10_scores[Ri] = discounted_cumulative_gain_at_rank_10(ranking, relevance)

    map_score = mean_score(ap_scores, Topics)
    average_p10_score = mean_score(p10_scores, Topics)
    average_dcg10_score = mean_score(dcg10_scores, Topics)

    return map_score, average_p10_score, average_dcg10_score

def run_modelC_with_parameters(
    top_r,
    bottom_nr,
    alpha,
    beta,
    gamma,
    lambda_model,
    theta,
    title_weight,
    desc_weight,
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
        title_weight,
        desc_weight,
    )

def print_best_result(title, result, output_file):
    """
    Function that prints the best parameter set for one evaluation metric.\n
    Parameters:
        `title` - name of the evaluation metric
        `result` - dictionary containing parameters and evaluation scores\n
    Return value: none
    """
    print_and_write(output_file, title)
    print_and_write(output_file, f"MAP = {result['map_score']:.3f}")
    print_and_write(output_file, f"Average Precision@10 = {result['average_p10_score']:.3f}")
    print_and_write(output_file, f"Average DCG10 = {result['average_dcg10_score']:.3f}")
    print_and_write(output_file, "Parameters:")

    for parameter_name, parameter_value in result["parameters"].items():
        print_and_write(output_file, f"{parameter_name} = {parameter_value}")

    print_and_write(output_file)

def grid_search_modelC(output_file):
    """
    Function that performs a simple for-loop grid search over Model_C parameters.\n
    Return value: tuple `(best_map_result, best_p10_result, best_dcg10_result)`
    """
    total_runs = (
        len(TOP_R_VALUES)
        * len(BOTTOM_NR_VALUES)
        * len(ALPHA_VALUES)
        * len(BETA_VALUES)
        * len(GAMMA_VALUES)
        * len(LAMBDA_MODEL_VALUES)
        * len(THETA_VALUES)
        * len(TITLE_WEIGHT_VALUES)
        * len(DESC_WEIGHT_VALUES)
    )
    run_count = 0
    best_map_result = None
    best_p10_result = None
    best_dcg10_result = None

    for top_r in TOP_R_VALUES:
        for bottom_nr in BOTTOM_NR_VALUES:
            for alpha in ALPHA_VALUES:
                for beta in BETA_VALUES:
                    for gamma in GAMMA_VALUES:
                        for lambda_model in LAMBDA_MODEL_VALUES:
                            for theta in THETA_VALUES:
                                for title_weight in TITLE_WEIGHT_VALUES:
                                    for desc_weight in DESC_WEIGHT_VALUES:
                                        run_count = run_count + 1
                                        parameters = {
                                            "top_r": top_r,
                                            "bottom_nr": bottom_nr,
                                            "alpha": alpha,
                                            "beta": beta,
                                            "gamma": gamma,
                                            "lambda_model": lambda_model,
                                            "theta": theta,
                                            "title_weight": title_weight,
                                            "desc_weight": desc_weight,
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
                                            title_weight,
                                            desc_weight,
                                        )

                                        map_score, average_p10_score, average_dcg10_score, = evaluate_modelC()

                                        result = {
                                            "parameters": parameters,
                                            "map_score": map_score,
                                            "average_p10_score": average_p10_score,
                                            "average_dcg10_score": average_dcg10_score,
                                        }

                                        print(f"MAP = {map_score:.3f}")
                                        print(f"Average Precision@10 = {average_p10_score:.3f}")
                                        print(f"Average DCG10 = {average_dcg10_score:.3f}")
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

                                        if (
                                            best_dcg10_result is None
                                            or average_dcg10_score
                                            > best_dcg10_result["average_dcg10_score"]
                                        ):
                                            best_dcg10_result = result

    print_best_result("Best parameters by MAP", best_map_result, output_file)
    print_best_result("Best parameters by Average Precision@10", best_p10_result, output_file)
    print_best_result("Best parameters by Average DCG10", best_dcg10_result, output_file)

    return best_map_result, best_p10_result, best_dcg10_result

def main():
    with open("task4_modelc_tuning_result.txt", "w") as output_file:
        grid_search_modelC(output_file)


if __name__ == "__main__":
    main()
