import os
from topics import load_topics

MODELS = ["Baseline1", "Baseline2", "ModelC"]

def load_relevance_judgements(judgement_path):
    """
    Function that reads relevance judgements for one topic.\n
    Parameters: `judgement_path` - path of `DatasetXXX.txt` relevance judgement file\n
    Return value: dictionary `{DocID: relevance_label}`
    """
    relevance = {}

    for line in open(judgement_path, "r"):
        line = line.strip()
        parts = line.split()
        relevance[parts[1]] = float(parts[2])

    return relevance


def load_ranking(ranking_path):
    """
    Function that reads a saved model ranking file in rank order.\n
    Parameters: `ranking_path` - path of ranking `.dat` file\n
    Return value: ordered list of document IDs
    """
    ranking = []

    for line in open(ranking_path, "r"):
        line = line.strip()
        if len(line) == 0:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        # Header lines such as `Query is ...` and `Doc_ID BM25_Score` are skipped.
        try:
            float(parts[1])
            ranking.append(parts[0])
        except ValueError:
            continue

    return ranking


def average_precision(ranking, relevance):
    """
    Function that computes average precision using positional precision of relevant documents.\n
    Parameters:
        `ranking` - ordered list of document IDs
        `relevance` - relevance dictionary `{DocID: relevance_label}`\n
    Return value: average precision score
    """
    R = len([DocID for DocID, label in relevance.items() if label > 0])

    if R == 0:
        return 0

    ri = 0
    map1 = 0.0

    for position, DocID in enumerate(ranking, start=1):
        if relevance.get(DocID, 0) > 0:
            ri = ri + 1
            pi = float(ri) / float(position)
            map1 = map1 + pi

    return map1 / float(R)


def precision_at_rank_10(ranking, relevance):
    """
    Function that computes precision at rank position 10.\n
    Parameters:
        `ranking` - ordered list of document IDs
        `relevance` - relevance dictionary `{DocID: relevance_label}`
    Return value: precision at rank position 10
    """
    rank = 10
    relevant_found = 0

    for DocID in ranking[:rank]:
        if relevance.get(DocID, 0) > 0:
            relevant_found = relevant_found + 1

    return float(relevant_found) / float(rank)

def mean_score(scores, topics):
    """
    Function that computes the mean score over all topics.\n
    Parameters: `scores` - dictionary `{topic_id: score}`, `topic_ids` - list of topic IDs\n
    Return value: mean score
    """
    if len(topics) == 0:
        return 0

    return sum([scores[topic.topic_id] for topic in topics]) / float(len(topics))


def print_table(title, topics, model_scores, summary_label):
    """
    Function that prints an evaluation table for all models.\n
    Parameters:
        `title` - table title
        `topic_ids` - list of topic IDs
        `model_scores` - dictionary `{model_name: {topic_id: score}}`
        `summary_label` - final row label, e.g. `MAP` or `Average`\n
    Return value: none
    """
    print(title)
    print("Topic\t" + "\t".join(MODELS))

    for topic in topics:
        topic_id = topic.topic_id
        row = [topic_id]
        for model_name in MODELS:
            row.append(f"{model_scores[model_name][topic_id]:.3f}")
        print("\t".join(row))

    row = [summary_label]
    for model_name in MODELS:
        row.append(f"{mean_score(model_scores[model_name], topics):.3f}")
    print("\t".join(row))
    print()


def evaluate_all_models(ranking_dir, judgement_dir):
    """
    Function that evaluates all ranking models using AP and precision@10.\n
    Parameters:
        `ranking_dir` - folder containing model ranking files
        `judgement_dir` - folder containing relevance judgement files\n
    Return value: tuple `(ap_results, p10_results)`
    """
    topics = load_topics('Topics.txt')
    ap_results = {}
    p10_results = {}

    for model_name in MODELS:
        ap_results[model_name] = {}
        p10_results[model_name] = {}

    for topic in topics:
        topic_id = topic.topic_id
        dataset_id = topic_id[1:]
        judgement_path = os.path.join(judgement_dir, f"Dataset{dataset_id}.txt")
        relevance = load_relevance_judgements(judgement_path)

        for model_name in MODELS:
            ranking_path = os.path.join(ranking_dir, f"{model_name}_{topic_id}_Ranking.dat")
            ranking = load_ranking(ranking_path)
            
            ap_score = average_precision(ranking, relevance)
            p10_score = precision_at_rank_10(ranking, relevance)

            ap_results[model_name][topic_id] = ap_score
            p10_results[model_name][topic_id] = p10_score

    print_table("Average Precision", topics, ap_results, "MAP")
    print_table(f"Precision@10", topics, p10_results, "Average")

    return ap_results, p10_results


def main():
    evaluate_all_models("ModelOutputs", "Relevant_Judgements")


if __name__ == "__main__":
    main()
