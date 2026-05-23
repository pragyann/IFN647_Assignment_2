import os
from math import log10

from coll import BowColl, parse_documents, parseQuery, load_stop_words
from topics import Topic, load_topics


LAMBDA = 0.3

def build_index(coll: BowColl):
    """
    Function that builds an inverted index and document length dictionary from a collection.\n
    Parameters: `coll` - collection of documents as `BowColl`\n
    Return value: tuple `(I, DocLengths)` where `I` is `{term: {DocID: frequency}}` and `DocLengths` is `{DocID: length}`
    """
    I = {}
    DocLengths = {}

    for DocID, D in coll.get_docs().items():
        # Store |D|, the length of the current document
        DocLengths[DocID] = D.get_doc_len()

        # Add every term frequency to the posting list for that term
        for term, freq in D.get_term_freq_dict().items():
            if term not in I:
                I[term] = {}
            I[term][DocID] = freq

    return I, DocLengths


def collection_len(DocLengths):
    """
    Function that computes total number of word occurrences in a dataset.\n
    Parameters: `DocLengths` - document length dictionary `{DocID: length}`\n
    Return value: total number of word occurrences `|Dataset_i|`
    """
    total_dataset_length = 0

    # Sum all document lengths to compute |Dataset_i|
    for dl in DocLengths.values():
        total_dataset_length = total_dataset_length + dl

    return total_dataset_length


def collection_term_counts(I, Qi):
    """
    Function that computes collection frequency for each query term.\n
    Parameters:
        `I` - inverted index `{term: {DocID: frequency}}`
        `Qi` - parsed topic title query dictionary `{term: frequency}`\n
    Return value: dictionary `{q_j: c_qj}`
    """
    C = {}

    for q_j in Qi:
        # c_qj = total number of times query term q_j occurs in Dataset_i
        if q_j in I:
            C[q_j] = sum(I[q_j].values())
        else:
            C[q_j] = 0

    return C


def jm_likelihood(I, DocLengths, Qi, C, lambda_value=LAMBDA):
    """
    Function that computes JM-smoothed likelihood scores for all documents in one dataset.\n
    Parameters:
        `I` - inverted index `{term: {DocID: frequency}}`
        `DocLengths` - document length dictionary `{DocID: length}`
        `Qi` - parsed topic title query dictionary `{term: qf_j}`
        `C` - collection term counts `{q_j: c_qj}`
        `lambda_value` - JM smoothing parameter\n
    Return value: dictionary of JM scores `{DocID: JM_score}`
    """
    Score = {}
    Dataset_i_len = collection_len(DocLengths)

    if Dataset_i_len == 0:
        return Score

    for DocID, D_len in DocLengths.items():
        Score[DocID] = 0.0

        # Skip probability calculation for an empty document to avoid division by zero
        if D_len == 0:
            continue

        # Score[D] is the sum of log probabilities for all query terms q_j
        for q_j, qf_j in Qi.items():
            # f_qj_D = number of times query term q_j occurs in document D
            f_qj_D = I.get(q_j, {}).get(DocID, 0)

            # c_qj = number of times query term q_j occurs in the whole Dataset_i
            c_qj = C.get(q_j, 0)

            # JM probability: (1 - lambda) * P(q_j|D) + lambda * P(q_j|Dataset_i)
            jm_prob = ((1 - lambda_value) * (f_qj_D / D_len)) + (lambda_value * (c_qj / Dataset_i_len))

            # If a query term never appears in the dataset, it contributes no log score
            if jm_prob > 0:
                Score[DocID] = Score[DocID] + (qf_j * log10(jm_prob))

    return Score


def rank_documents(Score):
    """
    Function that sorts documents by JM score in descending order.\n
    Parameters: `Score` - dictionary of JM scores `{DocID: JM_score}`\n
    Return value: ranked list of `(DocID, JM_score)` tuples
    """
    return sorted(Score.items(), key=lambda x: x[1], reverse=True)


def save_ranking(RankedList, Ti: Topic, output_dir):
    """
    Function that saves the ranked list for one topic to a `.dat` output file.\n
    Parameters:
        `RankedList` - ranked list of `(DocID, JM_score)` tuples
        `Ti` - topic object used for ranking
        `output_dir` - output folder for model ranking files\n
    Return value: path of saved ranking file
    """
    Ri = Ti.topic_id
    outname = os.path.join(output_dir, f"Baseline2_{Ri}_Ranking.dat")

    output_file = open(outname, "w")
    output_file.write(f"Query is the topic title = \"{Ti.title}\"\n\n")
    output_file.write("Doc_ID JM_Score\n")

    # Write each document ID and its JM score using the required two-column format
    for DocID, score in RankedList:
        output_file.write(f"{DocID} {score}\n")

    output_file.close()

    return outname


def baseline2(topics_file_path, doc_collection_path):
    """
    Function that runs Baseline 2 JM-smoothed likelihood ranking over all topics and datasets.\n
    Parameters:
        `topics_file_path` - path of `Topics.txt`
        `doc_collection_path` - path of folder containing Dataset101 to Dataset150\n
    Return value: dictionary `{topic_id: ranked_list}`
    """
    output_dir = "ModelOutputs"
    os.makedirs(output_dir, exist_ok=True)

    # Read all topics and stop words before looping through datasets
    Topics = load_topics(topics_file_path)
    stop_words = load_stop_words()
    all_rankings = {}

    for Ti in Topics:
        # Ri is the topic number, e.g. R101
        Ri = Ti.topic_id

        # Qi is parsed from topic title as required by the assignment specification
        Qi = parseQuery(Ti.title, stop_words)

        # Dataset101 corresponds to topic R101, Dataset102 to R102, and so on
        dataset_directory = os.path.join(doc_collection_path, f"Dataset{Ri[1:]}")

        # ParseDocuments returns Dataset_i as a BowColl of BowDoc objects
        dataset = parse_documents(dataset_directory, stop_words)

        # Build I and DocLengths for JM likelihood calculation
        I, DocLengths = build_index(dataset)

        # C stores collection frequency c_qj for every query term q_j
        C = collection_term_counts(I, Qi)

        # Compute JM Score[D] for every document D in Dataset_i
        Score = jm_likelihood(I, DocLengths, Qi, C, LAMBDA)

        # Sort documents in descending order of JM score and save the ranking
        RankedList = rank_documents(Score)
        save_ranking(RankedList, Ti, output_dir)
        all_rankings[Ri] = RankedList

        # Print top 10 documents for the appendix / checking output
        print(f"{Ri} (Doc_ID JM_Score):")
        for DocID, score in RankedList[:10]:
            print(f"{DocID} {score}")
        print("...")

    return all_rankings


def main():
    baseline2("Topics.txt", "Doc_Collection")


if __name__ == "__main__":
    main()
