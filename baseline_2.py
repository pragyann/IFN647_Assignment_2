import glob
import os
import string
from math import log10

from coll import parseQuery, load_stop_words
from stemming.porter2 import stem
from topics import Topic, load_topics

LAMBDA = 0.3


def index_docs(inputpath, stop_words):
    """
    Function that builds an inverted index directly from XML files in a dataset folder.\n
    Parameters: `inputpath` - path of dataset folder, `stop_words` - list of stop words to exclude\n
    Return value: inverted index `{term: {DocID: frequency}}`
    """
    Index = {}

    for file_ in glob.glob(f"{inputpath}/*.xml"):
        docid = None
        start_end = False

        for line in open(file_, "r", encoding="latin-1", errors="ignore"):
            line = line.strip()

            if start_end == False:
                # Extract document ID from itemid in the <newsitem> tag
                if line.startswith("<newsitem "):
                    for part in line.split():
                        if part.startswith("itemid="):
                            docid = part.split("=")[1].split("\"")[1]
                            break

                # Start indexing only after the <text> tag
                if line.startswith("<text>"):
                    start_end = True

            elif line.startswith("</text>"):
                # Stop reading terms after the document text ends
                break

            elif docid is not None:
                # Clean paragraph tags, digits, and punctuation before term extraction
                line = line.replace("<p>", "").replace("</p>", "")
                line = line.translate(str.maketrans("", "", string.digits)).translate(
                    str.maketrans(string.punctuation, " " * len(string.punctuation))
                )

                for term in line.split():
                    # Apply stemming, length check, and stop-word removal
                    term = stem(term.lower())
                    if len(term) > 2 and term not in stop_words:
                        if term not in Index:
                            Index[term] = {}
                        Index[term][docid] = Index[term].get(docid, 0) + 1

    return Index


def collection_len(I):
    """
    Function that computes total number of word occurrences in a dataset.\n
    Parameters: `I` - inverted index `{term: {DocID: frequency}}`\n
    Return value: total number of word occurrences `|Dataset_i|`
    """
    total_dataset_length = 0

    # Count collection length by summing all term frequencies in the index
    for posting_list in I.values():
        for freq in posting_list.values():
            total_dataset_length = total_dataset_length + freq

    return total_dataset_length


def collection_term_counts(I, Q):
    """
    Function that computes collection frequency for each query term.\n
    Parameters:
        `I` - inverted index `{term: {DocID: frequency}}`
        `Q` - parsed topic title query dictionary `{term: frequency}`\n
    Return value: dictionary `{q_j: c_qj}`
    """
    CF = {}

    for q_j in Q:
        # CF[q_j] = total number of times query term q_j occurs in Dataset_i
        CF[q_j] = 0
        if q_j in I:
            CF[q_j] = sum(I[q_j].values())

    return CF


def likelihood_JM(I, Q, lamda=LAMBDA):
    """
    Function that computes JM-smoothed log likelihood scores.\n
    Parameters:
        `I` - single inverted index `{term: {DocID: frequency}}` used as both collection and document index
        `Q` - parsed topic title query dictionary `{term: frequency}`
        `lamda` - JM smoothing parameter\n
    Return value: dictionary of JM likelihood scores `{DocID: JM_score}`
    """
    L = {} # L is the selected inverted list
    R = {} # R is a dictionary of docId:score
    D_len = {}

    # Get all document IDs, initialize scores as 0, and select query posting lists
    for term, posting_list in I.items():
        for DocID in posting_list:
            R[DocID] = 0.0
            D_len[DocID] = 0.5 # initialize a small non-zero value as it will be used as denominator

        if term in Q:  # select inverted lists based on the query
            L[term] = posting_list

    # Add empty posting lists for query terms that do not occur in this dataset
    for q_term in Q:
        if q_term not in L:
            L[q_term] = {}

    # Count document length from the index
    for posting_list in I.values():
        for DocID, freq in posting_list.items():
            D_len[DocID] = D_len[DocID] + freq

    # Compute collection frequency CF[q_j] for each query term from the same single index
    CF = collection_term_counts(I, Q)
    C_len = collection_len(I)

    if C_len == 0:
        return R

    # JM log likelihood:
    # score(D) = sum over query terms of log10((1-lamda) * f_qj,D / |D| + lamda * CF[q_j] / |C|)
    for DocID, sd in R.items():
        for term, posting_list in L.items():
            f_qj_D = posting_list.get(DocID, 0)
            jm_prob = ((1 - lamda) * f_qj_D / D_len[DocID]) + (lamda * CF[term] / C_len)

            # Only positive probabilities have a finite logarithm
            if jm_prob > 0:
                sd = sd + (log10(jm_prob))

        R[DocID] = sd

    return R


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

        # Qi is parsed from the topic title
        Qi = parseQuery(Ti.title, stop_words)

        # Dataset101 corresponds to topic R101, Dataset102 to R102, and so on
        dataset_directory = os.path.join(doc_collection_path, f"Dataset{Ri[1:]}")

        # Build I directly from XML files
        I = index_docs(dataset_directory, stop_words)

        # Compute JM Score[D] for every document D in Dataset_i
        Score = likelihood_JM(I, Qi, LAMBDA)

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
