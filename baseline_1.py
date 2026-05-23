import os
from math import log10

from coll import BowColl, parse_documents, parseQuery
from topics import Topic, load_topics




def load_stop_words():
    """
    Function that loads stop words from a comma-separated stop words file.\n
    Parameters: `path` - path of stop words file to load\n
    Return value: list of stop words
    """

    stop_words_file = open("common-english-words.txt", "r")
    stop_words = stop_words_file.read().split(",")
    stop_words_file.close()

    return stop_words


def df(coll: BowColl):
    """
    Function that takes a document collection and returns document frequency of each term in the collection.\n
    Parameters: `coll` - collection of documents as `BowColl`\n
    Return value: document frequency dictionary `{term: df}`
    """
    term_df = {}

    # Count each term once per document to compute n_t for BM25
    for doc in coll.get_docs().values():
        for term in doc.terms:
            term_df[term] = term_df.get(term, 0) + 1

    return term_df


def avg_doc_len(coll: BowColl):
    """
    Function that computes the average document length in the given collection.\n
    Parameters: `coll` - collection of documents as `BowColl`\n
    Return value: average document length `avdl`
    """
    total_dataset_length = 0

    # Add each document length to compute total length of Dataset_i
    for doc in coll.get_docs().values():
        total_dataset_length = total_dataset_length + doc.get_doc_len()

    return total_dataset_length / coll.get_num_docs()


def bm25(coll: BowColl, Qi, term_df):
    """
    Function that computes BM25 scores for all documents in one dataset.\n
    Parameters:
        `coll` - collection of documents as `BowColl`
        `Qi` - parsed topic title query dictionary `{term: qf_t}`
        `term_df` - document frequency dictionary `{term: n_t}`\n
    Return value: dictionary of BM25 scores `{DocID: BM25_score}`
    """
    Score = {}
    N = coll.get_num_docs()

    K1 = 1.2
    K2 = 500
    B = 0.75

    avdl = avg_doc_len(coll)

    # Initialize score of every document to 0 before accumulating term scores
    for DocID in coll.get_docs():
        Score[DocID] = 0.0

    for t, qf_t in Qi.items():
        # n_t = number of documents in Dataset_i containing term t
        n_t = term_df.get(t, 0)

        # If the query term does not appear in this dataset, it contributes no score
        if n_t == 0:
            continue

        # First BM25 component: inverse document frequency for term t
        idf_t = log10(1 + ((N - n_t + 0.5) / (n_t + 0.5)))

        for DocID, D in coll.get_docs().items():
            # f_t = number of times term t occurs in document D
            f_t = D.get_term_count(t)

            # If term t does not occur in D, the document term-frequency component is 0
            if f_t == 0:
                continue

            # dl = length of document D and K is the BM25 length-normalization value
            dl = D.get_doc_len()
            K = K1 * ((1 - B) + B * (dl / avdl))

            # Second BM25 component: normalized term frequency in document D
            doc_tf = ((K1 + 1) * f_t) / (K + f_t)

            # Third BM25 component: normalized query term frequency in Qi
            query_tf = ((K2 + 1) * qf_t) / (K2 + qf_t)

            Score[DocID] = Score[DocID] + (idf_t * doc_tf * query_tf)

    return Score


def rank_documents(Score):
    """
    Function that sorts documents by BM25 score in descending order.\n
    Parameters: `Score` - dictionary of BM25 scores `{DocID: BM25_score}`\n
    Return value: ranked list of `(DocID, BM25_score)` tuples
    """
    return sorted(Score.items(), key=lambda x: x[1], reverse=True)


def save_ranking(RankedList, Ti: Topic, output_dir):
    """
    Function that saves the ranked list for one topic to a `.dat` output file.\n
    Parameters:
        `RankedList` - ranked list of `(DocID, BM25_score)` tuples
        `Ti` - topic object used for ranking
        `output_dir` - output folder for model ranking files\n
    Return value: path of saved ranking file
    """
    Ri = Ti.topic_id
    outname = os.path.join(output_dir, f"Baseline1_{Ri}_Ranking.dat")

    output_file = open(outname, "w")
    output_file.write(f"Query is the topic title = \"{Ti.title}\"\n\n")
    output_file.write("Doc_ID BM25_Score\n")

    # Write each document ID and its BM25 score using the required two-column format
    for DocID, score in RankedList:
        output_file.write(f"{DocID} {score}\n")

    output_file.close()

    return outname


def baseline1(topics_file_path, doc_collection_path):
    """
    Function that runs Baseline 1 BM25 ranking over all topics and datasets.\n
    Parameters:
        `topics_file_path` - path of `Topics.txt`
        `doc_collection_path` - path of folder containing Dataset101 to Dataset150
        `output_dir` - folder where ranking output files are saved
        `stop_words_path` - path of stop words file\n
    Return value: dictionary `{topic_id: ranked_list}`
    """
    output_dir = 'ModelOutputs'
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

        # Get document frequency n_t for every term in Dataset_i
        term_df = df(dataset)

        # Compute BM25 Score[D] for every document D in Dataset_i
        Score = bm25(dataset, Qi, term_df)

        # Sort documents in descending order of BM25 score and save the ranking
        RankedList = rank_documents(Score)
        save_ranking(RankedList, Ti, output_dir)
        all_rankings[Ri] = RankedList

        # Print top 10 documents for the appendix / checking output
        print(f"{Ri} (Doc_ID BM25_Score):")
        for DocID, score in RankedList[:10]:
            print(f"{DocID} {score}")
        print("...")

    return all_rankings


def main():
    baseline1("Topics.txt", "Doc_Collection")


if __name__ == "__main__":
    main()
