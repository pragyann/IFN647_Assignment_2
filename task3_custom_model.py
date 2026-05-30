import os
from task3_baseline1 import bm25, df
from coll import BowColl, parseQuery, parse_documents, load_stop_words, rank_documents
from topics import Topic, load_topics


TOP_R = 20
BOTTOM_NR = 0
ALPHA = 20
BETA = 16
GAMMA = 0 
LAMBDA_MODEL = 0.9
THETA = 2.5
TITLE_WEIGHT = 1.0
DESC_WEIGHT = 0.2


def build_weighted_query(Ti: Topic, stop_words, title_weight, desc_weight):
    """
    Function that builds a weighted topic query from title, description, and narrative fields.\n
    Parameters:
        `Ti` - topic object
        `stop_words` - list of stop words to exclude during parsing
        `title_weight` - weight applied to title query terms
        `desc_weight` - weight applied to description query terms\n
    Return value: weighted query dictionary `{term: weight}`
    """
    # Adds terms from one topic field to Q using that field's weight
    def add_weighted_terms(parsed_terms, weight):
        for term, freq in parsed_terms.items():
            Q[term] = Q.get(term, 0) + (weight * freq)

    Q = {}

    # Parse each topic field separately before applying field weights
    TitleTerms = parseQuery(Ti.title, stop_words)
    DescTerms = parseQuery(Ti.description, stop_words)
    NarrTerms = parseQuery(Ti.narrative, stop_words)

    # Title terms are the most important, followed by description and narrative terms
    add_weighted_terms(TitleTerms, title_weight)
    add_weighted_terms(DescTerms, desc_weight)
    add_weighted_terms(NarrTerms, 0)

    return Q

def select_feedback_docs(RankedList, coll: BowColl, top_r, bottom_nr):
    """
    Function that selects pseudo-relevant and pseudo non-relevant documents from an initial ranking.\n
    Parameters:
        `RankedList` - initial ranked list of `(DocID, score)` tuples
        `coll` - collection of documents as `BowColl`
        `top_r` - number of pseudo-relevant documents
        `bottom_nr` - number of pseudo non-relevant documents\n
    Return value: tuple `(D_plus, D_minus)` containing selected document objects
    """
    # D_plus contains the top ranked documents
    D_plus_ids = [DocID for DocID, _ in RankedList[:top_r]]

    # D_minus contains the bottom ranked documents
    D_minus_ids = [DocID for DocID, _ in RankedList[-bottom_nr:]] if bottom_nr > 0 else []

    D_plus = [coll.get_doc(DocID) for DocID in D_plus_ids]
    D_minus = [coll.get_doc(DocID) for DocID in D_minus_ids]

    return D_plus, D_minus


def rocchio_query_update(Q, D_plus, D_minus, alpha, beta, gamma):
    """
    Function that updates query weights using pseudo-relevance feedback.\n
    Parameters:
        `Q` - original weighted topic query `{term: weight}`
        `D_plus` - pseudo-relevant document list
        `D_minus` - pseudo non-relevant document list
        `alpha` - original query weight
        `beta` - pseudo-relevant feedback weight
        `gamma` - pseudo non-relevant feedback weight\n
    Return value: Rocchio-modified query dictionary `{term: weight}`
    """
    Qm = {}
    R = len(D_plus)
    NR = len(D_minus)

    for t in Q:
        # Average evidence for term t in pseudo-relevant and pseudo non-relevant documents
        relWeight = 0
        nonrelWeight = 0

        for D in D_plus:
            if D.get_doc_len() > 0:
                relWeight = relWeight + D.get_term_count(t)

        for D in D_minus:
            if D.get_doc_len() > 0:
                nonrelWeight = nonrelWeight + D.get_term_count(t)

        relMean = relWeight / R if R > 0 else 0
        nonrelMean = nonrelWeight / NR if NR > 0 else 0

        # Rocchio query modification
        Qm[t] = (alpha * Q[t]) + (beta * relMean) - (gamma * nonrelMean)

    return Qm


def pseudo_relevance_counts(D_plus):
    """
    Function that counts how many pseudo-relevant documents contain each term.\n
    Parameters: `D_plus` - pseudo-relevant document list\n
    Return value: dictionary `{term: r_t}`
    """
    r = {}

    for D in D_plus:
        for t in D.terms:
            r[t] = r.get(t, 0) + 1

    return r


def candidate_terms(Qm, D_plus):
    """
    Function that builds candidate feature terms from the modified query and pseudo-relevant documents.\n
    Parameters:
        `Qm` - Rocchio-modified query dictionary `{term: weight}`
        `D_plus` - pseudo-relevant document list\n
    Return value: set of candidate terms
    """
    T = set()

    # Add terms from the modified query
    for t in Qm:
        T.add(t)

    # Add terms from pseudo-relevant documents
    for D in D_plus:
        for t in D.terms:
            T.add(t)

    return T


def w5_feedback_weights(T, Qm, term_df, r, N, R, lambda_model):
    """
    Function that computes final feature weights using feedback and modified-query evidence.\n
    Parameters:
        `T` - candidate feature terms
        `Qm` - Rocchio-modified query dictionary `{term: weight}`
        `term_df` - document frequency dictionary `{term: n_t}`
        `r` - pseudo-relevant document count dictionary `{term: r_t}`
        `N` - number of documents in the dataset
        `R` - number of pseudo-relevant documents
        `lambda_model` - weighting factor for Rocchio evidence\n
    Return value: dictionary `{term: W5_weight}`
    """
    W5 = {}

    for t in T:
        # r_t = number of pseudo-relevant documents containing term t
        r_t = r.get(t, 0)

        # n_t = number of documents in Dataset_i containing term t
        n_t = term_df.get(t, 0)

        # Relevance-feedback evidence for term t
        numerator = (r_t + 0.5) / (R - r_t + 0.5)
        denominator = (n_t - r_t + 0.5) / (N - n_t - R + r_t + 0.5)
        w5 = numerator / denominator
        
        # Combine Rocchio-modified query evidence and w5 feedback evidence
        rocchio_weight = Qm.get(t, 0)
        W5[t] = (lambda_model * rocchio_weight) + ((1 - lambda_model) * w5)
    return W5

def select_features(W5, theta):
    """
    Function that keeps feature terms whose learned weights are above the selection threshold.\n
    Parameters: `W5` - final feature weights `{term: weight}`, `theta` - threshold offset\n
    Return value: selected feature dictionary `{term: weight}`
    """
    Features = {}

    if len(W5) == 0:
        return Features

    meanW5 = sum(W5.values()) / len(W5)
    threshold = theta + meanW5

    for t, weight in W5.items():
        if weight > threshold:
            Features[t] = weight

    return Features


def feature_score(coll: BowColl, Features):
    """
    Function that scores documents using selected binary features and learned feature weights.\n
    Parameters:
        `coll` - collection of documents as `BowColl`
        `Features` - selected feature dictionary `{term: weight}`\n
    Return value: dictionary of final Model_C scores `{DocID: score}`
    """
    Score = {}

    for DocID, D in coll.get_docs().items():
        Score[DocID] = 0.0

        for t, weight in Features.items():
            # I(t, D) = 1 if term t occurs in document D, otherwise 0
            if D.get_term_count(t) > 0:
                Score[DocID] = Score[DocID] + weight

    return Score


def save_ranking(RankedList, Ti: Topic, output_dir):
    """
    Function that saves the ranked list for one topic to a `.dat` output file.\n
    Parameters:
        `RankedList` - ranked list of `(DocID, ModelC_score)` tuples
        `Ti` - topic object used for ranking
        `output_dir` - output folder for model ranking files\n
    Return value: path of saved ranking file
    """
    Ri = Ti.topic_id
    outname = os.path.join(output_dir, f"ModelC_{Ri}_Ranking.dat")

    output_file = open(outname, "w")
    output_file.write(f"Query is the topic title = \"{Ti.title}\"\n\n")
    output_file.write("Doc_ID ModelC_Score\n")

    # Write each document ID and its final score using the required two-column format
    for DocID, score in RankedList:
        output_file.write(f"{DocID} {score}\n")

    output_file.close()

    return outname


def modelC(
    topics_file_path,
    doc_collection_path,
    top_r=TOP_R,
    bottom_nr=BOTTOM_NR,
    alpha=ALPHA,
    beta=BETA,
    gamma=GAMMA,
    lambda_model=LAMBDA_MODEL,
    theta=THETA,
    title_weight=TITLE_WEIGHT,
    desc_weight=DESC_WEIGHT,
):
    """
    Function that runs Model_C ranking over all topics and datasets.\n
    Parameters:
        `topics_file_path` - path of `Topics.txt`
        `doc_collection_path` - path of folder containing Dataset101 to Dataset150
        `top_r` - number of pseudo-relevant documents
        `bottom_nr` - number of pseudo non-relevant documents
        `alpha` - original query weight
        `beta` - pseudo-relevant feedback weight
        `gamma` - pseudo non-relevant feedback weight
        `lambda_model` - weighting factor for Rocchio evidence
        `theta` - feature selection threshold offset
        `title_weight` - weight applied to title query terms of the topic
        `desc_weight` - weight applied to description query terms of the topic\n
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

        dataset_directory = os.path.join(doc_collection_path, f"Dataset{Ri[1:]}")
        dataset = parse_documents(dataset_directory, stop_words)

        # Q is built from title, description, and narrative fields
        Q = build_weighted_query(Ti, stop_words, title_weight, desc_weight)

        # Dataset101 corresponds to topic R101, Dataset102 to R102, and so on

        # ParseDocuments returns Dataset_i as a BowColl of BowDoc objects
        N = dataset.get_num_docs()

        # Get document frequency n_t for every term in Dataset_i
        term_df = df(dataset)

        # STEP 3: Initial ranking using BM25
        # First-pass BM25 ranking for pseudo-relevance feedback
        InitialScore = bm25(dataset, Q, term_df)
        InitialRankedList = rank_documents(InitialScore)

        # Select pseudo-relevant and pseudo non-relevant documents
        D_plus, D_minus = select_feedback_docs(InitialRankedList, dataset, top_r, bottom_nr)
        R = len(D_plus)

        # Update the query using Rocchio pseudo-relevance feedback
        Qm = rocchio_query_update(Q, D_plus, D_minus, alpha, beta, gamma)

        # Compute feedback counts and final feature weights
        r = pseudo_relevance_counts(D_plus)
        T = candidate_terms(Qm, D_plus)
        W5 = w5_feedback_weights(T, Qm, term_df, r, N, R, lambda_model)

        # Final set of features
        Features = select_features(W5, theta)

        # Final ranking based on selected features and learned weights
        Score = feature_score(dataset, Features)
        RankedList = rank_documents(Score)
        save_ranking(RankedList, Ti, output_dir)
        all_rankings[Ri] = RankedList

        # Print top 10 documents for the appendix / checking output
        print(f"{Ri} (Doc_ID ModelC_Score):")
        for DocID, score in RankedList[:10]:
            print(f"{DocID} {score}")
        print("...")

    return all_rankings

def main():
    modelC("data/Topics.txt", "data/Doc_Collection")

if __name__ == "__main__":
    main()
