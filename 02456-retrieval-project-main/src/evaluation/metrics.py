import math

from src.data_utils.query import Query

def calculateRecall(relevancies: list[bool], query: Query):
    return sum(relevancies) / min(len(relevancies), query.getNumberOfRelevantDocuments())

def Evaluate(relevancies: list[list[bool]], queries: list[Query]):
    assert len(relevancies) == len(queries), 'The number of relevant passages must match the number of queries'
    scores = {'recall': None}
    n_queries = len(queries)
    total_recall = 0
    
    for i in range(n_queries):
        total_recall += calculateRecall(relevancies[i], queries[i])
    
    scores['recall'] = total_recall / n_queries
    return scores


# Precision of the P@k metric
def Pk(relevancies: list[bool], k: int):
    if len(relevancies) < k:
        raise ValueError(f"Length of relevancies ({len(relevancies)}) is less than k ({k})")
    
    relevant_count = sum(relevancies[:k])
    return relevant_count / k


# Recall of the R@k metric
def Rk(relevancies: list[bool], k: int, query: Query):
    if len(relevancies) < k:
        raise ValueError(f"Length of relevancies ({len(relevancies)}) is less than k ({k})")
    
    relevant_count = sum(relevancies[:k])
    return relevant_count / min(k, query.getNumberOfRelevantDocuments())

# Mean Reciprocal Rank (MRR) of the MRR@k metric
def MRR(relevancies: list[bool], k: int):
    if len(relevancies) < k:
        raise ValueError(f"Length of relevancies ({len(relevancies)}) is less than k ({k})")
    
    for i in range(k):
        if relevancies[i]:
            return 1 / (i + 1)
    return 0

# Mean Average Precision (MAP) of the MAP@k metric
def MAP(relevancies: list[bool], k: int):
    if len(relevancies) < k:
        raise ValueError(f"Length of relevancies ({len(relevancies)}) is less than k ({k})")
    
    relevant_count = 0
    average_precision = 0
    
    for i in range(k):
        if relevancies[i]:
            relevant_count += 1
            average_precision += relevant_count / (i + 1)
    
    if relevant_count == 0:
        return 0
    
    return average_precision / relevant_count

# Normalized Discounted Cumulative Gain (NDCG) of the NDCG@k metric
def NDCG(relevancies: list[bool], k: int):
    if len(relevancies) < k:
        raise ValueError(f"Length of relevancies ({len(relevancies)}) is less than k ({k})")
    
    dcg = 0
    for i in range(k):
        if relevancies[i]:
            dcg += 1 / (math.log2(i + 2))  # log2(i + 2) because i is 0-indexed
    
    # Calculate ideal DCG
    ideal_relevancies = sorted(relevancies[:k], reverse=True)
    idcg = 0
    for i in range(k):
        if ideal_relevancies[i]:
            idcg += 1 / (math.log2(i + 2))
    
    if idcg == 0:
        return 0
    
    return dcg / idcg




# Evaluate all metrics for a given set of relevancies and queries
def evaluate_metrics(relevancies: list[list[bool]], queries: list[Query], k: int):
    assert len(relevancies) == len(queries), 'The number of relevant passages must match the number of queries'
    scores = {'P@k': [], 'R@k': [], 'MRR@k': [], 'MAP@k': [], 'NDCG@k': []}
    n_queries = len(queries)
    
    for i in range(n_queries):
        scores['P@k'].append(Pk(relevancies[i], k))
        scores['R@k'].append(Rk(relevancies[i], k, queries[i]))
        scores['MRR@k'].append(MRR(relevancies[i], k))
        scores['MAP@k'].append(MAP(relevancies[i], k))
        scores['NDCG@k'].append(NDCG(relevancies[i], k))
    
    # Calculate mean scores
    mean_scores = {metric: sum(scores[metric]) / n_queries for metric in scores}
    
    return mean_scores


# Return a summary of metrics and their scores
# Input: relevancies, queries, k
# Output: summary of metrics and their scores
def metrics_summary(relevancies: list[list[bool]], queries: list[Query], k: int):
    mean_scores = evaluate_metrics(relevancies, queries, k)
    summary = f"Metrics Summary (k={k}):\n"
    for metric, score in mean_scores.items():
        summary += f"{metric}: {score:.4f}\n"
    return summary

# return a formatted table of metrics and their scores
# Input: relevancies, queries, k
# Output: formatted table of metrics and their scores
def metrics_table(relevancies: list[list[bool]], queries: list[Query], k: int):
    mean_scores = evaluate_metrics(relevancies, queries, k)
    table = f"{'Metric':<10} | {'Score':<10}\n"
    table += "-" * 25 + "\n"
    for metric, score in mean_scores.items():
        table += f"{metric:<10} | {score:<10.4f}\n"
    return table

# Compare two sets of metrics and return a formatted comparison
# Input: relevancies1, queries1, relevancies2, queries2, k
# Output: formatted comparison of metrics
def compare_metrics(relevancies1: list[list[bool]], queries1: list[Query], relevancies2: list[list[bool]], queries2: list[Query], k: int):
    mean_scores1 = evaluate_metrics(relevancies1, queries1, k)
    mean_scores2 = evaluate_metrics(relevancies2, queries2, k)
    
    comparison = f"Metrics Comparison (k={k}):\n"
    comparison += f"{'Metric':<10} | {'Score 1':<10} | {'Score 2':<10}\n"
    comparison += "-" * 40 + "\n"
    for metric in mean_scores1:
        comparison += f"{metric:<10} | {mean_scores1[metric]:<10.4f} | {mean_scores2[metric]:<10.4f}\n"
    return comparison