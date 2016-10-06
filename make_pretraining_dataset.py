"""
Code for making the pre-training dataset for Twitter dialogues.

(to be completed)
"""
import numpy as np


def load_data():
    pass

def encode_tweets():
    pass


def calculate_score(e_cont, e_resp, e_propcont, e_propresp):
    context_term = np.dot(e_cont, e_propcont) / (np.norm(e_cont) * np.norm(e_propcont)) # cosine similarity
    response_term = np.dot(e_resp, e_propresp) / (np.norm(e_resp) * np.norm(e_propresp))
    
    return context_term + 0.2 * response_term  # TODO: come up with better formula

def get_pos_indices(scores, upperbound, lowerbound):
    pos_indices = []
    neg_indices = []
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if scores[i][j] < lowerbound:
                neg_indices.append((i,j))
                neg_indices.append((j, i))
            if scores[i][j] > upperbound:
                pos_indices.append((i,j))
                pos_indices.append((j, i))
    return pos_indices, neg_indices

def get_mid_indices(scores):
    pass # TODO: write this

def add_to_dataset(dataset, indices, data):
    for i, j in zip(indices):
        dataset.append([]) # TODO: write this. depends on how data comes in

def get_dataset(scores, data):
    # Input: matrix of scores
    # Ouptut: list of the form [context, true_response, proposed_response, score]
    dataset = []
    pos_indices, neg_indices = get_pos_indices(scores)  # calculate pairs of indices for pos/neg examples
    mid_indices = get_mid_indices(scores)
    add_to_dataset(dataset, pos_indices)
    add_to_dataset(dataset, neg_indices)
    add_to_dataset(dataset, mid_indices)
    return dataset


def create_dataset():
    data = load_data()
    e_context, e_response = encode_tweets(data)
    dataset = []
    # scores is a matrix that gives the score for some candidate response (col) being the response
    # instead of the true response (row)
    scores = np.zeros(len(e_context))
    
    for i in range(len(e_context)):
        for j in range(i + 1, len(e_context)): # scores are symmetric so only calculate once
            scores[i][j] = calculate_score(e_context[i], e_response[i], e_context[j], e_response[j])
                
    dataset = get_dataset(scores)

