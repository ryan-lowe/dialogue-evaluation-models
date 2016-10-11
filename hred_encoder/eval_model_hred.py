"""
Dialogue Evaluation Model using VHRED

This code learns to predict human scores (on a small dataset, previously collected for emnlp 2016),
using a simple linear model on top of VHRED embeddings.

Currently, all embeddings are set to 0.
"""

import numpy as np
import sys
import csv
from scipy.stats import pearsonr
import scipy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as pp
import theano
import theano.tensor as T
import time
import math
import cPickle

from dialog_encdec import DialogEncoderDecoder
from numpy_compat import argpartition
from state import prototype_state
from compute_dialogue_embeddings import compute_encodings

import os
os.sys.path.insert(0,'../TwitterData/BPE/subword_nmt') 

from apply_bpe import BPE

def load_data(filein):
    # Input: csv file name (string)
    # Output: 
    with open(filein,'r') as f1:
        data = []
        csv1 = csv.reader(f1)
        for row in csv1:
            data.append(row)
    return data

def get_score(data):
    score = []
    for row in data:
        score.append(float(row[-1]))
    return score

def preprocess_tweet(s):
    s = s.replace('@user', '<at>').replace('&lt;heart&gt;', '<heart>').replace('&lt;number&gt;', '<number>').replace('  ', ' </s> ').replace('  ', ' ')

    # Make sure we end with </s> token
    while s[-1] == ' ':
        s = s[0:-1]
    if not s[-5:] == ' </s>':
        s = s + ' </s>'

    return s

def strs_to_idxs(data, bpe, str_to_idx):
    out = []
    for row in data:
        bpe_segmented = bpe.segment(row.strip())
        # Note: there shouldn't be any unknown tokens with BPE!
        #out.append([str_to_idx[word] for word in bpe_segmented.split()])
        out.append([str_to_idx[word] for word in bpe_segmented.split() if word in str_to_idx])

    return out

def get_context(data):
    out = []
    for row in data:
        out.append('</s> ' + preprocess_tweet(row[0][5:-2]))

    return out

# TODO: Double check this indexing is correct...
def get_gtresponse(data):
    out = []
    for row in data:
        out.append(preprocess_tweet(row[1][5:-2]))

    return out

# TODO: Double check this indexing is correct...
def get_modelresponse(data):
    out = []
    for row in data:
        out.append(preprocess_tweet(row[2][5:-2]))

    return out


# Compute model embeddings for contexts or responses 
def compute_model_embeddings(data, model):
    model_compute_encoding = model.build_encoder_function()

    embeddings = []
    context_ids_batch = []
    batch_index = 0
    batch_total = int(math.ceil(float(len(data)) / float(model.bs)))
    for context_ids in data:
        context_ids_batch.append(context_ids)

        if len(context_ids_batch) == model.bs:
            batch_index = batch_index + 1

            print '     Computing embeddings for batch ' + str(batch_index) + ' / ' + str(batch_total)
            encs = compute_encodings(context_ids_batch, model, model_compute_encoding)
            for i in range(len(encs)):
                embeddings.append(encs[i])

            context_ids_batch = []

    return embeddings


def make_plot(model_scores, human_scores, filename):
    pp.plot(human_scores, model_scores, 'ko')
    pp.savefig(filename)


#####################
# Code for learning #
#####################

def set_shared_variable(x):
    return theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)

def correlation(output, score):
    """
    Returns a list of the Spearman and Pearson ranks, and the significance
    """
    cor = []
    spearman = scipy.stats.spearmanr(output, score)
    pearson = scipy.stats.pearsonr(output, score)
    cor.append(spearman)
    cor.append(pearson)
    return cor


class LinearEvalModel(object):
    """
    Class for learning two weight matrices, M and N, and bias b
    Output is sigmoid( cMr + r'Nr )

    input has shape (batch size x 3 x emb dimensionality)
    """
    def __init__(self, input, emb_dim, batch_size):
        self.M = theano.shared(np.eye(emb_dim).astype(theano.config.floatX), borrow=True)
        self.N = theano.shared(np.eye(emb_dim).astype(theano.config.floatX), borrow=True)
        
        # Set embeddings by slicing tensor
        self.emb_context = input[:,0,:]
        self.emb_response = input[:,1,:]
        self.emb_true_response = input[:,2,:]


        # Compute predictions
        self.pred1 = T.sum(self.emb_context * T.dot(self.emb_response, self.M), axis=1)
        self.pred2 = T.sum(self.emb_true_response * T.dot(self.emb_response, self.N), axis=1)
        self.pred = self.pred1 + self.pred2

        # Julian: I think adding a squared error on top of a sigmoid function will be difficult to train.
        #         Let's just try with a linear output first. We can always clip it to be within [0, 5] later.
        #self.output = 5 * T.clip(T.nnet.sigmoid(self.pred), 1e-7, 1 - 1e-7)
        self.output = 2.5 + 5 * self.pred


    def squared_error(self, score):
        return T.mean((self.output - score)**2)




def train_model(train_x, test_x, train_y, test_y, learning_rate=0.01, num_epochs=10,
        batch_size=1):
    
    print '...building model'
    n_train_batches = train_x.shape[0] / batch_size
    emb_dim = int(train_x.shape[2])
    
    train_x = set_shared_variable(train_x)
    test_x = set_shared_variable(test_x)
    train_y = set_shared_variable(train_y)
    
    index = T.lscalar()
    x = T.tensor3('x')
    y = T.fvector('y')
    
    model = LinearEvalModel(input=x, emb_dim=emb_dim, batch_size=batch_size)

    # TODO: Try out L2 regularization
    cost = model.squared_error(y)
        
    get_output = theano.function(
        inputs=[],
        outputs=model.output,
        givens={
            x: test_x
        }
    )
    
    get_pred = theano.function(
        inputs=[],
        outputs=model.pred,
        givens={
            x: test_x
        }
    )
    
    g_M = T.grad(cost=cost, wrt=model.M)
    g_N = T.grad(cost=cost, wrt=model.N)
    updates = [ (model.M, model.M - learning_rate * g_M),
                (model.N, model.N - learning_rate * g_N) ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_x[index * batch_size: (index + 1) * batch_size],
            y: train_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    print '..starting training'
    epoch = 0
    first_output = get_output()
    best_output = np.zeros((50,)) 
    best_cor = [0,0]
    best_correlation = -np.inf
    start_time = time.time()
    while (epoch < num_epochs):
        epoch += 1
        print 'Starting epoch',epoch
        for minibatch_index in xrange(n_train_batches):
            minibatch_cost = train_model(minibatch_index)
        model_out = get_output()

        test_correlation = correlation(model_out, test_y)
        print test_correlation
        if test_correlation[0][0] > best_correlation:
            best_correlation = test_correlation[0][0]
            best_cor = test_correlation
            best_output = get_output()
            with open('best_model.pkl', 'w') as f:
                cPickle.dump(model, f)

    end_time = time.time()
    print 'Finished training. Took %f s'%(end_time - start_time)
    print 'Final Spearman correlation: ', best_cor[0]
    print 'Final Peason correlation: ', best_cor[1]
    make_plot(best_output, test_y, 'correlation_vhred_learned.png')
    make_plot(first_output, test_y, 'correlation_vhred_first.png')
    make_plot(get_pred(), test_y, 'correlation_vhred_nosig.png')
    

if __name__ == '__main__':
    test_pct = 0.5

    ubuntu_file = '../ubuntu_human_data.csv'
    twitter_file = '../twitter_human_data.csv'
    #embedding_file = './hred_embeddings.pkl'
    twitter_bpe_dictionary = '../TwitterData/BPE/Twitter_Codes_5000.txt'
    twitter_bpe_separator = '@@'
    twitter_model_dictionary = '../TwitterData/BPE/Dataset.dict.pkl'

    twitter_model_prefix = '../TwitterModel/1470516214.08_TwitterModel__405001'

    # Load in Twitter dictionaries
    twitter_bpe = BPE(open(twitter_bpe_dictionary, 'r').readlines(), twitter_bpe_separator)
    twitter_dict = cPickle.load(open(twitter_model_dictionary, 'r'))
    twitter_str_to_idx = dict([(tok, tok_id) for tok, tok_id, _, _ in twitter_dict])

    # Get data, for Ubuntu
    #ubuntu_data = load_data(ubuntu_file)
    #ubuntu_human_scores = get_score(ubuntu_data)

    # Get data, for Twitter
    twitter_data = np.array(load_data(twitter_file))
    twitter_contexts = get_context(twitter_data)
    twitter_gtresponses = get_gtresponse(twitter_data)
    twitter_modelresponses = get_modelresponse(twitter_data)
    twitter_human_scores = get_score(twitter_data)

    # Encode text into BPE format
    twitter_context_ids = strs_to_idxs(twitter_contexts, twitter_bpe, twitter_str_to_idx)
    twitter_gtresponses_ids = strs_to_idxs(twitter_gtresponses, twitter_bpe, twitter_str_to_idx)
    twitter_modelresponses_ids = strs_to_idxs(twitter_modelresponses, twitter_bpe, twitter_str_to_idx)

    # Compute VHRED embeddings
    if 'gpu' in theano.config.device.lower():
        state = prototype_state()
        state_path = twitter_model_prefix + "_state.pkl"
        model_path = twitter_model_prefix + "_model.npz"

        with open(state_path) as src:
            state.update(cPickle.load(src))

        state['bs'] = 20
        state['dictionary'] = twitter_model_dictionary

        model = DialogEncoderDecoder(state) 

        print 'Computing context embeddings...'
        twitter_context_embeddings = compute_model_embeddings(twitter_context_ids, model)

        print 'Computing ground truth response embeddings...'
        twitter_gtresponses_embeddings = compute_model_embeddings(twitter_gtresponses_ids, model)

        print 'Computing model response embeddings...'
        twitter_modelresponses_embeddings = compute_model_embeddings(twitter_modelresponses_ids, model)
        
        assert len(twitter_context_embeddings) == len(twitter_gtresponses_embeddings)
        assert len(twitter_context_embeddings) == len(twitter_modelresponses_embeddings)

        emb_dim = twitter_context_embeddings[0].shape[0]

        twitter_dialogue_embeddings = np.zeros((len(twitter_context_embeddings), 3, emb_dim))
        for i in range(len(twitter_context_embeddings)):
            twitter_dialogue_embeddings[i, 0, :] =  twitter_context_embeddings[i]
            twitter_dialogue_embeddings[i, 1, :] =  twitter_gtresponses_embeddings[i]
            twitter_dialogue_embeddings[i, 2, :] =  twitter_modelresponses_embeddings[i]

    else:
        # Set embeddings to 0 for now. alternatively, we can load them from disc...
        #embeddings = cPickle.load(open(embedding_file, 'rb'))
        print 'ERROR: No GPU specified!'
        print 'ERROR: No GPU specified!'
        print 'ERROR: No GPU specified!'
        print ' To save testing time, model will be trained with zero context / response embeddings...'
        twitter_dialogue_embeddings = np.zeros((len(twitter_context_embeddings), 3, emb_dim))

    # TODO: Compute PCA projection here
    
    # Separate into training and test sets
    train_index = int((1 - test_pct) * twitter_dialogue_embeddings.shape[0])
    train_x = twitter_dialogue_embeddings[:train_index]
    test_x = twitter_dialogue_embeddings[train_index:]
    train_y = np.array(twitter_human_scores[:train_index])
    test_y = np.array(twitter_human_scores[train_index:])

    train_model(train_x, test_x, train_y, test_y)


    # Start training with:
    #   THEANO_FLAGS=mode=FAST_COMPILE,floatX=float32 python eval_model_hred.py
    # or
    #   THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python eval_model_hred.py

