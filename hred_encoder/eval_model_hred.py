"""
Dialogue Evaluation Model using hred

This code learns to predict human scores (on a small dataset, previously collected for emnlp 2016),
using a simple linear model on top of hred embeddings.

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
import cPickle

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

def make_plot(model_scores, human_scores, filename):
    pp.plot(human_scores, model_scores, 'ko')
    pp.savefig(filename)


#####################
# Code for learning #
#####################

def set_shared_variable(x):
    return theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)

def get_emb(embedding):
    """                         IS THIS RIGHT????????????????/
    Input: unordered list of embeddings of the form
            [context?, candidate resp, true resp]
    Output: lists for context, candidate resp, true resp
    """
    emb_dim = embedding.shape[1]
    emb_num = embedding.shape[0] / 3
    emb_array = np.array(embedding)
    emb_tensor = emb_array.reshape([emb_dim,3,emb_num])
    emb_tensor = np.transpose(emb_tensor, (2,1,0))
    return emb_tensor

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
        self.output = 5 * T.clip(T.nnet.sigmoid(self.pred), 1e-7, 1 - 1e-7)


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
        test_correlation = correlation(get_output(), test_y)
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
    make_plot(best_output, test_y, 'correlation_hred_learned.png')
    make_plot(first_output, test_y, 'correlation_hred_first.png')
    make_plot(get_pred(), test_y, 'correlation_hred_nosig.png')
    

if __name__ == '__main__':
    test_pct = 0.5

    ubuntu_file = '../ubuntu_human_data.csv'
    twitter_file = '../twitter_human_data.csv'
    embedding_file = './hred_embeddings.pkl'

    # Get data, separate into twitter and ubuntu
    ubuntu_data = load_data(ubuntu_file)
    twitter_data = np.array(load_data(twitter_file))
    ubuntu_human_scores = get_score(ubuntu_data)
    twitter_human_scores = get_score(twitter_data)
    #embeddings = cPickle.load(open(embedding_file, 'rb'))
   
    # set embeddings to 0 for now
    twitter_embeddings = np.zeros((300,500)) # there are 300 twitter contexts / responses
     

    # Construct embedding tensor
    emb_tensor = get_emb(twitter_embeddings)
    
    # Separate into training and test sets
    train_index = int((1 - test_pct) * emb_tensor.shape[0])
    train_x = emb_tensor[:train_index]
    test_x = emb_tensor[train_index:]
    train_y = np.array(twitter_human_scores[:train_index])
    test_y = np.array(twitter_human_scores[train_index:])
    
    train_model(train_x, test_x, train_y, test_y)


        

    





