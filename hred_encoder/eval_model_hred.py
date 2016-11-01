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
from sklearn.decomposition import PCA
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from random import randint

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
    ''' Encodes strings in BPE form '''
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
       # TODO: this also puts the </s> token at the beginning of every context... 
       # is that what we want?

    return out

def get_gtresponse(data):
    out = []
    for row in data:
        out.append(preprocess_tweet(row[2][5:-2]))

    return out

def get_modelresponse(data):
    out = []
    for row in data:
        out.append(preprocess_tweet(row[1][5:-2]))
    return out

def get_twitter_data(clean_data_file, context_file, gt_file):
    '''
    Loads Twitter data from dictionaries.
    '''
    with open(clean_data_file, 'r') as f1:
        clean_data = cPickle.load(f1)
    with open(context_file, 'r') as f1:
        contexts = cPickle.load(f1)
    gt_unordered = []
    with open(gt_file, 'r') as f1:
        for row in f1:
            gt_unordered.append(row)
    
    # Retrieve scores and valid context ids from clean_data.pkl
    score_dic = {}
    for user in clean_data:
        for dic in clean_data[user]:
            if int(dic['c_id']) >= 0:
                score_dic[dic['c_id']] = [dic['overall1'], dic['overall2'], dic['overall3'], dic['overall4']]

    context_list = []
    gt_responses = []
    model_responses = []
    scores = []

    # Retrieve contexts and model responses from contexts.pkl
    for c in contexts:
        if int(c[0]) in score_dic:
            context_list.append(c[1])
            model_responses.append(c[2:6])
            scores.append(score_dic[int(c[0])])
            gt_responses.append(gt_unordered[int(c[0])])
    model_responses = [i for sublist in model_responses for i in sublist] # flatten list
    scores = [float(i) for sublist in scores for i in sublist] # flatten list
    
    return context_list, gt_responses, model_responses, scores

# Compute mean and range of dot products to find right constants in model
def compute_init_values(emb):
    prod_list = []
    for i in xrange(len(emb[0][0])):
        prod_list.append(np.dot(emb[i, 0], emb[i, 2]) + np.dot(emb[i, 1], emb[i, 2]))
    return sum(prod_list) / float(len(prod_list)), max(prod_list) - min(prod_list)


# Prints BLEU, METEOR, etc. correlation scores on the test set
def show_overlap_scores(twitter_gtresponses, twitter_modelresponses, twitter_human_scores, test_pct):
    # Align ground truth with model responses
    temp_gt = []
    for i in xrange(len(twitter_gtresponses)):
        temp_gt.append([twitter_gtresponses[i]]*4)
    twitter_gtresponses = [i for sublist in temp_gt for i in sublist]

    assert len(twitter_modelresponses) == len(twitter_gtresponses)
    assert len(twitter_modelresponses) == len(twitter_human_scores)
   
    test_index = 0 # If you want to evaluate on the whole dataset
    #test_index = int( (1 - test_pct) * len(twitter_modelresponses) )
    test_gtresponses = twitter_gtresponses[test_index:]
    test_modelresponses = twitter_modelresponses[test_index:]
    test_scores = twitter_human_scores[test_index:]

    bleu1_list = []
    bleu2_list = []
    bleu3_list = []
    bleu4_list = []
    rouge_list = []
    meteor_list = []
    print Meteor()._score(test_gtresponses[0], test_modelresponses[0])
    for i in xrange(len(test_modelresponses)):
        dict_input = {0: [test_gtresponses[i]]}, {0: [test_modelresponses[i]]}
        bleu1_list.append(Bleu(1).compute_score(dict_input)[0][0])
        bleu2_list.append(Bleu(2).compute_score(dict_input)[0][1])
        bleu3_list.append(Bleu(3).compute_score(dict_input)[0][2])
        bleu4_list.append(Bleu(4).compute_score(dict_input)[0][3])
        rouge_list.append(Rouge().compute_score(dict_input)[0])
        meteor_list.append(Meteor().compute_score(dict_input))

    metric_list = [bleu1_list, bleu2_list, bleu3_list, bleu4_list, rouge_list, meteor_list]
    metric_name = ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'rouge', 'meteor']
    for metric, name in zip(metric_list, metric_name):
        spearman = scipy.stats.spearmanr(metric, test_scores)
        pearson = scipy.stats.pearsonr(metric, test_scores)
        print 'For ' + name + ' score:'
        print spearman
        print pearson

# Computes PCA decomposition for the context, gt responses, and model responses separately
def compute_separate_pca(pca_components, twitter_dialogue_embeddings):
    pca = PCA(n_components = pca_components)
    tw_embeddings_pca = np.zeros((twitter_dialogue_embeddings.shape[0], 3, pca_components))
    for i in range(3):
        tw_embeddings_pca[:,i] = pca.fit_transform(twitter_dialogue_embeddings[:, i])
    return tw_embeddings_pca

# Computes PCA decomposition for the context, gt responses, and model responses together
# NOTE: this computes the PCA on the training embeddings, and then applies them to the
# test embeddings (it does not compute PCA on the testing embeddings)
def compute_pca(pca_components, twitter_dialogue_embeddings, train_index):
    pca = PCA(n_components = pca_components)
    tw_nonpca_train = twitter_dialogue_embeddings[:train_index]
    tw_nonpca_test = twitter_dialogue_embeddings[train_index:]

    num_ex_train = tw_nonpca_train.shape[0]
    num_ex_test = tw_nonpca_test.shape[0]
    dim = twitter_dialogue_embeddings.shape[2]
    tw_embeddings_pca_train = np.zeros((num_ex_train * 3, dim))
    tw_embeddings_pca_test = np.zeros((num_ex_test * 3, dim))
    for i in range(3):
        tw_embeddings_pca_train[num_ex_train*i: num_ex_train*(i+1),:] = tw_nonpca_train[:,i]
        tw_embeddings_pca_test[num_ex_test*i: num_ex_test*(i+1),:] = tw_nonpca_test[:,i]
    tw_embeddings_pca_train = pca.fit_transform(tw_embeddings_pca_train)
    tw_embeddings_pca_test = pca.transform(tw_embeddings_pca_test)
    tw_emb_train = np.zeros((num_ex_train, 3, pca_components))
    tw_emb_test = np.zeros((num_ex_test, 3, pca_components))
    for i in range(3):
        tw_emb_train[:,i] = tw_embeddings_pca_train[num_ex_train*i: num_ex_train*(i+1),:]
        tw_emb_test[:,i] = tw_embeddings_pca_test[num_ex_test*i: num_ex_test*(i+1),:]
    return tw_emb_train, tw_emb_test


# Compute model embeddings for contexts or responses 
# Embedding type can be 'CONTEXT' or 'DECODER'
# NOTE: vhred_retrieval.py has code for saving the embeddings
# from the whole Twitter dataset into .pkl files
def compute_model_embeddings(data, model, embedding_type):
    model_compute_encoding = model.build_encoder_function()
    model_compute_decoder_encoding = model.build_decoder_encoding()
    model.bs = 20
    print model.bs
    embeddings = []
    context_ids_batch = []
    batch_index = 0
    batch_total = int(math.ceil(float(len(data)) / float(model.bs)))
    counter = 0 
    for context_ids in data:
        counter += 1
        context_ids_batch.append(context_ids)

        if len(context_ids_batch) == model.bs:# or counter == len(data):
            batch_index += 1
            #if counter == len(data):
            #    model.bs = counter % model.bs
            print '     Computing embeddings for batch ' + str(batch_index) + ' / ' + str(batch_total)
            encs = compute_encodings(context_ids_batch, model, model_compute_encoding, model_compute_decoder_encoding, embedding_type)
            for i in range(len(encs)):
                embeddings.append(encs[i])

            context_ids_batch = []

    return embeddings

def get_auxiliary_features(contexts, gtresponses, modelresponses, num_examples):
    aux_features = np.zeros((num_examples, 5))
    bleu1 = []
    bleu2 = []
    bleu3 = []
    bleu4 = []
    meteor = []
    rouge = []
    for i in xrange(num_examples):
        bleu1.append(Bleu(1).compute_score({0: [gtresponses[i]]}, {0: [modelresponses[i]]})[0][0])
        bleu2.append(Bleu(2).compute_score({0: [gtresponses[i]]}, {0: [modelresponses[i]]})[0][1])
        bleu3.append(Bleu(3).compute_score({0: [gtresponses[i]]}, {0: [modelresponses[i]]})[0][2])
        bleu4.append(Bleu(4).compute_score({0: [gtresponses[i]]}, {0: [modelresponses[i]]})[0][3])
        rouge.append(Rouge().compute_score({0: [gtresponses[i]]}, {0: [modelresponses[i]]})[0])
    aux_features[:,0] = bleu1
    aux_features[:,1] = bleu2
    aux_features[:,2] = bleu3
    aux_features[:,3] = bleu4
    aux_features[:,4] = rouge
    return aux_features

def make_plot(model_scores, human_scores, filename):
    pp.clf()
    pp.plot(human_scores, model_scores, 'ko')
    pp.savefig(filename)

def make_line_plot(model_scores, human_scores, filename):
    pp.clf()
    pp.plot(human_scores, model_scores)
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
    def __init__(self, input, feat, emb_dim, batch_size, init_mean, init_range, feat_dim=0):
        self.M = theano.shared(np.eye(emb_dim).astype(theano.config.floatX), borrow=True)
        self.N = theano.shared(np.eye(emb_dim).astype(theano.config.floatX), borrow=True)
        self.f = theano.shared(np.zeros((feat_dim,)).astype(theano.config.floatX), borrow=True)
        
        # Set embeddings by slicing tensor
        self.emb_context = input[:,0,:]
        self.emb_response = input[:,1,:]
        self.emb_true_response = input[:,2,:]
        self.feat = feat


        # Compute predictions
        self.pred1 = T.sum(self.emb_context * T.dot(self.emb_response, self.M), axis=1)
        self.pred2 = T.sum(self.emb_true_response * T.dot(self.emb_response, self.N), axis=1)
        #self.pred3 = 0
        # TODO: add condition here that works
        self.pred3 = T.dot(self.feat, self.f)
        self.pred = self.pred1 + self.pred2 + self.pred3

        # Julian: I think adding a squared error on top of a sigmoid function will be difficult to train.
        #         Let's just try with a linear output first. We can always clip it to be within [0, 5] later.
        #self.output = 5 * T.clip(T.nnet.sigmoid(self.pred), 1e-7, 1 - 1e-7)
        self.output = 2.5 + 5 * (self.pred - init_mean) / init_range # to re-scale dot product values to [0,5] range


    def squared_error(self, score):
        return T.mean((self.output - score)**2)

    def l2_regularization(self):
        return self.M.norm(2) + self.N.norm(2)
    
    def l1_regularization(self):
        return self.M.norm(1) + self.N.norm(1)


def train(train_x, test_x, train_y, test_y, init_mean, init_range, learning_rate=0.01, num_epochs=100, \
        batch_size=16, l2reg=0, l1reg=0, train_feat=None, test_feat=None, pca_name=None, \
        exp_folder=None):
    
    print '...building model'
    n_train_batches = train_x.shape[0] / batch_size
    emb_dim = int(train_x.shape[2])
    feat_dim = int(train_feat.shape[1])
    
    
    train_y_values = train_y
    train_x = set_shared_variable(train_x)
    test_x = set_shared_variable(test_x)
    train_y = set_shared_variable(train_y)
    train_feat = set_shared_variable(train_feat)
    test_feat = set_shared_variable(test_feat)
    
    index = T.lscalar()
    x = T.tensor3('x')
    y = T.fvector('y')
    feat = T.fmatrix('feat')

    model = LinearEvalModel(input=x, feat=feat, emb_dim=emb_dim, batch_size=batch_size, init_mean=init_mean, init_range=init_range, \
            feat_dim=feat_dim)

    cost = model.squared_error(y) + l2reg * model.l2_regularization() + l1reg * model.l1_regularization()
        
    get_output = theano.function(
        inputs=[],
        outputs=model.output,
        givens={
            x: test_x,
            feat: test_feat
        }
    )

    get_output_train = theano.function(
        inputs=[],
        outputs=model.output,
        givens={
            x: train_x,
            feat: train_feat
        }
    )
    
    g_M = T.grad(cost=cost, wrt=model.M)
    g_N = T.grad(cost=cost, wrt=model.N)
    g_f = T.grad(cost=cost, wrt=model.f)
    updates = [ (model.M, model.M - learning_rate * g_M),
                (model.N, model.N - learning_rate * g_N), 
                (model.f, model.f - learning_rate * g_f) ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_x[index * batch_size: (index + 1) * batch_size],
            y: train_y[index * batch_size: (index + 1) * batch_size],
            feat: train_feat[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    print '..starting training'
    epoch = 0
    first_output = get_output()
    best_output = np.zeros((50,)) 
    best_cor = [0,0]
    loss_list = []
    spearman_train = []
    pearson_train = []
    spearman_test = []
    pearson_test = []
    best_correlation = -np.inf
    start_time = time.time()
    while (epoch < num_epochs):
        epoch += 1    
        if epoch % 100 == 0:
            print 'Starting epoch',epoch
        cost_list = []
        for minibatch_index in xrange(n_train_batches):
            minibatch_cost = train_model(minibatch_index)
            cost_list.append(minibatch_cost)
        model_out = get_output()
        model_train_out = get_output_train()
        loss = sum(cost_list) / float(len(cost_list))
        loss_list.append(loss)

        train_correlation = correlation(model_train_out, train_y_values)
        test_correlation = correlation(model_out, test_y)
        spearman_train.append(train_correlation[0][0])
        spearman_test.append(test_correlation[0][0])
        pearson_train.append(train_correlation[1][0])
        pearson_test.append(test_correlation[1][0])
        
        if test_correlation[0][0] > best_correlation:
            best_correlation = test_correlation[0][0]
            best_cor = test_correlation
            best_output = get_output()
            #with open('best_model.pkl', 'w') as f:
            #    cPickle.dump(model, f)
    
    end_time = time.time()
    folder_name = pca_name + '_bs=' + str(batch_size) + '_lr=' + str(learning_rate) + '_l1=' + \
            str(l1reg) + '_l2=' + str(l2reg) + '_epochs=' + str(num_epochs) 
    print_string = '%%%% ' + folder_name + ' %%%%'
    print_string += '\n Finished training. Took %f s'%(end_time - start_time)
    print_string += '\n Best Spearman correlation: ' + str(best_cor[0])
    print_string += '\n Best Peason correlation: ' + str(best_cor[1])
    print_string += '\n Final Spearman correlation: (test) ' + str(test_correlation[0])
    print_string += '\n Final Peason correlation (test): ' + str(test_correlation[1])
    print_string +=  '\n Final Spearman correlation: (train) ' + str(train_correlation[0])
    print_string +=  '\n Final Peason correlation (train): ' + str(train_correlation[1])
    print print_string
    if not os.path.exists('./results/' + exp_folder + folder_name):
        os.makedirs('./results/' + exp_folder + folder_name)
    
    # Make scatter plots
    make_plot(best_output, test_y, './results/' + exp_folder + folder_name + '/best.png')
    make_plot(first_output, test_y, './results/' + exp_folder + folder_name + '/init.png')
    make_plot(model_out, test_y, './results/' + exp_folder + folder_name + '/final(test).png')
    make_plot(model_train_out, train_y_values, './results/' + exp_folder + folder_name + '/final(train).png')
    
    # Make learning curves
    epoch_list = range(len(loss_list))
    make_line_plot(loss_list, epoch_list, './results/' + exp_folder + folder_name + '/loss.png')
    make_line_plot(spearman_train, epoch_list, './results/' + exp_folder + folder_name + '/spear_train.png')
    make_line_plot(spearman_test, epoch_list, './results/' + exp_folder + folder_name + '/spear_test.png')
    make_line_plot(pearson_train, epoch_list, './results/' + exp_folder + folder_name + '/pear_train.png')
    make_line_plot(pearson_test, epoch_list, './results/' + exp_folder + folder_name + '/pear_test.png')
    
    # Save summary info
    with open('./results/' + exp_folder + folder_name + '/results.txt', 'w') as f1:
        f1.write(print_string)
    return '\n\n' + print_string

if __name__ == '__main__':
    test_pct = 0.20
    pca_components = 5
    use_aux_features = True
    use_precomputed_embeddings = True
    eval_overlap_metrics = True

    print 'Loading data...'
    
    #ubuntu_file = '../ubuntu_human_data.csv'
    #twitter_file = '../twitter_human_data.csv'
    clean_data_file = '../clean_data.pkl'   # Dictionary with userid as key, list of dicts as values, where each
                                            # dict represents a single context (c_id is the key for looking up contexts in context.pkl)
    twitter_file = '../contexts.pkl' # List of the form [context_id, context, resp1, resp2, resp3, resp4]
    twitter_gt_file = '../true.txt' # File with ground-truth responses. Line no. corresponds to context_id
    
    if len(sys.argv) > 2:
        if sys.argv[2] != None:
            embedding_type = sys.argv[2]
    else:
        embedding_type = 'CONTEXT' # Can be "CONTEXT" or "DECODER"   
    print 'Embedding type is ' + embedding_type
    
    if embedding_type == 'CONTEXT':
        context_embedding_file = './context_emb_vhredcontext.pkl'
        modelresponses_embedding_file = './modelresponses_emb_vhredcontext.pkl'
        gtresponses_embedding_file = './gtresponses_emb_vhredcontext.pkl'
    elif embedding_type == 'DECODER':
        context_embedding_file = './context_emb_vhreddecoder.pkl'
        modelresponses_embedding_file = './modelresponses_emb_vhreddecoder.pkl'
        gtresponses_embedding_file = './gtresponses_emb_vhreddecoder.pkl'
    
    twitter_bpe_dictionary = '../TwitterData/BPE/Twitter_Codes_5000.txt'
    twitter_bpe_separator = '@@'
    twitter_model_dictionary = '../TwitterData/BPE/Dataset.dict.pkl'

    twitter_model_prefix = '/home/ml/rlowe1/TwitterData/hred_twitter_models/1470516214.08_TwitterModel__405001'
    #twitter_model_prefix = '../TwitterModel/1470516214.08_TwitterModel__405001'
    # previously: '../TwitterModel/1470516214.08_TwitterModel__405001'
    # changed due to disk space limitations on Ryan's machine
    
    # Load Twitter evaluation data from .pkl files
    twitter_contexts, twitter_gtresponses, twitter_modelresponses, twitter_human_scores = get_twitter_data(clean_data_file, \
            twitter_file, twitter_gt_file)
    
    if eval_overlap_metrics:
        show_overlap_scores(twitter_gtresponses, twitter_modelresponses, twitter_human_scores, test_pct)
        
    # Load in Twitter dictionaries
    twitter_bpe = BPE(open(twitter_bpe_dictionary, 'r').readlines(), twitter_bpe_separator)
    twitter_dict = cPickle.load(open(twitter_model_dictionary, 'r'))
    twitter_str_to_idx = dict([(tok, tok_id) for tok, tok_id, _, _ in twitter_dict])

    # Get data, for Ubuntu
    #ubuntu_data = load_data(ubuntu_file)
    #ubuntu_human_scores = get_score(ubuntu_data)

    # Get data, for Twitter (with old formatting)
    #twitter_data = np.array(load_data(twitter_file))
    #twitter_contexts = get_context(twitter_data)
    #twitter_gtresponses = get_gtresponse(twitter_data)
    #twitter_modelresponses = get_modelresponse(twitter_data)
    #twitter_human_scores = get_score(twitter_data)

    # Encode text into BPE format
    twitter_context_ids = strs_to_idxs(twitter_contexts, twitter_bpe, twitter_str_to_idx)
    twitter_gtresponses_ids = strs_to_idxs(twitter_gtresponses, twitter_bpe, twitter_str_to_idx)
    twitter_modelresponses_ids = strs_to_idxs(twitter_modelresponses, twitter_bpe, twitter_str_to_idx)
    
    # Compute VHRED embeddings
    if use_precomputed_embeddings:
        print 'Loading precomputed embeddings...'
        with open(context_embedding_file, 'r') as f1:
            twitter_context_embeddings = cPickle.load(f1)
        with open(gtresponses_embedding_file, 'r') as f1:
            twitter_gtresponses_embeddings = cPickle.load(f1)
        with open(modelresponses_embedding_file, 'r') as f1:
            twitter_modelresponses_embeddings = cPickle.load(f1)
    
    elif 'gpu' in theano.config.device.lower():
        print 'Loading model...'
        state = prototype_state()
        state_path = twitter_model_prefix + "_state.pkl"
        model_path = twitter_model_prefix + "_model.npz"

        with open(state_path) as src:
            state.update(cPickle.load(src))

        state['bs'] = 20
        state['dictionary'] = twitter_model_dictionary

        model = DialogEncoderDecoder(state) 
        
        print 'Computing context embeddings...'
        twitter_context_embeddings = compute_model_embeddings(twitter_context_ids, model, embedding_type)
        with open(context_embedding_file, 'w') as f1:
            cPickle.dump(twitter_context_embeddings, f1)
        print 'Computing ground truth response embeddings...'
        twitter_gtresponses_embeddings = compute_model_embeddings(twitter_gtresponses_ids, model, embedding_type)
        with open(gtresponses_embedding_file, 'w') as f1:
            cPickle.dump(twitter_gtresponses_embeddings, f1)
        print 'Computing model response embeddings...'
        twitter_modelresponses_embeddings = compute_model_embeddings(twitter_modelresponses_ids, model, embedding_type)
        with open(modelresponses_embedding_file, 'w') as f1:
            cPickle.dump(twitter_modelresponses_embeddings, f1)
    
    else:
        # Set embeddings to 0 for now. alternatively, we can load them from disc...
        #embeddings = cPickle.load(open(embedding_file, 'rb'))
        print 'ERROR: No GPU specified!'
        print ' To save testing time, model will be trained with zero context / response embeddings...'
        twitter_context_embeddings = np.zeros((len(twitter_context_embeddings), 3, emb_dim))
        twitter_gtresponses_embedding = np.zeros((len(twitter_context_embeddings), 3, emb_dim))
        twitter_modelresponses_embeddings = np.zeros((len(twitter_context_embeddings), 3, emb_dim))
    

    # Copy the contexts and gt responses 4 times (to align with the model responses)
    temp_c_emb = []
    temp_gt_emb = []
    temp_gt = []
    for i in xrange(len(twitter_context_embeddings)):
        temp_c_emb.append([twitter_context_embeddings[i]]*4)
        temp_gt_emb.append([twitter_gtresponses_embeddings[i]]*4)
        temp_gt.append([twitter_gtresponses[i]]*4)
    twitter_context_embeddings = [i for sublist in temp_c_emb for i in sublist]
    twitter_gtresponses_embeddings = [i for sublist in temp_gt_emb for i in sublist]
    twitter_gtresponses = [i for sublist in temp_gt for i in sublist]

    assert len(twitter_context_embeddings) == len(twitter_gtresponses_embeddings)
    assert len(twitter_context_embeddings) == len(twitter_modelresponses_embeddings)

    emb_dim = twitter_context_embeddings[0].shape[0]
    
    twitter_dialogue_embeddings = np.zeros((len(twitter_context_embeddings), 3, emb_dim))
    for i in range(len(twitter_context_embeddings)):
        twitter_dialogue_embeddings[i, 0, :] =  twitter_context_embeddings[i]
        twitter_dialogue_embeddings[i, 1, :] =  twitter_gtresponses_embeddings[i]
        twitter_dialogue_embeddings[i, 2, :] =  twitter_modelresponses_embeddings[i]
    
    if len(sys.argv) > 0 and sys.argv[1] != None:
        exp_name = sys.argv[1]
    else:
        exp_name = 'rand_exp_' + str(randint(0,99))
    exp_folder = exp_name + '/'
    
    print 'Computing auxiliary features...'
    if use_aux_features:
        aux_features = get_auxiliary_features(twitter_contexts, twitter_gtresponses, twitter_modelresponses, len(twitter_modelresponses))
    else:
        aux_features = np.zeros((len(twitter_contexts), 5))

    # Main loop through hyperparameter search
    separate_pca = False
    total_summary = ''
    pca_list = [5, 7, 10, 15, 20, 35, 50, 100]
    l1reg_list = [0.005, 0.01, 0.02, 0.03, 0.05]#1e-3, 1e-2, 0.1]
    l2reg_list = [0]
    lr_list = [0.01]
    last_pca = 0
    for pca_components in pca_list:
        for lr in lr_list:
            for l2reg in l2reg_list:
                for l1reg in l1reg_list:
                    print '\n%%%%%%%%%%%%%  Running experiment with PCA=' + str(pca_components) + ', l2reg=' \
                            + str(l2reg) + ', l1reg=' + str(l1reg) + ', aux_feat=' + str(use_aux_features) + \
                            ' %%%%%%%%%%%%%%'
                    if separate_pca == False:
                        print 'Also, combined PCA with ' + embedding_type + ' embeddings'
                    
                    # Separate into train and test (for the embedding data, this is done inside
                    # the PCA function
                    train_index = int((1 - test_pct) * twitter_dialogue_embeddings.shape[0])
                    train_y = np.array(twitter_human_scores[:train_index])
                    test_y = np.array(twitter_human_scores[train_index:])
                    
                    # Reduce the dimensionality of the embeddings with PCA
                    if pca_components == last_pca:
                        print 'Using embeddings from last round...'                    
                    else:
                        print 'Computing PCA...'
                        if pca_components < emb_dim:
                            if separate_pca:
                                twitter_dialogue_embeddings2 = compute_separate_pca(pca_components, twitter_dialogue_embeddings)
                                train_x = twitter_dialogue_embeddings2[:train_index]
                                test_x = twitter_dialogue_embeddings2[train_index:]
                                pca_prefix = 'sep'
                            else: 
                                train_x, test_x = compute_pca(pca_components, twitter_dialogue_embeddings, train_index)
                                pca_prefix = ''
                        else:
                            twitter_dialogue_embeddings2 = twitter_dialogue_embeddings
                            pca_prefix = ''
                    init_mean, init_range = compute_init_values(train_x)                                    

                    train_feat = aux_features[:train_index]
                    test_feat = aux_features[train_index:]
                    print 'Training model...'
                    summary = train(train_x, test_x, train_y, test_y, init_mean, init_range, learning_rate=lr, l2reg=l2reg, l1reg=l1reg, train_feat=train_feat, \
                            test_feat=test_feat, pca_name=pca_prefix+'pca'+str(pca_components), exp_folder=exp_folder)
                    total_summary += summary
                    last_pca = pca_components
     
    with open('./results/summary_of_experiment_' + exp_name + '.txt', 'w') as f1:
        f1.write(total_summary)

    # Start training with:
    #   THEANO_FLAGS=mode=FAST_COMPILE,floatX=float32 python eval_model_hred.py
    # or
    #   THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python eval_model_hred.py

