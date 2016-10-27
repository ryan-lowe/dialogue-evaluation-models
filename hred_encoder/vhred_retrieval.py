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

from scipy.spatial import KDTree

from dialog_encdec import DialogEncoderDecoder
from numpy_compat import argpartition
from state import prototype_state
from compute_dialogue_embeddings import compute_encodings

import os
os.sys.path.insert(0,'../TwitterData/BPE/subword_nmt')

from apply_bpe import BPE



def preprocess_tweet(s):
    s = s.replace('@user', '<at>').replace('&lt;heart&gt;', '<heart>').replace('&lt;number&gt;', '<number>').replace('  ', ' </s> ').replace('  ', ' ')

    # Make sure we end with </s> token
    while s[-1] == ' ':
        s = s[0:-1]
    if not s[-5:] == ' </s>':
        s = s + ' </s>'

    return s


def process_dialogues(dialogues):
    ''' Removes </d> </s> at end, splits into contexts/ responses '''
    contexts = []
    responses = []
    for d in dialogues:
        d_proc = d[:-3]
        index_list = [i for i, j in enumerate(d_proc) if j == 1]
        split = index_list[-1] + 1
        contexts.append(d_proc[:split])
        responses.append(d_proc[split:] + [1])
    return contexts, responses

def strs_to_idxs(data, bpe, str_to_idx):
    ''' Encodes strings in BPE form '''
    out = []
    for row in data:
        bpe_segmented = bpe.segment(row.strip())
        # Note: there shouldn't be any unknown tokens with BPE!
        #out.append([str_to_idx[word] for word in bpe_segmented.split()])
        out.append([str_to_idx[word] for word in bpe_segmented.split() if word in str_to_idx])

    return out


def idxs_to_strs(data, bpe, idx_to_str):
    ''' Converts from BPE form to strings '''
    out = []
    for row in data:
        out.append(' '.join([idx_to_str[idx] for idx in row if idx in idx_to_str]).replace('@@ ',''))
    return out

def flatten_list(l1):
    return [i for sublist in l1 for i in sublist]

# Compute model embeddings for contexts or responses 
# Embedding type can be 'CONTEXT' or 'DECODER'
def compute_model_embeddings(data, model, embedding_type, ftype):
    model_compute_encoding = model.build_encoder_function()
    model_compute_decoder_encoding = model.build_decoder_encoding()

    embeddings = []
    context_ids_batch = []
    batch_index = 0
    batch_total = int(math.ceil(float(len(data)) / float(model.bs)))
    counter = 0
    fcounter = 0
    start = time.time()
    for context_ids in data:
        context_ids_batch.append(context_ids)
        counter += 1
        if len(context_ids_batch) == model.bs or counter ==  len(data):
            batch_index += 1

            print '     Computing embeddings for batch ' + str(batch_index) + ' / ' + str(batch_total),
            encs = compute_encodings(context_ids_batch, model, model_compute_encoding, model_compute_decoder_encoding, embedding_type)
            for i in range(len(encs)):
                embeddings.append(encs[i])

            context_ids_batch = []
            print time.time() - start
            start = time.time()

        if batch_index % 1000 == 0 or counter == len(data):
            fcounter += 1
            if embedding_type == 'CONTEXT':
                cPickle.dump(embeddings, open('/home/ml/rlowe1/TwitterData/vhred_context_emb/'+ftype+'_context_emb'+str(fcounter)+'.pkl', 'w'))
            elif embedding_type == 'DECODER':
                cPickle.dump(embeddings, open('/home/ml/rlowe1/TwitterData/vhred_decoder_emb/'+ftype+'_context_emb'+str(fcounter)+'.pkl', 'w'))
            embeddings = []

    return embeddings


def scale_points(train_emb, test_emb, max_val):
    '''
    Scales all points in train_emb, test_emb such that
    max_i ||x_i||_2 = max_val
    max_val corresponds to U in the Auvolat et al. paper
    '''
    pass


def transform_data_points(train_emb):
    emb = []
    pass

def brute_force_search(train_emb, query_emb):
    max_index = -1
    largest_product = -1e9
    for i in xrange(len(train_emb)):
        prod = np.dot(train_emb[i], query_emb)
        if prod > largest_product:
            largest_product = prod
            max_index = i
    return max_index, largest_product

def mat_vector_2norm(mat):
    '''
    Takes as input a matrix, and returns a vector correponding to the 2-norm
    of each row vector.
    '''
    norm_list = []
    for i in xrange(mat.shape[0]):
        norm_list.append(np.sqrt(np.dot(mat[0], mat[0].T)))
    return np.array(norm_list)


def test_model(train_emb, test_emb, train_responses, test_responses, train_contexts, test_contexts, output_file):
    '''
    Tests the model by finding the closest context embedding in the training set
    for each test query (using approximate MIPS). Then, outputs the corresponding response from
    the training set.
    Approximate MIPS is done using the spherical k-means method from Auvolat et al. (2016)    
    '''
    
    #train_emb, test_emb = scale_points(train_emb, test_emb, U)

    #train_emb = transform_data_points(train_emb)
    #test_emb = transform_query_vectors(test_emb)

    test_ar = np.array(test_emb)
    train_ar = np.array(train_emb)
    prod_matrix = np.dot(train_ar, test_ar.T) # has shape (train_ex, test_ex)
    prod_matrix = prod_matrix / mat_vector_2norm(test_ar) # divide by 2-norm of vectors to produce cosine sim
    prod_matrix = (prod_matrix.T / mat_vector_2norm(train_ar) ).T
    argmax_ar = np.argmax(prod_matrix, axis=0)

    model_responses = []
    closest_contexts = []
    highest_dotproduct = []

    for train_index, example in zip(list(argmax_ar), range(len(argmax_ar))):
        model_responses.append(train_responses[train_index])
        closest_contexts.append(train_contexts[train_index])
        highest_dotproduct.append(prod_matrix[train_index][example])

    # Write data to output CSV
    with open(output_file, 'w') as out:
        writer = csv.writer(out)
        writer.writerow(['Context', 'Score', 'Model Response', 'GT Response', 'Closest Context'])
        for i in xrange(len(model_responses)):
            writer.writerow([test_contexts[i], highest_dotproduct[i], model_responses[i], \
                    test_responses[i], closest_contexts[i]])





if __name__ == '__main__':
    twitter_bpe_dictionary = '../TwitterData/BPE/Twitter_Codes_5000.txt'
    twitter_bpe_separator = '@@'
    twitter_model_dictionary = '../TwitterData/BPE/Dataset.dict.pkl'

    twitter_model_prefix = '/home/ml/rlowe1/TwitterData/hred_twitter_models/1470516214.08_TwitterModel__405001'
    twitter_data_prefix = '/home/ml/rlowe1/TwitterData/hred_retrieval_model/'
    
    max_trainemb_index = 759 # max = 759
    max_testemb_index = 20 # max = 99
    use_precomputed_embeddings = False
    embedding_type = 'CONTEXT' #'CONTEXT'

    # Load in Twitter dictionaries
    twitter_bpe = BPE(open(twitter_bpe_dictionary, 'r').readlines(), twitter_bpe_separator)
    twitter_dict = cPickle.load(open(twitter_model_dictionary, 'r'))
    twitter_str_to_idx = dict([(tok, tok_id) for tok, tok_id, _, _ in twitter_dict])
    twitter_idx_to_str = dict([(tok_id, tok) for tok, tok_id, _, _ in twitter_dict])
    
    # Get data, for Twitter
    train_file = '/home/ml/rlowe1/TwitterData/TwitterDataBPE/Train.dialogues.pkl'
    test_file = '/home/ml/rlowe1/TwitterData/TwitterDataBPE/Test.dialogues.pkl'
    output_file = './output.csv'

    with open(train_file) as f1:
        train_data = cPickle.load(f1)
    with open(test_file) as f1:
        test_data = cPickle.load(f1)
    
    train_contexts, train_responses = process_dialogues(train_data)
    test_contexts, test_responses = process_dialogues(test_data)

    train_contexts_txt = idxs_to_strs(train_contexts, twitter_bpe, twitter_idx_to_str)
    train_responses_txt = idxs_to_strs(train_responses, twitter_bpe, twitter_idx_to_str)
    test_contexts_txt = idxs_to_strs(test_contexts, twitter_bpe, twitter_idx_to_str)
    test_responses_txt = idxs_to_strs(test_responses, twitter_bpe, twitter_idx_to_str)


    # Encode text into BPE format
    #twitter_context_ids = strs_to_idxs(twitter_contexts, twitter_bpe, twitter_str_to_idx)
    #twitter_gtresponses_ids = strs_to_idxs(twitter_gtresponses, twitter_bpe, twitter_str_to_idx)
    #twitter_modelresponses_ids = strs_to_idxs(twitter_modelresponses, twitter_bpe, twitter_str_to_idx)

    # Compute VHRED embeddings
    if use_precomputed_embeddings:
        # Load embeddings from /home/ml/rlowe1/TwitterData/hred_retrieval_model
        print 'Loading training context embeddings...'
        train_emb = []
        for i in xrange(1, max_trainemb_index + 1):
            if i % 20 == 0:
                path = twitter_data_prefix + 'train_context_emb' + str(i) + '.pkl'
                with open(path, 'r') as f1:
                    train_emb.append(cPickle.load(f1))

        print 'Loading testing context embeddings...'
        test_emb = []
        for i in xrange(1, max_testemb_index + 1):
            if i % 20 == 0:
                path = twitter_data_prefix + 'test_context_emb' + str(i) + '.pkl'
                with open(path, 'r') as f1:
                    test_emb.append(cPickle.load(f1))
        train_context_embeddings = flatten_list(train_emb)
        test_context_embeddings = flatten_list(test_emb)


    elif 'gpu' in theano.config.device.lower():
        state = prototype_state()
        state_path = twitter_model_prefix + "_state.pkl"
        model_path = twitter_model_prefix + "_model.npz"
        
        with open(state_path) as src:
            state.update(cPickle.load(src))

        state['bs'] = 20
        state['dictionary'] = twitter_model_dictionary

        model = DialogEncoderDecoder(state) 
        
        print 'Computing training context embeddings...'
        train_context_embeddings = compute_model_embeddings(train_contexts, model, embedding_type, 'train')
        #cPickle.dump(twitter_context_embeddings, open('/home/ml/rlowe1/TwitterData/hred_retrieval_model/train_context_emb.pkl', 'w'))

        print 'Computing test context embeddings...'
        test_context_embeddings = compute_model_embeddings(test_contexts, model, embedding_type, 'test')
        #cPickle.dump(twitter_context_embeddings, open('/home/ml/rlowe1/TwitterData/hred_retrieval_model/test_context_emb.pkl', 'w'))

        
        #assert len(train_context_embeddings) == len(test_context_embeddings)

    else:
        # Set embeddings to 0 for now. alternatively, we can load them from disc...
        #embeddings = cPickle.load(open(embedding_file, 'rb'))
        print 'ERROR: No GPU specified!'
    
    start = time.time() 
    print 'Testing model...'
    test_model(train_context_embeddings, test_context_embeddings, train_responses_txt, test_responses_txt, train_contexts_txt, test_contexts_txt, output_file)
    print 'Took %f seconds'%(time.time() - start)


###############

