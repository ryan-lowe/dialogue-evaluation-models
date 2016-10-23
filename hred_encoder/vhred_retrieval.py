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


# Compute model embeddings for contexts or responses 
def compute_model_embeddings(data, model, ftype):
    model_compute_encoding = model.build_encoder_function()

    embeddings = []
    context_ids_batch = []
    batch_index = 0
    batch_total = int(math.ceil(float(len(data)) / float(model.bs)))
    counter = 0
    fcounter = 0
    start = time.time()
    for context_ids in data:
        context_ids_batch.append(context_ids)

        if len(context_ids_batch) == model.bs:
            counter += 1
            batch_index = batch_index + 1

            print '     Computing embeddings for batch ' + str(batch_index) + ' / ' + str(batch_total),
            encs = compute_encodings(context_ids_batch, model, model_compute_encoding)
            for i in range(len(encs)):
                embeddings.append(encs[i])

            context_ids_batch = []
            print time.time() - start
            start = time.time()

        if counter % 1000 == 0 or counter == len(data):
            fcounter += 1

            cPickle.dump(embeddings, open('/home/ml/rlowe1/TwitterData/hred_retrieval_model/'+ftype+'_context_emb'+str(fcounter)+'.pkl', 'w'))
            embeddings = []

    return embeddings



def test_model(train_emb, test_emb, train_responses, test_responses, train_contexts, test_contexts, output_file):
    '''
    Tests the model by looping through all test context embeddings, and finding the closest
    context embedding in the training set. Then, outputs the corresponding response from
    the training set
    '''
    # Build kdtree that is used for MIPS
    kdtree = KDTree(zip(train_emb, test_emb))

    model_responses = []
    closest_contexts = []
    highest_dotproduct = []
    for emb in test_emb:
        best_emb = tree.query(emb, k=1)
        train_index = train_emb.index(best_emb)
        model_responses.append(train_responses[train_index])
        closest_contexts.append(train_contexts[train_index])
        highest_dotproduct.append(np.dot(np.array(best_emb), np.array(emb)))

    # Write data to output CSV
    with open(outputfile, 'w') as out:
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

    train_context_txt = idxs_to_strs(train_contexts, twitter_bpe, twitter_idx_to_str)
    train_responses_txt = idxs_to_strs(train_responses, twitter_bpe, twitter_idx_to_str)
    test_context_txt = idxs_to_strs(test_contexts, twitter_bpe, twitter_idx_to_str)
    test_responses_txt = idxs_to_strs(test_responses, twitter_bpe, twitter_idx_to_str)


    print train_context_txt[0:2]
    print train_responses_txt[0:2]

    # Encode text into BPE format
    #twitter_context_ids = strs_to_idxs(twitter_contexts, twitter_bpe, twitter_str_to_idx)
    #twitter_gtresponses_ids = strs_to_idxs(twitter_gtresponses, twitter_bpe, twitter_str_to_idx)
    #twitter_modelresponses_ids = strs_to_idxs(twitter_modelresponses, twitter_bpe, twitter_str_to_idx)

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
        
        if load_embeddings == True:
            train_emb_data_file = twitter_data_prefix + 'train_context_emb'
            test_emb_data_file = twitter_data_prefix + 'test_context_emb'
            train_embfile_num = 2
            test_embfile_num = 2
            
            print 'Loading training context embeddings...'
            train_emb = []
            for i in xrange(train_embfile_num):
                train_emb.append(cPickle.load(train_emb_data_file))
            
            print 'Loading testing context embeddings...'
            for i in xrange(test_embfile_num):
                    test_emb.append(cPickle.load(test_emb_data_file))

        else:
            print 'Computing training context embeddings...'
            train_context_embeddings = compute_model_embeddings(train_contexts, model, 'train')
            #cPickle.dump(twitter_context_embeddings, open('/home/ml/rlowe1/TwitterData/hred_retrieval_model/train_context_emb.pkl', 'w'))

            print 'Computing test context embeddings...'
            test_context_embeddings = compute_model_embeddings(test_contexts, model, 'test')
            #cPickle.dump(twitter_context_embeddings, open('/home/ml/rlowe1/TwitterData/hred_retrieval_model/test_context_emb.pkl', 'w'))

        
        #assert len(train_context_embeddings) == len(test_context_embeddings)

        emb_dim = train_context_embeddings[0].shape[0]

        twitter_dialogue_embeddings = np.zeros((len(twitter_context_embeddings), 3, emb_dim))
        for i in range(len(twitter_context_embeddings)):
            twitter_dialogue_embeddings[i, 0, :] =  twitter_context_embeddings[i]
            twitter_dialogue_embeddings[i, 1, :] =  twitter_gtresponses_embeddings[i]
            twitter_dialogue_embeddings[i, 2, :] =  twitter_modelresponses_embeddings[i]

    else:
        # Set embeddings to 0 for now. alternatively, we can load them from disc...
        #embeddings = cPickle.load(open(embedding_file, 'rb'))
        print 'ERROR: No GPU specified!'
    
    print 'Testing model...'
    test_model(train_context_embeddings, test_context_embeddings, train_responses_txt, test_responses_txt, train_contexts_txt, test_contexts_txt, output_file)
