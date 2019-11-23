# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 21:24:05 2019

@author: DusteJ
"""
import numpy as np
from glob import glob
from scipy.special import expit as sigmoid
from sklearn.utils import shuffle
from datetime import datetime
from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances
import string
#Load Data
def get_corpus():
    V = 5000
    files = glob('./wiki/AA/wiki_*')
    all_words_counts = {}
    
    for f in files:
        for line in open(f, encoding="utf8"):
            if line and line[0] not in '[*-|=\{\}':
                s = remove_punctuation(line).lower().split()
                if len(s) > 1:
                    for word in s:
                        
                        if word not in all_words_counts:
                            all_words_counts[word] = 0
                         
                        all_words_counts[word] += 1  
    #Get minimum between Max Words and Corpus 
    V = min(V, len(all_words_counts))
    print("Word count is %s" %(len(all_words_counts)))
    #sort based on word counts
    all_words_counts = sorted(all_words_counts.items(), key = lambda x: x[1], reverse = True) 
    
    #Get just to Max words from corpus
    top_words = [word for word, count in all_words_counts[:V-1]] + ['<UNKNOWN>']  
    #Store word and there indexes as dictionary
    word2idx = {w:i for i, w in enumerate(top_words)}   
    unkonwnidx = word2idx['<UNKNOWN>']
    sentences = []    
    #encode sentences
    for f in files:
        for line in open(f, encoding="utf8"):
            if line and line[0] not in '[*-|=\{\}':
                s = remove_punctuation(line).lower().split()
                if len(s) > 1:
                    sentence = [word2idx[word] if word in word2idx else unkonwnidx for word in s]
                    sentences.append(sentence)
    return sentences, word2idx
            

#Clean Data
def remove_punctuation(s):
    return s.translate(str.maketrans('','',string.punctuation))                

#Negative Sampling Distribution to sample more often the words that occour more often
def get_negative_sampling_distribution(sentences, vocab_size):
    word_freq = np.zeros(vocab_size)
    
    for sentence in sentences:
        for word in sentence:
            word_freq[word] += 1
    
    #smooth the negative distribution
    p_neg = word_freq**0.75
    
    #normalize it
    p_neg = p_neg / p_neg.sum()
    
    #assert(np.all(p_neg > 0))
    
    return p_neg

def get_context(context_pos, sentence, window_size):
    
    start = max(0, context_pos - window_size)
    end_ = min(len(sentence), context_pos + window_size)
    context = []
    for idx, word_idx in enumerate(sentence[start:end_], start=start):
        if context_pos != idx:
            context.append(word_idx)
    
    return context


    
    
#SGD
def sgd(input_, targets, label, learning_rate, W, H):
    
    # W[input_] shape D
    # H[:, targets] shape N X D
    
    activation = W[input_].dot(H[:, targets])
    probability = sigmoid(activation)
    
    #compute gradient
    gH = np.outer(W[input_], probability - label) #D X N
    gW = np.sum((probability - label)*H[:, targets], axis = 1) #D
    
    #update weights
    H[:, targets] -= learning_rate * gH
    W[input_] -= learning_rate * gW
    
    # return cost (binary cross entropy)
    cost = label * np.log(probability + 1e-10) + (1 - label) * np.log(1 - probability + 1e-10)
    
    return cost.sum()
    
#train model
def train_model():
    sentences, word2idx = get_corpus()
    # config
    window_size = 5
    learning_rate = 0.025
    final_learning_rate = 0.0001
    num_negatives = 5 # number of negative samples to draw per input word
    epochs = 20
    D = 50 # word embedding size
    vocab_size = len(word2idx)
    W = np.random.randn(vocab_size, D)
    H = np.random.randn(D, vocab_size)
     
    #learning rate decay delta
    learning_rate_delta = (learning_rate - final_learning_rate)/epochs
     
    #distribution for drawing -ve samples
    p_neg = get_negative_sampling_distribution(sentences, vocab_size)
     
    #save cost to plot them per iteration
    costs = []
     
    #for subsampling each sentence
    threshold = 1e-5
    p_drop = 1 - np.sqrt(threshold/p_neg)
     
    #loop through number of epochs
    for epoch in range(epochs):
        
        np.random.shuffle(sentences)
         
        cost = 0
        counter = 0
        t0 = datetime.now()
        for sentence in sentences:             
            sentence = [w for w in sentence \
                        if np.random.random() < (1 - p_drop[w])]
             
            if len(sentence) < 2:
                continue
            
            randomly_ordered_positions = np.random.choice(len(sentence), size=len(sentence), replace=False,)
             
            for pos in randomly_ordered_positions:
                
                word = sentence[pos]
                contexts = get_context(pos, sentence, window_size)
                neg_word = np.random.choice(vocab_size, p=p_neg)
                targets = np.array(contexts)
                 
                #stochastic gradient descent with +ve words
                c = sgd(word, targets, 1, learning_rate, W, H)                
                cost += c
                #stochastic gradient descent with -ve words
                c = sgd(neg_word, targets, 0, learning_rate, W, H)
                cost += c
                
            counter += 1 
                                                              
            if counter % 100 == 0:
                print("Processed %s %s \r" % (counter, len(sentences)))
           
        t1 = datetime.now() - t0
        costs.append(cost)
           
        print("epoch completed:", epoch, "cost:", cost, "time taken:", t1)
           
        learning_rate = learning_rate - learning_rate_delta
     
        
     
    return word2idx, W, H
 
#Analogy
def analogy(p1, n1, p2, n2, word2idx, idx2word, WordEmbeddings):
    V,D = WordEmbeddings.shape
    for w in (p1, n1, p2, n2):
        if w not in word2idx:
            print("%s not found in word2idx" % w)
            return
        
    print("testing %s - %s = %s - %s" % (p1, n1, p2, n2))        
    vec1 = WordEmbeddings[word2idx[p1]]
    vec2 = WordEmbeddings[word2idx[n2]]
    vec3 = WordEmbeddings[word2idx[p2]]
    vec4 = WordEmbeddings[word2idx[n2]]
    
    vec = vec1 - vec2 + vec4
    
    closest_neighbours = pairwise_distances(vec.reshape(1, D), WordEmbeddings, metric='cosine').reshape(V)
    top_ten = closest_neighbours.argsort()[:10]
    
    best_pick = -1
    keep_out = [word2idx[w] for w in (p1, n1, n2)]
    
    for idx in top_ten:
        if idx not in keep_out:
            best_pick = idx
            break
        
            
    print("got %s - %s = %s - %s" % (p1, n1, idx2word[best_pick], n2))
    
    print("distance to %s", p2, cos_dist(vec3, vec))
        
#Test model
def test_model(word2idx, W, H):
    
    idx2word = {i:w for w, i in word2idx.items()}
    
    for We in (W, (W + H.T) / 2):
        print("**********")
    
        analogy('king', 'man', 'queen', 'woman', word2idx, idx2word, We)
        analogy('king', 'prince', 'queen', 'princess', word2idx, idx2word, We)
        analogy('miami', 'florida', 'dallas', 'texas', word2idx, idx2word, We)
        analogy('einstein', 'scientist', 'picasso', 'painter', word2idx, idx2word, We)
        analogy('japan', 'sushi', 'germany', 'bratwurst', word2idx, idx2word, We)
        analogy('man', 'woman', 'he', 'she', word2idx, idx2word, We)
        analogy('man', 'woman', 'uncle', 'aunt', word2idx, idx2word, We)
        analogy('man', 'woman', 'brother', 'sister', word2idx, idx2word, We)
        analogy('man', 'woman', 'husband', 'wife', word2idx, idx2word, We)
        analogy('man', 'woman', 'actor', 'actress', word2idx, idx2word, We)
        analogy('man', 'woman', 'father', 'mother', word2idx, idx2word, We)
        analogy('heir', 'heiress', 'prince', 'princess', word2idx, idx2word, We)
        analogy('nephew', 'niece', 'uncle', 'aunt', word2idx, idx2word, We)
        analogy('france', 'paris', 'japan', 'tokyo', word2idx, idx2word, We)
        analogy('france', 'paris', 'china', 'beijing', word2idx, idx2word, We)
        analogy('february', 'january', 'december', 'november', word2idx, idx2word, We)
        analogy('france', 'paris', 'germany', 'berlin', word2idx, idx2word, We)
        analogy('week', 'day', 'year', 'month', word2idx, idx2word, We)
        analogy('week', 'day', 'hour', 'minute', word2idx, idx2word, We)
        analogy('france', 'paris', 'italy', 'rome', word2idx, idx2word, We)
        analogy('paris', 'france', 'rome', 'italy', word2idx, idx2word, We)
        analogy('france', 'french', 'england', 'english', word2idx, idx2word, We)
        analogy('japan', 'japanese', 'china', 'chinese', word2idx, idx2word, We)
        analogy('china', 'chinese', 'america', 'american', word2idx, idx2word, We)
        analogy('japan', 'japanese', 'italy', 'italian', word2idx, idx2word, We)
        analogy('japan', 'japanese', 'australia', 'australian', word2idx, idx2word, We)
        analogy('walk', 'walking', 'swim', 'swimming', word2idx, idx2word, We)
    
#load Model


if __name__ == '__main__':
    word2idx, W, H = train_model()
    test_model(word2idx, W, H)


