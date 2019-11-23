# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 13:37:45 2019

@author: DusteJ
"""

import numpy as np
from scipy.special import expit as sigmoid
from sklearn.utils import shuffle
from datetime import datetime
import sys
import os
import json
from Word2Vec_NumPy import get_corpus, analogy
import matplotlib.pyplot as plt

class Glove:
    def __init__(self, V, D, ContextSize):
        #V is vocabulary size
        #D is embedding dimensionality
        #ContextSize is sentence context size
        self.V = V
        self.D = D
        self.ContextSize = ContextSize
        
            
    def fit(self, Sentences, co_matrix=None, lr=1e-5, reg=0.01, alpha=0.75, xmax=100, epochs=5, gd=False):
        startTime = datetime.now()
        V = self.V
        D = self.D
        if not os.path.exists(co_matrix):
            X = np.zeros((V,V)) #Co-Occurance Matrix
            N = len(Sentences)
            it = 0
            for sentence in Sentences:
                it += 1 
                #build co-occurance matrix
                for i in range(len(sentence)):
                    start = max(0, i - self.ContextSize)
                    end = min(len(sentence), i + self.ContextSize)
                    wi = sentence[i] #index of word in sentence
                    
                    #Check if word is near to <START>
                    if i - self.ContextSize < 0:
                        score = 1.0 / (i + 1)
                        X[0, wi] = score
                        X[wi, 0] = score
                    
                    #Check if word is near to <END>
                    if i + self.ContextSize > len(sentence):
                        score = 1.0 / (len(sentence) - i)
                        X[1, wi] = score
                        X[wi, 1] = score
                    
                    #Calculate left side co-occurance
                    for j in range(start, i):
                        wj = sentence[j]
                        score = 1.0 / (i - j)
                        X[wi, wj] = score
                        X[wj, wi] = score
                        
                    
                    #Calculate right side co-occurance
                    for j in range(i+1, end):
                        wj = sentence[j]
                        score = 1.0 / (j - i)
                        X[wi, wj] = score
                        X[wj, wi] = score
                   
                if it % 100 == 0:
                    print("Completed %s/%s"% (it, N))
                    
            np.save(co_matrix, X)   
            print(X[:200])
            
        else:
            X = np.load(co_matrix)
            
        print("time to build co-occurrence matrix:", (datetime.now() - startTime))  
        
        #initialize weights and train model  
        Fx = np.zeros((V,V)) 
        W = np.random.randn(V, D) / np.sqrt(V + D) 
        #Row Bias
        b = np.zeros(V) 
      
        U = np.random.randn(V, D) / np.sqrt(V + D) 
        #Column Bias
        c = np.zeros(V) 
        
        Fx[X < xmax] = (X[X < xmax] / float(xmax))**alpha
        Fx[X >= xmax] = 1
        logX = np.log(X + 1)
        mu = logX.mean()
        costs = []
        
        for epoch in range(epochs):
            print("processing epoch %s/%s" % (epoch, epoch))
            delta = W.dot(U.T) + b.reshape(1, V) + c.reshape(V, 1) + mu - logX
            cost = (Fx * delta * delta).sum()
            costs.append(cost)
            
            #update W
            for i in range(V):
                W[i] -= lr * (Fx[i,:] * delta[i,:]).dot(U)
            #Regularize W    
            W -= reg * lr * W
            
            #update b
            for i in range(V):
                b[i] -= lr * Fx[i, :].dot(delta[i, :])
            
            #Update U
            for j in range(V):
                U[j] -= lr * (Fx[:,j] * delta[:, j]).dot(W)
            #Regularize U
            U -= reg * lr * U
            
            #update c
            for j in range(V):
                c[j] -= lr * Fx[:, j].dot(delta[:, j])
        
        self.W = W
        self.U = U
        plt.plot(costs)
        
    
    def save(self,fileName):
        arrays = [self.W, self.U.T]
        np.savez(fileName, *arrays)
    
    

def run(WordEmbeddingFile, Word2IdxFile):
    #check if file exsits
    co_matrix = "co_matrix_10.npy"
    
    if os.path.exists(co_matrix):
        with open(Word2IdxFile) as f:
            word2idx = json.load(f)
        sentences = []
    else:
        #get corpus
         sentences, word2idx = get_corpus()   
         
    with open(Word2IdxFile, 'w') as f:
        json.dump(word2idx, f)     
    
    
    
    #Create a Glove Model
    V = len(word2idx)
    model = Glove(V, 100, 15)
    #fit the model with indexed sentences
    model.fit(sentences, co_matrix = co_matrix, epochs=20)
    model.save(WordEmbeddingFile)
    
    

if __name__ == '__main__':
    we_file = "glove_model_10.npz"
    w2idx_file = "glove_w2idx_10.json"
    run(we_file, w2idx_file)
   
    # load embeddings
    npz = np.load(we_file)
    W1 = npz['arr_0']
    W2 = npz['arr_1']

    with open(w2idx_file) as f:
        word2idx = json.load(f)
        idx2word = {i:w for w,i in word2idx.items()}
    #find analogies
    for concat in (True, False):
        print("** concat:", concat)

        if concat:
            We = np.hstack([W1, W2.T])
        else:
            We = (W1 + W2.T) / 2
            
            
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
    
