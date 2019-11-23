# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 19:54:46 2019

@author: DusteJ
"""

import numpy as np
from builtins import input, range
import operator
import random
#download brown corpus
#nltk.download('brown')
from nltk.corpus import brown
from datetime import datetime
import matplotlib.pyplot as plt


def getSentences():
    return brown.sents()

def getSentencesAndIdxWords(vocabSize=2000):
    sentences = getSentences()
    indexedSentences = []
    word2IdxCount = {0: float('inf'), 1: float('inf')}
    word2Idx = {"START": 0, "STOP": 1}
    Idx2Word = {0: "START", 1: "STOP"}
    i = 2
    for sentence in sentences:
        indexedSentence = []
        for token in sentence:
            token = token.lower()
            if token not in word2Idx:
                word2Idx[token] = i
                Idx2Word[i] = token
                i += 1
            idx = word2Idx[token]
            word2IdxCount[idx] = word2IdxCount.get(idx, 0) + 1                        
            indexedSentence.append(word2Idx[token])
        indexedSentences.append(indexedSentence)
    sortedWord2IdxCount = sorted(word2IdxCount.items(), key=operator.itemgetter(1), reverse=True)
    smallWord2Idx = {}
    Idx2NewIdxMap = {}
    newIdx2Word = {}
    newIdx = 0
    smallSentences = []
    
    for idx, count in sortedWord2IdxCount[:vocabSize]:
        word = Idx2Word[idx]
        smallWord2Idx[word] = newIdx
        newIdx2Word[newIdx] = word
        Idx2NewIdxMap[idx] = newIdx
        newIdx += 1
    smallWord2Idx['UNKNOWN'] = newIdx  
    newIdx2Word[newIdx] = 'UNKNOWN'
    unknown = newIdx
    for sentence in indexedSentences:
        if len(sentence) > 1:
            newIndexedSentence = [Idx2NewIdxMap[idx] if idx in Idx2NewIdxMap else unknown for idx in sentence]
            smallSentences.append(newIndexedSentence)
        
    return smallSentences, smallWord2Idx, newIdx2Word


def softmax(a):
    a = a - a.max()
    exp_a = np.exp(a)
    
    return exp_a/exp_a.sum(axis = 1, keepdims=True)

def smoothed_loss(x, decay=0.99):
    y = np.zeros(len(x))
    last = 0
    for t in range(len(x)):
      z = decay * last + (1 - decay) * x[t]
      y[t] = z / (1 - decay ** (t + 1))
      last = z
    return y

sentences, word2Idx, Idx2Word = getSentencesAndIdxWords(2000)

D = 100
V = len(word2Idx)
startIdx = word2Idx["START"]
endIdx = word2Idx["STOP"]
epochs = 2
lr = 1e-2
losses = []

#Weight between Input and Hidden layer
W1 = np.random.randn(V, D) / np.sqrt(V)

#Weight between Hidden and Output layer
W2 = np.random.randn(D, V) / np.sqrt(D)

t0 = datetime.now()
for epoch in range(epochs):
    random.shuffle(sentences)
    j = 0
    #Loop through all sentences and train network
    for sentence in sentences:
        sentence = [startIdx] + sentence + [endIdx]
        n = len(sentence)
        #Create one hot encoding for inputs and targets
        inputs = np.zeros((n-1, V))
        targets = np.zeros((n-1, V))
        
        inputs[np.arange(n-1), sentence[:n-1]] = 1
        targets[np.arange(n-1), sentence[1:]] = 1
        
        #get output predictions
        hidden = np.tanh(inputs.dot(W1))
        predictions = softmax(hidden.dot(W2))
        
        
        #do gradient descent and update W1 and W2
        W2 = W2 - lr * hidden.T.dot(predictions - targets)
        dhidden = (predictions - targets).dot(W2.T) * (1 - hidden * hidden)    
        W1 = W1 - lr * inputs.T.dot(dhidden)
        
        loss = -np.sum(targets * np.log(predictions)) / (n - 1)
        losses.append(loss)
        j += 1
        if j % 10 == 0:
            print("epoch", epoch, "Sentence %s/%s" % (j, len(sentences)), "loss", loss)
    
timeElapsed = datetime.now() - t0

print("Time Elapsed", timeElapsed)

plt.plot(losses)

plt.plot(smoothed_loss(losses))

    
    
    

    
    

