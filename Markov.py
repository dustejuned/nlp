# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 19:35:41 2019

@author: DusteJ
"""

#import nltk
import numpy as np
from builtins import input, range
import operator

#download brown corpus
#nltk.download('brown')
from nltk.corpus import brown

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

def getBiGramProbabilities(sentences, dimension, startIdx, endIdx, smoothing=1):   
    
    bigramProps = np.ones((dimension, dimension)) * smoothing
    for sentence in sentences:
        for i in range(len(sentence)):            
            #beginning
            if i == 0:
                bigramProps[startIdx, sentence[i]] += 1                            
            #end
            elif i == len(sentence) - 1:
                bigramProps[sentence[i], endIdx] += 1  
            else:
                bigramProps[sentence[i - 1], sentence[i]] += 1  
    bigramProps /= bigramProps.sum(axis=1, keepdims=True)                
    return bigramProps

def getScore(sentence, startIdx, endIdx):
    score = 0
    for i in range(len(sentence)):            
        #beginning
        if i == 0:
           score += np.log(bigramProbabilites[startIdx, sentence[i]])                           
        #end
        elif i == len(sentence) - 1:
            score += np.log(bigramProbabilites[sentence[i], endIdx])
        else:
            score += (bigramProbabilites[sentence[i - 1], sentence[i]])
        
    return score / (len(sentence) + 1)

def getActualSentence(sentence):
    return ' '.join(Idx2Word[i] for i in sentence)



    
sentences, word2Idx, Idx2Word = getSentencesAndIdxWords(20000)

V = len(word2Idx)

sampleProbability = np.ones(V)
sampleProbability[0] = 0
sampleProbability[1] = 0
sampleProbability /= sampleProbability.sum()

bigramProbabilites = getBiGramProbabilities(sentences, V, 0, 1, smoothing=0.1)

real_idx = np.random.choice(len(sentences))

real_sentence = sentences[real_idx]

print("Real Sentence in Corpus:", getActualSentence(real_sentence), "Score:", getScore(real_sentence, 0, 1))

fake_sentence = np.random.choice(V, size=len(real_sentence), p=sampleProbability)

print("Fake Sentence:", getActualSentence(fake_sentence), "Score:", getScore(fake_sentence, 0, 1))

while True:
    badSentence = False
    userInput = input("Get Score for your sentence:\n")
    userInput = userInput.lower().split()
    
    for token in userInput:
        if token not in word2Idx:
            badSentence = True
            break

    if badSentence:
        print("Sorry! Entered words are not in vocabulary.\n")
    else:
        mappedUserInput = [word2Idx[token] for token in userInput]
        print("The Score is:", getScore(mappedUserInput, 0, 1))
    
    shall_cont = input("Do you want to continue? [y\n]")
    
    if shall_cont and shall_cont in ('N', 'n'):
        break
