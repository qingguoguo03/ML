import re
import numpy as np

def createVocablist(dataSet):
    vocabSet = set()
    for doc in dataSet:
        vocabSet.update(doc)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    vec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            index = vocabList.index(word)
            vec[index] = 1
        else:
            print(' the word not in the vob', word)
    return vec

def bagOfWords2Vec(vocablist, inputSet):
    vec = [0]*len(vocablist)
    for word in inputSet:
        if word in vocablist:
            index  = vocablist.index(word)
            vec[index] += 1
        else:
            print(' the word not in the vob', word)
    return vec
