import csv
import os
import sys
import numpy as np
import re
from nltk import SnowballStemmer

regEx = re.compile('\\W*')
regNumber = re.compile('^\\d+$')
regStartNumber = re.compile('^\\d+')
regEndNumber = re.compile('\\d+$')
stemmer = SnowballStemmer("english")

def GetTokenList(text):
    rawWordList = regEx.split(text)
    wordDict = {};
    wordList =[tok.lower() for tok in rawWordList if len(tok) > 2 and len(tok) < 20 and regNumber.match(tok)==None]
    for word in wordList:
        wordStem = word
        #Remove numbers begein and end of the wordDict
        startNumberMatch =  regStartNumber.match(wordStem)
        if startNumberMatch != None:
            wordStem = wordStem[startNumberMatch.span()[1]:]
        endNumberMatch = re.search('\\d+$',wordStem)
        if endNumberMatch != None:
            wordStem = wordStem[:endNumberMatch.span()[0]]
        wordStem = stemmer.stem(wordStem)
        if len(wordStem) >2 :
            wordDict[wordStem] = 1
    return wordDict.kes()

def GetAllTokenDict(fileName):
    pass
def GetTextVector(tokenList, allTokenDict):
    
    #Init all value to -0.5
    for key in allTokenDict.keys():
        allTokenDict[key] = -0.5
        
    for token in tokenList:
        if token in allTokenDict:
            allTokenDict[token] = 0.5  
    return [allTokenDict[key] for key in allTokenDict.keys()]
     
      