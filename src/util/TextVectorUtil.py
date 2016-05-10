import csv
import os
import sys
import numpy as np
import re
from nltk import SnowballStemmer
import CSVUtil

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
    return wordDict.keys()

def GetAllTokenDict(fileName):
    #'../../data/all_tokens.csv'
    tokenList = CSVUtil.ReadCSV(fileName)
    tokenDict = {}
    for token in tokenList:
        tokenDict[token[0]] = 0
    #print tokenDict
    return tokenDict
    
def GetTextVector(tokenList, allTokenDict):
    #Init all value to -0.5
    for key in allTokenDict.keys():
        allTokenDict[key] = 0
        
    for token in tokenList:
        if token in allTokenDict:
            allTokenDict[token] = 1  
    return [allTokenDict[key] for key in allTokenDict.keys()]

def BuildText2DimArray(listOftokenList, allTokenDict):
    resultArray = np.ndarray(
                shape=[len(listOftokenList),len(allTokenDict)],
                dtype=np.float32)
    for index in range(len(listOftokenList)):
        resultArray[index,]= GetTextVector(listOftokenList[index],allTokenDict)
    return  resultArray   
                    
'''   
if __name__ == "__main__":

    tokenDict = GetAllTokenDict('../../data/all_trainning_tokens.csv')
    tokenList = ['para', 'recuperaren', 'queri', 'existen', 'goa', 'hrecuperando', 'sup', 'date', 'aceptar', 'with']
    exist_list = [item for item in tokenList if item in tokenDict]
    print exist_list
    result = BuildText2DimArray([tokenList],tokenDict)
    print [index for index in range(len(tokenDict)) if result[0,index]>0]
    #GetAllTokenDict('../../data/all_tokens.csv')
    '''