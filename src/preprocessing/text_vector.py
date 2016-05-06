import csv
import os
import sys
import numpy as np
import re
from nltk import SnowballStemmer

def CreateTextVector():
    fileName='ocr_result _1.csv'
    wordDict = {};
    csvfile = file(fileName, 'rb')
    reader = csv.reader(csvfile)
    regEx = re.compile('\\W*')
    regNumber = re.compile('^\\d+$')
    regStartNumber = re.compile('^\\d+')
    regEndNumber = re.compile('\\d+$')
    stemmer = SnowballStemmer("english")
    for line in reader:
        if len(line) > 0:
            currentFileName = line[0]
            content = line[1]
            rawWordList = regEx.split(content)
            wordList =[tok.lower() for tok in rawWordList if len(tok) > 2 and len(tok) < 20 and regNumber.match(tok)==None]
            #print wordList
            #print len(wordList)
            for currentWord in wordList:
                wordStem = stemmer.stem(currentWord)
                #Remove numbers begein and end of the wordDict
                startNumberMatch =  regStartNumber.match(wordStem)
                if startNumberMatch != None:
                    wordStem = wordStem[startNumberMatch.span()[1]:]
                endNumberMatch = re.search('\\d+$',wordStem)
                if endNumberMatch != None:
                    wordStem = wordStem[:endNumberMatch.span()[0]]
                if len(wordStem) >2 :
                    if wordStem in wordDict:
                        wordDict[wordStem] = wordDict[wordStem] + 1
                    else:
                        wordDict[wordStem] = 1
            #break;
    csvfile.close()
    sortedList = sorted(wordDict.iteritems(), key=lambda d:d[1], reverse = False)
    keys = [item[0] for item in sortedList if item[1]>2]
    #print len(keys)
    print keys
if __name__ == "__main__":  
    #ResizeImages("../sample_data/gray_images","../sample_data/100_100",(100,100))
    #Process_OCR("../../data/gray_images")
    CreateTextVector()
