import os
import nltk
import re
import glob
from nltk.corpus import wordnet



def extractTrain(path):

    training_data = []          
    for file_name in glob.glob(path, recursive=True):
        data = open(file_name).read()
        training_data.extend(getTrainFeatures(data))

    return training_data


def getTrainFeatures(data):
    train_data = []

    tokenized_data = nltk.word_tokenize(data)      
    pos_tokenized_data = nltk.pos_tag(tokenized_data)

    chk_tagged_tokens = nltk.chunk.ne_chunk(pos_tokenized_data)

    for chk in chk_tagged_tokens.subtrees():
        features = {}  
        if chk.label().upper() == 'PERSON':  
            name = ' '.join([i[0] for i in chk])

            features['name_len_s'] = len(name)  
            features['name_len'] = len(name.replace(' ', ''))
            features['word_cnt'] = len(name.split(' '))
            features['white_space'] = len(name) - len(name.replace(' ', ''))
            features['w1_len'] = 0  
            features['w2_len'] = 0  
            features['w3_len'] = 0  
            features['w4_len'] = 0 
            words = name.split(' ')

            for i in range(len(words)):  
                if i == 0:
                    features['w1_len'] = len(words[i])
                elif i == 1:
                    features['w2_len'] = len(words[i])
                elif i == 2:
                    features['w3_len'] = len(words[i])
                elif i == 3:
                    features['w4_len'] = len(words[i])

            train_data.append((features, name))

    return train_data



def extractRedacted(path):

    redacted_data = []          

    data = open(path).read()        
    redacted_data.extend(getRedactedFeatures(data))

    return redacted_data  


def getRedactedFeatures(data):

    redacted_names = []  
    pattern = re.compile('\u2588+\s*\u2588*\s*\u2588*\s*\u2588+')

    matched_names = re.findall(pattern, data)

    for names in matched_names:
        features = {}

        name = re.sub('\s+', ' ',  names)

        features['name_len_s'] = len(name)  
        features['name_len'] = len(name.replace(' ', ''))
        features['word_cnt'] = len(name.split(' '))
        features['white_space'] = len(name) - len(name.replace(' ', ''))
        features['w1_len'] = 0
        features['w2_len'] = 0  
        features['w3_len'] = 0  
        features['w4_len'] = 0  
        words = name.split(' ')

        for i in range(len(words)): 
            if i == 0:
                features['w1_len'] = len(words[i])
            elif i == 1:
                features['w2_len'] = len(words[i])
            elif i == 2:
                features['w3_len'] = len(words[i])
            elif i == 3:
                features['w4_len'] = len(words[i])

        redacted_names.append((features, names))
    return redacted_names
