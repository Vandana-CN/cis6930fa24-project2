import nltk
import re
from nltk.corpus import wordnet


def redactNames(path):
    print('redactnames')
    data = open(path).read()  

    tokenized_data = nltk.word_tokenize(data)       
    print('tokenized')
    pos_tokenized_data = nltk.pos_tag(tokenized_data)

    
    chk_tagged_tokens = nltk.chunk.ne_chunk(pos_tokenized_data)

    for chk in chk_tagged_tokens.subtrees():

        if chk.label().upper() == 'PERSON':  
            for name in chk:
                
                data = re.sub('\\b{}\\b'.format(name[0]),
                              '\u2588'*len(name[0]), data)  
   
    redactedDoc = open(path.replace('.txt', '.redacted'), 'w')
    redactedDoc.write(data) 

    redactedDoc.close()
    print('done with names')
    return path.replace('.txt', '.redacted')
