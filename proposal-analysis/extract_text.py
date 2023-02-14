import json
import numpy as np
import os
from itertools import chain
from glob import iglob
from tqdm import tqdm

import nltk
from nltk import word_tokenize

import spacy
from spacy import displacy
from collections import Counter
import pandas as pd

import string

rootdir = '../data/json_2.1.0'

nlp = spacy.load("en_core_web_sm")

def extract_text():
    '''
    Extract the high-level instruction text from the json files and write to a text file
    '''
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.json') == True:
                with open('all_instructions.txt', 'a') as f:
                    json_f = open(os.path.join(subdir, file))
                    data = json.load(json_f)
                    try:
                        for instructions in data['turk_annotations']['anns']:
                            for line in instructions['high_descs']:
                                # print(line)
                                f.write(line)
                                f.write('\n')
                    except:
                        pass

                    json_f.close()

def get_pos():
    '''
    Test method to get the part of speech of each word in the instructions
    '''
    with open('all_instructions.txt', 'r') as f:
        for line in f:
            tokens = word_tokenize(line)
            tagged = nltk.pos_tag(tokens, tagset = "universal")
            print(tagged)


def get_most_frequent_nltk():
    '''
    Get the top 10 most frequent nouns and verbs using NLTK
    '''
    nouns = []
    verbs = []
    with open('all_instructions.txt', 'r') as f:
        for line in tqdm(f):
            # Remove punctuation, convert to lowercase
            translator = str.maketrans('', '', string.punctuation)
            s = line.translate(translator)
            s = s.lower()

            tokens = word_tokenize(s)
            tagged = nltk.pos_tag(tokens, tagset = "universal")
            for token in tagged:
                if token[1] == 'NOUN':
                    nouns.append(token[0])
                    # print(token[0])
                elif token[1] == 'VERB':
                    verbs.append(token[0])
                    # print(token[0])

    nouns_tally = Counter(nouns)
    verbs_tally = Counter(verbs)

    df_n = pd.DataFrame(nouns_tally.most_common(), columns=['noun', 'count'])
    df_v = pd.DataFrame(verbs_tally.most_common(), columns=['verb', 'count'])
    print(df_n[:20])
    print(df_v[:20])


def get_most_frequent_spacy():
    '''
    Get the top 10 most frequent nouns and verbs using Spacy
    '''
    nouns = []
    verbs = []
    with open('all_instructions.txt', 'r') as f:
        for line in tqdm(f):
            # Remove punctuation, convert to lowercase
            translator = str.maketrans('', '', string.punctuation)
            s = line.translate(translator)
            s = s.lower()

            doc = nlp(s)
            for token in doc:
                if token.pos_ == 'NOUN':
                    nouns.append(token.text)
                    # print(token.text)
                elif token.pos_ == 'VERB':
                    verbs.append(token.text)
                    # print(token.text)

    nouns_tally = Counter(nouns)
    verbs_tally = Counter(verbs)

    df_n = pd.DataFrame(nouns_tally.most_common(), columns=['noun', 'count'])
    df_v = pd.DataFrame(verbs_tally.most_common(), columns=['verb', 'count'])
    print(df_n[:20])
    print(df_v[:20])

if __name__ == '__main__':
    # get_most_frequent_nltk()
    get_most_frequent_spacy()