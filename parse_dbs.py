import pandas as pd
import tarfile
from itertools import chain
from operator import itemgetter
import numpy as np
from pycontractions import Contractions as Ctr
import nltk
import os
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
CWD = os.getcwd()
TARNAME = 'newsroom-release.tar'
TEST_GZ = CWD+'/Databases/release/test.jsonl.gz'
word_lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))
# We allow "foreign" letters in order to handle words with foreign etymology like «naïve»
special_chars = r"^A-Za-zéëèàáäőóöœï\s"
cont = Ctr(api_key='glove-twitter-200')
#tar_db = tarfile.open(CWD+'/Databases/'+TARNAME)
#tar_db.extractall(path=CWD+'/Databases')


def partition(list_, indices):
    if 0 not in indices: indices.append(0)
    #indices.append(len(list))
    return [" ".join(list_[idx:jdx]) for idx, jdx in zip(indices, indices[1:] + [None])]


''' Preprocess a column of strings of a given DataFrame in several steps: a. Change all capital letters to lower '''


def preprocess_texts(df, txt_col, sum_col):
    counter = 0
    processed_df = pd.DataFrame(columns=['title', 'text', 'summary', 'chunk of text no.'])

    for idx, row in df.iterrows():
        text = row[txt_col]
        summary = row[sum_col]
        tokenized_text = sent_tokenize(text)
        key_sentences_num = len(sent_tokenize(summary))
        scores_sentences = {tokenized_text.index(sentence): score_sentences(sentence, tokenized_text)
                            for sentence in tokenized_text}
        indices_of_key_sentences = sorted([sentence_idx
                                    for sentence_idx, score in sorted(scores_sentences.items(), key=itemgetter(1),
                                                                  reverse=True)[0:key_sentences_num]])

        partited_text = partition(tokenized_text, indices_of_key_sentences)
        split_summary = sent_tokenize(summary)
        paragraphs_and_sentences = [[partited_text[idx], split_summary[idx]] for idx in range(0, key_sentences_num)]

        for elem in paragraphs_and_sentences:
            elem[0] = expand_sentences(elem[0])
            elem[1] = expand_sentences(elem[1])
            processed_row = pd.Series([df['title'][counter], elem[0], elem[1], 0], index=processed_df.columns)
            # eliminate possesion (The man's children --> The man children) with which Contractions has some problems
            clean_processed_row = processed_row.str.replace("\'s", "", case=True, regex=True)
            clean_processed_row = clean_processed_row.str.replace("[^a-z A-Z]", "", case=True, regex=True)
            clean_processed_row = clean_processed_row.str.lower()
            clean_processed_row['chunck of text no.'] = counter
            processed_df = processed_df.append(clean_processed_row, ignore_index=True)

        counter += 1

    print("Preprocessed DataFrame is ready!")
    return processed_df


def expand_sentences(text):
    expanded_texts_list = []

    sentences = list(cont.expand_texts(sent_tokenize(text), precise=True))
    expanded_texts_list.append(' '.join(sentences))

    return expanded_texts_list[0]

'''
def contraction_expansion():

    

    download('stopwords')  # Download stopwords list.
    stop_words = stopwords.words('english')
    
    expanded_texts = expand_sentences(df[txt_col])
    expanded_summaries = expand_sentences(df[sum_col])
    return pd.DataFrame({ txt_col: expanded_texts, sum_col: expanded_summaries})
'''

def score_sentences(sentence, sentences):
    score = 0

    sentence_wo_special_chars = re.sub(special_chars, '', sentence.lower())
    clean_sentence = re.sub(r'\d+', '', sentence_wo_special_chars)
    tagged_pos_sentences = pos_tagging(clean_sentence)

    for word in tagged_pos_sentences:
        if word not in stop_words and word.lower() not in stop_words and len(word) > 1:
            word = word_lemmatizer.lemmatize(word.lower())
            score = score + word_tfidf(word, sentences, clean_sentence)

    return score


def pos_tagging(text):
    pos_tag = nltk.pos_tag(text.split())
    tagged_noun_verb = []
    tags = ["NN", "NNP", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

    for word, tag in pos_tag:
        if tag in tags:
            tagged_noun_verb.append(word)

    return tagged_noun_verb


def word_tfidf(word, sentences, sentence):
    tf = tf_score(word, sentence)
    idf = idf_score(len(sentences), word, sentences)
    return tf * idf


def tf_score(word, sentence):
    # The tf score of a word in a sentence
    words_in_sentence = len(sentence.split())

    return sentence.count(word) / words_in_sentence


def idf_score(sentences_num, word, sentences):
    sentences_cont_word = 0

    for sentence in sentences:
        sentence_wo_sc = re.sub(special_chars, '', sentence)
        sentence_wo_nums = re.sub(r'\d+', '', sentence_wo_sc)
        clean_sentence = [word.lower() for word in sentence_wo_nums.split() if
                          word.lower() not in stop_words and len(word) > 1]
        lem_sentence = [word_lemmatizer.lemmatize(w) for w in clean_sentence]
        if word in lem_sentence:
            sentences_cont_word += 1

    return np.log10(sentences_num / (1 + sentences_cont_word))


text='The name of the man\'s Harry Potter. Once upon a time\'s been a man.  \n \"Avada Kadabra\" Harry said to Voldemrt'
summary='Hello. By the holy name!'
example = {'title': ['Hello world!'], 'text': [text], 'summary': [summary], 'bullshit': [1]}
ladf = pd.DataFrame(example)
badf = preprocess_texts(ladf, 'text', 'summary')
tokenized_text = sent_tokenize(text)
key_sentences_num = len(sent_tokenize(summary))
scores_indices = {tokenized_text.index(sentence): score_sentences(sentence, tokenized_text)
                            for sentence in tokenized_text}
indices_of_key_sentences = sorted([idx
                                    for idx, score in sorted(scores_indices.items(), key=itemgetter(1),
                                                                  reverse=True)[0:key_sentences_num]])


print(score_sentences('Name is not their is',['Our are names Tom Riddle', 'Riddle is not their name is a names', 'OH name boy. The holy one']))
