import pandas as pd
import numpy as np
from numpy.linalg import norm
from gensim.models import FastText
from operator import itemgetter
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
special_chars = r"[^ A-Za-zéëèàáäőóöœï\s]"
cont = Ctr(api_key='glove-wiki-gigaword-50')
pr_con = 0.5


class Substitutable(str):
    def __new__(cls, *args, **kwargs):
        obj = str.__new__(cls, *args, **kwargs)
        obj.sub = lambda old_str, new_str: Substitutable(re.sub(old_str, new_str, obj))
        return obj


def partition(list_, indices):
    # Check whether the first two sentences are important.
    if 0 not in indices:
        indices.insert(0, 0)

    return [" ".join(list_[idx:jdx]) for idx, jdx in zip(indices, indices[1:] + [None])]


''' Preprocess a column of strings of a given DataFrame in several steps: a. Change all capital letters to lower '''


def clean_text(sentences):
    # a function that removes numbers, special characters and words of a signle letter from every sentence in a list
    clean_sentences = []
    new_vocab = []

    for sentence in sentences:
        subs_sentence = Substitutable(sentence.lower())
        clean_sub_sentence = subs_sentence.sub(special_chars, '').sub(r'\d+', '').sub(r'\b[a-zA-Z]{1}\b', '')
        if clean_sub_sentence:
            str_sentence = str(clean_sub_sentence)
            clean_sentences.append(str_sentence)
            new_vocab = new_vocab + [word for word in str_sentence.split()]
    return new_vocab, clean_sentences


def preprocess_texts(df, txt_col, sum_col):
    processed_df = pd.DataFrame(columns=['Index', txt_col, sum_col])
    vocab = []
    long_sources = []
    for idx, row in df.iterrows():
        text = row[txt_col]
        summary = row[sum_col]
        tokenized_text = sent_tokenize(text)
        txt_vocab, clean_tokenized_text = clean_text(tokenized_text)
        vocab += txt_vocab
        tokenized_sum = sent_tokenize(summary)
        summary_vocab, clean_tokenized_sum = clean_text(tokenized_sum)
        vocab += summary_vocab
        key_sentences_num = len(sent_tokenize(summary))

        line = {'Index': idx, txt_col: ' . '.join(clean_tokenized_text), sum_col: ' . '.join(clean_tokenized_sum),
                'sen_num': key_sentences_num}

        if key_sentences_num == 1:
            processed_df = processed_df.append(line, ignore_index=True)
            continue
        else:
            long_sources.apppend(line)

    model_ft = FastText(sentences=vocab, window=3, min_count=2, size=300)
    model_ft.train(sentences=vocab, total_examples=len(vocab), epochs=30)

    for source in long_sources:
        idx = source['index']
        vectors = vectorize_sentences(source[txt_col], model_ft)
        scores = score_sentences(vectors)
        indices_of_key_sentences = sorted([sentence_idx
                                          for sentence_idx, score in sorted(scores.items(), key=itemgetter(1),
                                           reverse=True)[0:source['sen_num']]])

        partited_text = partition(source[txt_col], indices_of_key_sentences)
        paragraphs_and_sentences = [[partited_text[idx], tokenized_sum[idx]] for idx in range(0, key_sentences_num)]

        for elem in paragraphs_and_sentences:
            line = processed_line(elem[0], elem[1], idx, processed_df)
            processed_df = processed_df.append(line, ignore_index=True)

    print("Preprocessed DataFrame is ready!")
    return processed_df


def processed_line(text, summary, title, new_df):
    exp_text = expand_sentences(text)
    exp_summary = expand_sentences(summary)
    processed_row = pd.Series([title, exp_text, exp_summary], index=new_df.columns)
    # eliminate possesion (The man's children --> The man children) with which Contractions has some problems
    clean_processed_row = processed_row.str.replace("\'s", " ", case=True, regex=True).\
        str.replace("[^a-z A-Z]", " ", case=True, regex=True).\
        str.lower()

    return clean_processed_row


def expand_sentences(text):
    expanded_texts_list = []

    sentences = list(cont.expand_texts(sent_tokenize(text), precise=True))
    expanded_texts_list.append(' '.join(sentences))

    return expanded_texts_list[0]


def vectorize_sentences(text, model):
    vectors = []

    for sentence in text.split('.'):
        tagged_pos_sentence = pos_tagging(sentence)
        vec = np.zeros(len(tagged_pos_sentence))

        for word in tagged_pos_sentence:

            if word not in stop_words:
                word = word_lemmatizer.lemmatize(word)
                vec += word_tfidf(word, text, sentence) * model[word]

        vectors.append(vec)

    return vectors


def score_sentences(vectors):

    idx = 0
    sentences_num = len(vectors)
    scores = np.zeros(sentences_num)

    for vector in vectors:
        cosine_similarities = [vector.dot(another_vector) / (norm(vector) * norm(another_vector))
                               for another_vector in vectors]
        jdx = 0
        score = pr_con

        while jdx < idx:
            sum_out = sum([cosine_similarities[k] for k in range(idx, sentences_num)])
            score += (1 - pr_con) * scores[jdx] * cosine_similarities[jdx] / sum_out
            jdx += 1

        scores[idx] = score

    return dict(zip([idx for idx in range(0, sentences_num)], scores))


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