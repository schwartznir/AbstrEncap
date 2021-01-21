'''
A module dedicated for developing a LSTM-based neural network
used for summerizing a text abstractively.
'''

import pandas as pd
import os
from keras import backend
from parse_dbs import preprocess_texts
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from b_attention import BahdanauAttentionLayer
from tensorflow.keras.callbacks import EarlyStopping
TXT_COL = 'text'
SUM_COL = 'summary'
TITLE_COL = 'title'

training_corpus = pd.read_csv(os.getcwd() + '/Datasets/train-stats.csv', usecols=[TXT_COL, SUM_COL, TITLE_COL])
val_corpus = pd.read_jsonl(os.getcwd() + '/Datasets/dev-stats.jsonl.gz', usecols=[TXT_COL, SUM_COL, TITLE_COL])
# Omit NA if exist (in fact using «df.loc[pd.isna(df["text"]), :].index» at least one text is indeed missing)
training_corpus.dropna(axis=0, inplace=True)
val_corpus.dropna(axis=0, inplace=True)
cleaned_training = preprocess_texts(training_corpus, TXT_COL, SUM_COL)
cleaned_val = preprocess_texts(val_corpus, TXT_COL, SUM_COL)

max_len_text = 10000 #TODO: CHECK ME!!!!!!
max_len_summary = 200 #TODO: CHECK ME AS WELL!!!!!!
x_tr = cleaned_training['text']
# Initialize a tokenizer
x_tokenizer = Tokenizer()
x_tr = x_tokenizer.fit_on_texts(list(x_tr))
# embed a text as a sequence
x_tr = pad_sequences(x_tr,  maxlen=max_len_text, padding='post')
x_val = cleaned_val['text']
x_val = x_tokenizer.fit_on_texts(list(x_val))
x_val = pad_sequences(x_val, maxlen=max_len_text, padding='post')
x_voc_size = len(x_tokenizer.word_index) + 1
# Doing the same for summaries
y_tr = cleaned_training['summary']
y_val = cleaned_val['summary']
y_tokenizer = Tokenizer()
y_tr = y_tokenizer.fit_on_texts(list(y_tr))
y_val = y_tokenizer.fit_on_texts(list(y_val))
y_tr = y_tokenizer.texts_to_sequences(y_tr)
y_val = y_tokenizer.texts_to_sequences(y_val)
y_tr = pad_sequences(y_tr, maxlen=max_len_summary, padding='post')
y_val = pad_sequences(y_val, maxlen=max_len_summary, padding='post')
y_voc_size = len(y_tokenizer.word_index) + 1
# Clear previous keras models
backend.clear_session()
latent_dim = 500
# Define an encoder
in_enc = Input(shape=(max_len_text,))
emb_enc = Embedding(x_voc_size, latent_dim, trainable=True)(in_enc)
# We define now 3 LSTMs
lstm1_enc = LSTM(latent_dim, return_sequences=True, return_state=True)
lstm2_enc = LSTM(latent_dim, return_sequences=True, return_state=True)
lstm3_enc = LSTM(latent_dim, return_sequences=True, return_state=True)
out1_enc, state_h1, state_c1 = lstm1_enc(emb_enc)
out2_enc, state_h2, state_c2 = lstm2_enc(out1_enc)
outs_enc, state_h, state_c = lstm3_enc(out2_enc)
# Define a decoder and its LSTM.
in_dec = Input(shape=(None,))
dec_emb_layer = Embedding(y_voc_size, latent_dim, trainable=True)
emb_dec = dec_emb_layer(in_dec)
lstm_dec = LSTM(latent_dim, return_sequences=True, return_state=True)
outs_dec, decoder_fwd_state, decoder_back_state = lstm_dec(emb_dec, initial_state=[state_h, state_c])
# Attention Layer -- the Bahdanau "flavour"
attn_layer = BahdanauAttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([outs_enc, outs_dec])
# Concat attention output and decoder LSTM output
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([outs_dec, attn_out])
# Dense layer
decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)
# Finalize the model def (and display a summary)
model = Model([in_enc, in_dec], decoder_outputs)
model.summary()
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
early_stop = EarlyStopping(monitor= 'val_loss', mode='min', verbose=1)

history=model.fit([x_tr, y_tr[:, :-1]], y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:, 1:],
                  epochs=50, callbacks=[early_stop], batch_size=512, validation_data=([x_val, y_val[:, :-1]],
                  y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:, 1:]))
