# AbstrEncap
## A naïve implementation of abstractive summarization (of texts) using LSTMs.

### Introduction
We train our RNN on «BigPatent» (cf. https://www.aclweb.org/anthology/P19-1212.pdf) which is already divided to training, development and test sets lying in directories with names a-h,y. The dataset can be obtained from https://evasharma.github.io/bigpatent/ after downloading it from the google drive link. Roughly speaking The dataset contains millions of descriptions and abstracts of patents.

As the dataset contains long texts with varied lengths of summaries we perform a preliminary step before training:  We split each text to paragraphs beginning in a key sentences. In order to obtain the key sentences in each paragraph we:
0. Create a list of sentences for the text
1. calculate for each sentence a representing vector. This vector is is obtained by multiplying every word2vec vector of each word by the tf-idf score of the word inside the text. 
2. Apply a forward "PageRanking" method (cf. https://www.aclweb.org/anthology/P04-3020.pdf), with cosine similarity.
3. Suppose the "summary" (abstract) of the text contains N sentences, we say a sentence in the original text is a key sentence if its score is one of the N maximal ones.

Then we train an RNN with 3 LSTMs with an implementation of Bahdanau's attention (https://arxiv.org/pdf/1409.0473.pdf). In the future we may test if Loung's attention or usage of local attention can improve the network.

### Requirements
All requirements are specified in `requirements.txt`

### Usage
An example of usage is given in `abs_sum.py` for reading, parsing and learning `BigPatent` database.
