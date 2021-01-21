# AbstrEncap
## A naïve implmentation of abstractive summerization (of texts) using LSTMs.

### Introduction
We train our RNN on Cornell's «Newsroom summarization dataset» (cf. https://arxiv.org/pdf/1804.11283.pdf) which is already divided to training, development and test sets. It can be obtained from http://lil.nlp.cornell.edu/newsroom/ after filling a google form. Roughly speaking The dataset contains over milion different texts from news websites and their summarizations (each summarization can be extractive/abstractive or mixed)

As the dataset contains long texts with varied lengths of summaries we perform a prelminiary step before training:  We split each text to paragraohs beginning in a key sentences. These key sentences are nothing but sentences with highest tf-idf score. The quantity of them is exactly the number of sentences consituting the summary.

Then we train an RNN with 3 LSTMs with an implementation of Bahdanau's attention (https://arxiv.org/pdf/1409.0473.pdf). In the future we may test if Loung's attention or usage of local attention can improve the newtork.

### Requirements
All requirements are specified in `requirements.txt`

### Usage
An example of usage is given in the bottom of `abs_sum.py` and depends on calling the function `abs_summarize`.
