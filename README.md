#  Overview
The project looks to deal with the problems faced by many sentiment analysers which is Sarcasm Detection by introducing a novel method MHA-BiLSTM deep learning approach with an easy user interface for communication. We implemented a general purpose sarcasm detection web application to recognize any text based sarcastic sentence using a Multi-Head Attention based Bidirectional Long-Short Term Memory (MHA-BiLSTM). User can also be able to use Discord alongside a BOT which will detect if the conversation is Sarcastic or not.

# Steps
## 1)Text Preprocessing and Glove Embedding matrix:
Using the text/csv file entered by the user, we preprocess the data by eliminating irrelevant aspects of the sentence (entire sentence can also be discarded) and generate vector representation of every word using Glove Embedding Matrix.

## 2)Working of MHA-BiLSTM and its analysis:
The preprocessed sentences are then put through the MHA-BiLSTM model for training and the best matrix is stored after every iteration through our test dataset. Thus, we achieve higher levels of Precision, Recall and F-Score on our dataset. Additional tweaking to the weights,parameters and hardware we can achieve better results overall.

## 3)Sarcasm Detection:
Using the best saved matrix in the system, the model will detect if there is sarcasm in the sentence or not and then the user will be notified accordingly.

#  Prerequisites
Python 3.6, sklearn, pytorch, flask, discord api
