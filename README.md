
## Language Modelling

Language models determine word probability by analyzing text data. They interpret this data by feeding it through an algorithm that establishes rules for context in natural language. Then, the model applies these rules in language tasks to accurately predict or produce new sentences. The model essentially learns the features and characteristics of basic language and uses those features to understand new phrases.

This is an advancement to a text summarizer project which I did earlier. Check that out here:

## Tokenization
Tokenization is a way of separating a piece of text into smaller units called tokens. Here, tokens can be either words, characters, or subwords. Hence, tokenization can be broadly classified into 3 types – word, character, and subword (n-gram characters) tokenization.

For example, consider the sentence: “Never give up”.

The most common way of forming tokens is based on space. Assuming space as a delimiter, the tokenization of the sentence results in 3 tokens – Never-give-up. As each token is a word, it becomes an example of Word tokenization.



## RNN
A recurrent neural network (RNN) is a type of artificial neural network which uses sequential data or time series data. These deep learning algorithms are commonly used for ordinal or temporal problems, such as language translation, natural language processing (nlp), speech recognition, and image captioning

Recurrent neural networks leverage backpropagation through time (BPTT) algorithm to determine the gradients, which is slightly different from traditional backpropagation as it is specific to sequence data. The principles of BPTT are the same as traditional backpropagation, where the model trains itself by calculating errors from its output layer to its input layer. These calculations allow us to adjust and fit the parameters of the model appropriately. BPTT differs from the traditional approach in that BPTT sums errors at each time step whereas feedforward networks do not need to sum errors as they do not share parameters across each layer.
In the process of BPTT,the RNNs suffer from either of the two problems

Exploding Gradient: occur when the gradient is too large, and continues to become larger in the BPTT thus creating an unstable model. In this case, the model weights will grow too large, and they will eventually be represented as NaN.
Vanishing Gradient: occur when the gradient is too small, it continues to become smaller in the BPTT, updating the weight parameters until they become diminshingly small tending towards 0 and eventually become 0.
To address the exploding gradient problem we can clip the gradients to threshold value, however, for vanishing gradient problem we make certain changes to the simple RNN architecture.


## LSTM and GRUs
Long Short Term Memory (LSTM): This is a popular RNN architecture designed to address the problem of long term dependencies. if the previous state that is influencing the current prediction is not in the recent past, the RNN model may not be able to accurately predict the current state. As an example, let’s say we wanted to predict the italicized words in following, “Alice is allergic to nuts. She can’t eat peanut butter.” The context of a nut allergy can help us anticipate that the food that cannot be eaten contains nuts. However, if that context was a few sentences prior, then it would make it difficult, or even impossible, for the RNN to connect the information. To remedy this, LSTMs have “cells” in the hidden layers of the neural network, which have three gates–an input gate, an output gate, and a forget gate. These gates control the flow of information which is needed to predict the output in the network.

Gated Recurrent Units (GRUs): This RNN variant is similar the LSTMs as it also works to address the short-term memory problem of RNN models. Instead of using a “cell state” regulate information, it uses hidden states, and instead of three gates, it has two—a reset gate and an update gate. Similar to the gates within LSTMs, the reset and update gates control how much and which information to retain.

Bidirectional recurrent neural networks (BRNN): These are a variant network architecture of RNNs. While unidirectional RNNs can only drawn from previous inputs to make predictions about the current state, bidirectional RNNs pull in future data to improve the accuracy of it. If we return to the example of “feeling under the weather” earlier in this article, the model can better predict that the second word in that phrase is “under” if it knew that the last word in the sequence is “weather.”


## Greedy search Decoding
In greedy search, at each decoding step, the decoder selects the token with the highest probability as the next token in the output sequence. This process is repeated until an end-of-sequence token is generated, indicating that the output sequence is complete.

## Beam Search Decoding
The algorithm works by maintaining a set of partially decoded sequences, called the beam, with each sequence represented by a node in the search tree. At each time step, the decoder generates a set of possible candidates by expanding the beam nodes and computing their conditional probabilities. The beam width limits the number of candidates to consider at each time step, and only the candidates with the highest conditional probabilities are retained in the beam.
The algorithm continues to generate candidates and update the beam until the end-of-sequence token is generated, at which point the candidate with the highest joint probability is selected as the output sequence.


## Model used
Here we have used a RNN (LSTM model) for training and testing purposes.
## Libraries and Usage

```
import pandas as pd 
import numpy as np
import re
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

tqdm.pandas()
sns.set_style('dark')
plt.rcParams['figure.figsize'] = (20,8)
plt.rcParams['font.size'] = 14


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Embedding, LSTM, add, Concatenate, Reshape,concatenate, Bidirectional, Dense, Input)

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler


```






## Accuracy

Initialy the model was overfit, and it can be seen by the low convergance graph here:

So, we used advanced techniques like Greedy search decoding and beam search decoding in orfer to predict our results accuractely.

The validation loss is around 3.64, which meacns we have very high accuracy.




## Run Locally

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  npm install
```

Start the server

```bash
  npm run start
```


## Used By
In the real world, this project is uded in the NLP industry,a nd is very funadmental for various E-Commerce businesses to improve their procut and client engagement by virtue of customer reviews.
## Appendix

A very crucial project in the realm of data science and NLP using visualization techniques as well as machine learning modelling.

## Acknowledgements

The project is taken from
https://www.kaggle.com/code/quadeer15sh/introduction-to-language-modelling-using-rnns
## Tech Stack

**Client:** Python, Decoding, RNN, machine learning, sequential model of ML, LSTM model, tensorflow, data visualization libraries of python.



## Feedback

If you have any feedback, please reach out to us at chawlapc.619@gmail.com

