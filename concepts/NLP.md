# Concepts related to Natural Language Processing

In usual Natural Language problems, such as text classification or speech translation, there are two components in the machine learning models.

1. **Embedding Layer**: This is a dictionary which maps tokens to their respective feature vectors. This is usully called word-to-vec phase. There are many frameworks that learn and generate word embeddings, such as, *Fasttext, Blazing Text or just normal tensorflow embedding layers* which have some optimizational difference. These algorithms in their core use either of two approaches: *CBOW or Skip-gram*.
2. **RNN or GRU or LSTM**: These layers use the word embeddings. They contains recurrent neural nodes which can take variable number of tokens as input.

## 1. RNN
1. Can take variable length of input
2. Can use multiple input vectors + output is influenced by, not just weights applied to inputs, but also by <ins>hidden state vectors</ins>.

![image](https://user-images.githubusercontent.com/33158202/118369052-84207500-b5c0-11eb-8f02-8a854dae2571.png)

### 1.1 Parameter sharing
Same weights are applied to different input.
Reason: If we have different weights for different input (vanilla input), then we ought to have fixed input length.

### 1.2 Hidden state
Tries to feed forward the "context" from previous words. This keeps on changing after forwarding with each input.
[img]

### 1.3 Deep RNN
Choices:
1. Increase number of hidden states
2. Add non-linear hidden layers<br>

![image](https://user-images.githubusercontent.com/33158202/118368504-f04ea900-b5bf-11eb-8af1-6fbcfad78241.png)


### 1.4 Biderectional RNN

![image](https://user-images.githubusercontent.com/33158202/118365727-b1b6ef80-b5bb-11eb-931b-a60778e2da78.png)

**Note**

![image](https://user-images.githubusercontent.com/33158202/118365762-d9a65300-b5bb-11eb-9d48-1fff83c3a4b6.png)

At each step, <ins>h is updated</ins> and <ins>y is outputted</ins>.<br>
There could be many use cases, i.e. input and output sequences.

**Case 1: Many to Many**<br>
Take <ins>series of input</ins> and produce <ins>series of output</ins>. For example, stocks prediction or Named entity recognition (NER Tagging).<br>

![image](https://user-images.githubusercontent.com/33158202/118366208-6dc4ea00-b5bd-11eb-9c2a-315af8b0496a.png)
<br>
**Case 2: Many to One**<br>
Feed a sequence of inputs, ignore all outputs except last one. It's also called <span style="color:red">sequence-to-vector network aka Encoder</span>. For example, text classification<br>

![image](https://user-images.githubusercontent.com/33158202/118366319-7fa68d00-b5bd-11eb-998e-085573edde98.png)

**Case 3: One to Many**<br>
Feed a single input at first (and all zeros in other steps) and let it output a sequence of outputs. This is called "vector-to-sequence" network aka Decoder. For example, input is image and output is caption for the image.<br>

![image](https://user-images.githubusercontent.com/33158202/118366461-f04da980-b5bd-11eb-896f-d97584464d91.png)

The image above shows another flavor of the same technique. The only difference is, instead of all zeros for next steps, we give output of the previous input, as the input now.
<br>
**Case 4: Many to Many (but number of input and output channels are different)**<br>
Encoder followed by decoder. For example, language translation.<br>

![image](https://user-images.githubusercontent.com/33158202/118367058-7669f000-b5be-11eb-8744-0b15976e7c3e.png)

### Handling long term dependencies
Commonly used activation functions are Sigmoid, Tanh and RELU. But even after that, we could encounter issues like vanishing gradient. This is because it is difficult to capture long term dependencies because of the multiplicative gradient can be exponentially increasing/decreasing with respect to number of layers.
We can use <ins>Gradient Clipping</ins> which caps the maximum value of gradient and we control exploding gradients.
In order to remedy vanishing gradient problem, we can introduce certain types of gates which have different roles, like, update gate decides how much past should matter now.
<ins>Relevance gates</ins> indicates how much past we shall drop. Likewise, there are <ins>Forget Gates</ins> and <ins>Output Gates</ins>
<br>

![image](https://user-images.githubusercontent.com/33158202/118368466-cf865380-b5bf-11eb-8e28-bd8b9fb73d2f.png)

These gates are used in some complex architectures like GRU and LSTM.<br>

![image](https://user-images.githubusercontent.com/33158202/118368451-c09fa100-b5bf-11eb-920a-548c13cb2ae7.png)

## 2. Popular models and frameworks

### 2.1 Vowpal Wabbit
*vw-varinfo: Summarize features of a training-set using VW*

  **Input**:          A vw training set file<br>
  **Output**:         A list of features, their VW hash values, min/max
                  values, regressor weights, and distance from
                  the best constant.

Algorithm:
  1)  Collect all variables and their ranges from training-set
  2)  Train with VW to determine regressor weights
  3)  Build a test-set with a single example including all variables
  4)  run VW with --audit on 3) to map variable names to hash values
      and weights.
  5)  Output collected information about the input variables.

### 2.2 Blazing Text
- Highly optimized implementations of Word2Vec and text classification (meaning both unsupervised and supervised training)
- Input: A single preprocessed text file with space-separated tokens where the training file should contain a training sentence per line along with the labels; labels are words that are prefixed by the string "label"
- PS: Word2Vec maps words to high-quality distributed vectors (a word embedding), such that words that are semantically similar correspond to vectors that are close together
- Blazing Text provides the Skip-gram and continuous bag-of-words (CBOW) training architectures
- Both skip-gram and cbow can be used

### 2.3 Fasttext
- Library for learning word embeddings as well as classify.
- Facebook's algorithm. Provided 294 trained models

### 2.4 ULMFiT
![image](https://user-images.githubusercontent.com/33158202/120424286-acaeba00-c389-11eb-927c-7dd397338ad2.png)
