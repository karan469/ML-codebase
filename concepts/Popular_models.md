# Popular models and frameworks

## 1. Vowpal Wabbit
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

## 2. Blazing Text
- Highly optimized implementations of Word2Vec and text classification (meaning both unsupervised and supervised training)
- Input: A single preprocessed text file with space-separated tokens where the training file should contain a training sentence per line along with the labels; labels are words that are prefixed by the string "label"
- PS: Word2Vec maps words to high-quality distributed vectors (a word embedding), such that words that are semantically similar correspond to vectors that are close together
- Blazing Text provides the Skip-gram and continuous bag-of-words (CBOW) training architectures
- Both skip-gram and cbow can be used

## 3. Fasttext
- Library for learning word embeddings as well as classify.
- Facebook's algorithm. Provided 294 trained models
