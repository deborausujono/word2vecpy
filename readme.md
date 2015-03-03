A Python implementation of the Continuous Bag of Words (CBOW) and skip-gram neural network architectures, and the hierarchical softmax and negative sampling learning algorithms for efficient learning of word vectors (Mikolov, et al., 2013a, b, c; http://code.google.com/p/word2vec/).

Usage
-----
To train word vectors:
```
word2vec.py [-h] -train FI -model FO [-cbow CBOW] [-negative NEG]
            [-dim DIM] [-alpha ALPHA] [-window WIN]
            [-min-count MIN_COUNT] [-processes NUM_PROCESSES]
            [-binary BINARY]

required arguments:
  -train FI                 Training file
  -model FO                 Output model file

optional arguments:
  -h, --help                show this help message and exit
  -cbow CBOW                1 for CBOW, 0 for skip-gram
  -negative NEG             Number of negative examples (>0) for negative sampling, 
                            0 for hierarchical softmax
  -dim DIM                  Dimensionality of word embeddings
  -alpha ALPHA              Starting learning rate
  -window WIN               Max window length
  -min-count MIN_COUNT      Min count for words used to learn <unk>
  -processes NUM_PROCESSES  Number of processes
  -binary BINARY            1 for output model in binary format, 0 otherwise
```
Each sentence in the training file is expected to be newline separated. 

Implementation Details
----------------------
Written in Python 2.7.6 and NumPy 1.9.1.

Evaluation
----------
Accuracy (%) on the word analogy task compared against the original C implementation (in parentheses). Trained on a preprocessed version of the first 10<sup>8</sup> bytes of the English Wikipedia dump on March 3, 2006 (http://mattmahoney.net/dc/textdata.html).

| Model        | Total         | Semantic      | Syntactic     |
|:------------ |--------------:| -------------:| -------------:|
| CBOW HS      | 6.76 (6.90)   | 4.86 (3.61)   | 7.93 (8.93)   |
| CBOW NS      | 4.52 (6.72)   | 3.94 (3.74)   | 4.88 (8.56)   |
| Skip-gram HS | 14.76 (14.59) | 11.40 (10.40) | 16.83 (17.18) |       
| Skip-gram NS | 8.43 (7.72)   | 4.91 (4.62)   | 10.62 (9.63)  |

References
----------
Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013a). Distributed representations of words and phrases and their compositionality. Advances in Neural Information Processing Systems. http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013b). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781. http://arxiv.org/pdf/1301.3781.pdf

Mikolov, T., Yih, W., & Zweig, G. (2013c). Linguistic Regularities in Continuous Space Word Representations. HLT-NAACL. http://msr-waypoint.com/en-us/um/people/gzweig/Pubs/NAACL2013Regularities.pdf
