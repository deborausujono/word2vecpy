import argparse
import math
import struct
import sys

import numpy as np

from multiprocessing import Process, Value

class VocabItem:
    def __init__(self, word):
        self.word = word
        self.count = 0
        self.path = None # Path (list of indices) from the root to the word (leaf)
        self.code = None # Huffman encoding

class Word2VecTrainer:
    def __init__(self):
        # These will change during training
        self.syn0 = None
        self.syn1 = None
        self.word_count = 0

        # These will stay the same once init
        self.vocab = []                     # List of VocabItem objects
        self.vocab_hash = {}                # Mapping from each token to its index in vocab
        self.table = None                   # List of unigrams to draw negative examples from
        self.train_words = 0
        self.train_bytes = 0

    def __sigmoid(self, z):
        if z > 6:
            return 1.0
        elif z < -6:
            return 0.0
        else:
            return 1 / (1 + math.exp(-z)) # TO DO: Use exp table to speed up computation

    def __init_vocab(self, fi, min_count):
        vocab = []
        vocab_hash = {}

        # Add special tokens <bol> (beginning of line) and <eol> (end of line)
        vocab_idx = 0
        for token in ['<bol>', '<eol>']:
            vocab.append(VocabItem(token))
            vocab_hash[token] = vocab_idx;
            vocab_idx += 1

        word_count = 0 # counting number of words in train file
        for line in fi:
            tokens = line.split()
            for token in tokens:
                if token not in vocab_hash:
                    vocab.append(VocabItem(token))
                    vocab_hash[token] = vocab_idx
                    vocab_idx += 1
                assert vocab[vocab_hash[token]].word == token, 'Wrong vocab_hash index'
                vocab[vocab_hash[token]].count += 1

                word_count += 1
                if word_count % 10000 == 0:
                    sys.stdout.write("\rReading word %d" % word_count)
                    sys.stdout.flush()

            # Incr bol and eol counts
            vocab[vocab_hash['<bol>']].count += 1
            vocab[vocab_hash['<eol>']].count += 1
            word_count += 2

        self.train_bytes = fi.tell()
        #print
        #print vocab[vocab_hash['<bol>']].count, vocab[vocab_hash['<eol>']].count

        # Add special token <unk> (unknown)
        self.vocab.append(VocabItem('<unk>'))
        unk_hash = 0;
    
        # Merge words occurring less than min_count into <unk>
        count_unk = 0
        for token in vocab:
            if token.count < min_count:
                count_unk += 1
                self.vocab[unk_hash].count += token.count
            else:
                self.vocab.append(token)
        

        # Sort vocab in descending order by frequency in train file
        # TO DO: keeping special tokens at the top or not?
        self.vocab.sort(key=lambda token : token.count, reverse=True)

        # Update vocab_hash
        self.vocab_hash = {}
        for i, token in enumerate(self.vocab):
            self.vocab_hash[token.word] = i

        assert word_count == sum([t.count for t in vocab]), 'word_count and sum of t.count do not agree'
        self.train_words = word_count

        print
        print 'Total words in training file: %d' % self.train_words
        print 'Total bytes in training file: %d' % self.train_bytes
        print 'Vocab size: %d' % len(self.vocab)
        print 'Unknown vocab size:', count_unk

    def __init_net(self, dim):
        vocab_size = len(self.vocab)

        # Init syn0 with random numbers from a uniform distribution on the interval [-0.5, 0.5]/dim
        self.syn0 = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(vocab_size, dim))
        #syn0_c = np.ctypeslib.as_ctypes(self.syn0)

        # Init syn1 with zeros
        self.syn1 = np.zeros(shape=(vocab_size, dim))
        #syn1_c = np.ctypeslib.as_ctypes(self.syn1)
    
        # Init syn1neg with zeros
        # self.syn1neg = np.zeros(shape=(vocab_size, dim))
    
    def __init_unigram_table(self):
        """Create a list of indices of unigrams in the vocab following a power law distribution.
        The list will be used to draw negative samples."""
        vocab_size = len(self.vocab)
        power = 0.75
        norm = sum([math.pow(u.count, power) for u in self.vocab]) # Normalizing constant
    
        table_size = 1e8 # Length of the unigram list
        self.table = []
    
        p = 0 # Cumulative probability
        print 'Filling unigram table'
        for i, unigram in enumerate(self.vocab):
            p += float(math.pow(unigram.count, power))/norm
            while len(self.table) < table_size and float(len(self.table)) / table_size < p:
                self.table.append(i)

        print len(self.table)
    
        print 'Filling remaining of unigram table'
        while len(self.table) < table_size: # Is this necessary?
            self.table.append(vocab_size - 1)

        print len(self.table)
    
        # TO DO: Needs further debugging
        """print 'Checking no unigram missing from the table'
        # Make sure that all unigrams in the vocab are present at least once in the table
        assert all([i in self.table for i in range(vocab_size)]), 'Unigrams missing from the table'"""
    
    def __encode_huffman(self):
        # Build a Huffman tree
        vocab_size = len(self.vocab)
        count = [t.count for t in self.vocab] + [1e15] * (vocab_size - 1)
        parent = [0] * (2 * vocab_size - 2)
        binary = [0] * (2 * vocab_size - 2)
        pos1 = vocab_size - 1
        pos2 = vocab_size
    
        for i in range(vocab_size - 1):
            # Find min1
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min1 = pos1
                    pos1 -= 1
                else:
                    min1 = pos2
                    pos2 += 1
            else:
                min1 = pos2
                pos2 += 1
    
            # Find min2
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min2 = pos1
                    pos1 -= 1
                else:
                    min2 = pos2
                    pos2 += 1
            else:
                min2 = pos2
                pos2 += 1
    
            count[vocab_size + i] = count[min1] + count[min2]
            parent[min1] = vocab_size + i
            parent[min2] = vocab_size + i
            binary[min2] = 1
    
        # Assign binary code and path pointers to each vocab word
        root_idx = 2 * vocab_size - 2
        for i, token in enumerate(self.vocab):
            path = [] # List of indices from the leaf to the root
            code = [] # Binary Huffman encoding from the leaf to the root
    
            node_idx = i
            while node_idx < root_idx:
                if node_idx >= vocab_size: path.append(node_idx)
                code.append(binary[node_idx])
                node_idx = parent[node_idx]
            path.append(root_idx)
    
            # These are path and code from the root to the leaf
            token.path = [j - vocab_size for j in path[::-1]]
            token.code = code[::-1]
    
    def __train_thread(self, fi, cbow, neg, dim, starting_alpha, epoch, win): # self.syn0, self.syn1, self.syn1neg, num_threads, thread_id
        # Set fi to point to the right chunk of the training file
        # TO DO: Needs further debugging, multithreading does NOT work!!
        # Use 3 word_counts: self.train_word_count, thread_word_count, last_thread_word_count
        start = 0 #self.train_bytes / num_threads * thread_id
        end = self.train_bytes #- 1 if thread_id == num_threads - 1 else self.train_bytes / num_threads * (thread_id + 1)
        #print 'Thread %d begins at %d, ends at %d' % (thread_id, start, end)
        fi.seek(start)
        """if thread_id > 0: print fi.readline()
        #print self.word_count.value, (0 == self.syn1).all()"""
        alpha = starting_alpha # * (1 - float(self.word_count.value)/self.train_words)

        #word_count = 0 # should be shared for all threads
        while fi.tell() < end: #<= end: # for line in fi:
            offset = fi.tell()
            line = fi.readline().strip()
            if not line:
                print "Blank line at %d of %d bytes: %s\n" % (offset, self.train_bytes, line)
                continue

            # Init sent, a list of indices of words in the line
            sent = [self.vocab_hash[token] if token in self.vocab_hash else self.vocab_hash['<unk>'] for token in line.split()]
            sent = [self.vocab_hash['<bol>']] + sent + [self.vocab_hash['<eol>']] # Add special tokens <bol> and <eol>
            # Does not work unless count of bol and eol >= 5
    
            for sent_pos, token in enumerate(sent):
                if self.word_count % 10000 == 0:
                    # Recalculate alpha
                    alpha = starting_alpha * (1 - float(self.word_count)/self.train_words)
                    if alpha < starting_alpha * 0.0001: alpha = starting_alpha * 0.0001
    
                    # Print progress info
                    sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                                     (alpha, self.word_count, self.train_words, float(self.word_count)/self.train_words * 100))
                    sys.stdout.flush()
    
                # Randomize window size, where win is the max window size
                current_win = np.random.randint(low=1, high=win+1)
                context_start = max(sent_pos - current_win, 0)
                context_end = min(sent_pos + current_win + 1, len(sent))
                context = sent[context_start:sent_pos] + sent[sent_pos+1:context_end]
    
                if cbow:
                    # Compute neu1
                    neu1 = np.mean(np.array([self.syn0[c] for c in context]), axis=0)
                    assert len(neu1) == dim, 'neu1 and dim do not agree'
    
                    # Init neu1e with zeros
                    neu1e = np.zeros(dim)
    
                    # Compute neu1e and update self.syn1
                    classifiers = range(neg + 1) if neg > 0 else zip(self.vocab[token].path, self.vocab[token].code)
                    for i in classifiers:
                        if neg > 0:
                            if i == 0:
                                target = token
                                label = 1
                            else:
                                target_idx = np.random.randint(low=0, high=len(self.table))
                                target = self.table[target_idx]
                                label = 0
                        else:
                            target, label = i
    
                        z = np.dot(neu1, self.syn1[target])
                        p = self.__sigmoid(z)
                        g = alpha * (label - p)
                        neu1e += g * self.syn1[target] # Error to backpropagate to self.syn0
                        self.syn1[target] += g * neu1 # Update self.syn1
    
                    """
                    if (neg > 0):
                        for k in range(neg)+1:
                            if k == 0:
                                target = token
                                label = 1
                            else:
                                neg_sample = np.random.randint(low=0, high=len(table))
                                target = table[neg_sample]
                                label = 0
    
                            z = np.dot(neu1, self.syn1neg[target])
                            p = sigmoid(z)
                            g = alpha * (label - p)
                            neu1e += g * self.syn1neg[target] # Error to backpropagate to self.syn0
                            self.syn1neg[target] += g * neu1 # Update self.syn1neg
                    else:
                        for target, binary in zip(token.path, token.code):
                            label = binary
                            z = np.dot(neu1, self.syn1[target])
                            p = sigmoid(z)
                            g = alpha * (label - p)
                            neu1e += g * self.syn1[target]
                            self.syn1[target] += g * neu1
                    """
    
                    # Update self.syn0
                    for context_word in context:
                        self.syn0[context_word] += neu1e
    
                else:
                    for context_word in context:
                        # Init neu1e with zeros
                        neu1e = np.zeros(dim)
    
                        # Compute neu1e and update self.syn1
                        classifiers = range(neg + 1) if neg > 0 else zip(self.vocab[token].path, self.vocab[token].code)
                        for i in classifiers:
                            if neg > 0:
                                if i == 0:
                                    target = token
                                    label = 1
                                else:
                                    target_idx = np.random.randint(low=0, high=len(self.table))
                                    target = self.table[target_idx]
                                    label = 0
                            else:
                                target, label = i
    
                            z = np.dot(self.syn0[context_word], self.syn1[target])
                            p = self.__sigmoid(z)
                            g = alpha * (label - p)
                            neu1e += g * self.syn1[target] # Error to backpropagate to self.syn0
                            self.syn1[target] += g * self.syn0[context_word] # Update self.syn1
    
                        # Update self.syn0
                        self.syn0[context_word] += neu1e
    
                self.word_count += 1
    
        # Print progress info
        sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                         (alpha, self.word_count, self.train_words, float(self.word_count)/self.train_words * 100))
        sys.stdout.flush()
        print
        print line
    
    def save(self, fo, binary):
        dim = len(self.syn0[0])
        if binary:
            fo = open(fo, 'wb')
            fo.write('%d %d\n' % (len(self.syn0), dim))
            fo.write('\n')
            for token, vector in zip(self.vocab, self.syn0):
                fo.write('%s ' % token.word)
                for s in vector:
                    fo.write(struct.pack('f', s))
                fo.write('\n')
        else:
            fo = open(fo, 'w')
            fo.write('%d %d\n' % (len(self.syn0), dim))
            for token, vector in zip(self.vocab, self.syn0):
                word = token.word
                vector_str = ' '.join([str(s) for s in vector])
                fo.write('%s %s\n' % (word, vector_str))

        fo.close()
    
    def train(self, fi, cbow, neg, dim, alpha, epoch, win, min_count, num_threads):
        # Open input file
        fi = open(fi, 'r')
    
        self.__init_vocab(fi, min_count)
        self.__init_net(dim)
        if neg > 0:
            print 'Initializing unigram table'
            self.__init_unigram_table()
        else:
            self.__encode_huffman()

        self.__train_thread(fi, cbow, neg, dim, alpha, epoch, win) #, num_threads, i
        """print 'Starting %d jobs' % num_threads
        jobs = []
        for i in range(num_threads):
            p = Process(target=self.__train_thread,
                        args=(fi, cbow, neg, dim, alpha, epoch, win, num_threads, i))
            jobs.append(p)
            p.start()
            p.join()
        
        #for job in jobs:
        #    job.join()

        # free syn0_c and syn1_c"""
    
        fi.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='Training file', dest='fi', required=True)
    parser.add_argument('-model', help='Output model file', dest='fo', required=True)
    parser.add_argument('-cbow', help='1 for CBOW, 0 for skip-gram', dest='cbow', default=1, type=int)
    parser.add_argument('-negative', help='Number of negative examples (>0) for negative sampling, 0 for hierarchical softmax', dest='neg', default=5, type=int)
    parser.add_argument('-dim', help='Dimensionality of word embeddings', dest='dim', default=100, type=int)
    parser.add_argument('-alpha', help='Starting alpha', dest='alpha', default=0.025, type=float)
    parser.add_argument('-epoch', help='Number of training epochs', dest='epoch', default=1, type=int)
    parser.add_argument('-window', help='Max window length', dest='win', default=5, type=int) 
    parser.add_argument('-min-count', help='Min count for words used to learn <unk>', dest='min_count', default=5, type=int)
    parser.add_argument('-num-threads', help='Number of threads', dest='num_threads', default=1, type=int)
    parser.add_argument('-binary', help='1 for output model in binary format, 0 otherwise', dest='binary', default=0, type=int)
    args = parser.parse_args()

    trainer = Word2VecTrainer()
    trainer.train(args.fi, bool(args.cbow), args.neg, args.dim, args.alpha, args.epoch, args.win,
                  args.min_count, args.num_threads) # Learn word embeddings
    trainer.save(args.fo, bool(args.binary))        # Save learned embeddings to output file
