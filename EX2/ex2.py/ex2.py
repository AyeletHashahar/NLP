"""
API for ex2, implementing the skip-gram model (with negative sampling).


"""


# you can use these packages (uncomment as needed)
import pickle
import pandas as pd
import numpy as np
import os,time, re, sys, random, math
from collections import Counter, defaultdict
from nltk import skipgrams



#static functions
def who_am_i():  # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Ayelet Hashahar Cohen', 'id': '206533895', 'email': 'ayelethc@post.bgu.ac.il'}




def normalize_text(fn):
    """ Loading a text file and normalizing it, returning a list of sentences.


    Args:
        fn: full path to the text file to process
    """
    sentences = []

    # read the text file
    with open(fn, "r") as full_path:
        original_sentences = full_path.read().splitlines()

    for sentence in original_sentences:
        # remove punctuation and convert to lower case
        sentence = re.sub(r"[^a-zA-Z0-9\s]", "", sentence).lower().strip()
        sentence = re.sub(r'\s+', ' ', sentence)  # remove double space
        # split the sentence into words
        if sentence:
            words = sentence.split()
            # add the words to the sentences list
            sentences.append(words)


    return sentences


def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))





def load_model(fn):
    """ Loads a model pickle and return it.


    Args:
        fn: the full path to the model to load.
    """


    with open(fn, "rb") as full_path:
        sg_model = pickle.load(full_path)
    return sg_model




class SkipGram:
    def __init__(self, sentences, d=100, neg_samples=4, context=4, word_count_threshold=5):
        self.sentences = sentences
        self.d = d  # embedding dimension
        self.neg_samples = neg_samples  # num of negative samples for one positive sample
        self.context = context #the size of the context window (not counting the target word)
        self.word_count_threshold = word_count_threshold #ignore low frequency words (appearing under the threshold)


        # Tips:
        # 1. It is recommended to create a word:count dictionary
        # 2. It is recommended to create a word-index map


        # Initialize mappings and counters
        self.word2index = {}         # word → index
        self.index2word = {}         # index → word
        self.word_count = {}         # word → count
        self.word_count_total = 0    # total word count in corpus

        # embedding matrices
        self.T = []
        self.C = []

        raw_counts = Counter()
        for sentence in sentences:
            raw_counts.update(sentence)

        # Filter words by frequency threshold
        filtered_vocab = {
            word: count for word, count in raw_counts.items()
            if count >= self.word_count_threshold
        }

        # Create word-index mappings
        for idx, word in enumerate(filtered_vocab):
            self.word2index[word] = idx
            self.index2word[idx] = word

        self.word_count = filtered_vocab
        self.word_count_total = sum(filtered_vocab.values())

        # Build a unigram distribution for negative sampling (raise to 0.75)
        self.word_probs = np.array([count ** 0.75 for count in filtered_vocab.values()])
        self.word_probs /= self.word_probs.sum()
        self.vocab_size = len(self.word2index)

    def compute_similarity(self, w1, w2):
        """ Returns the cosine similarity (in [0,1]) between the specified words.


        Args:
            w1: a word
            w2: a word
        Retunrns: a float in [0,1]; defaults to 0.0 if one of specified words is OOV.
    """
        sim  = 0.0 # default
        
        if w1 not in self.word2index or w2 not in self.word2index:
            return 0.0  # One or both words are OOV (out of vocabulary)
        

        idx1 = self.word2index[w1]
        idx2 = self.word2index[w2]
        vec1 = self.V[idx1]
        vec2 = self.V[idx2]

        # Compute cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        # Normalize from [-1, 1] to [0, 1]
        sim = (cosine_sim + 1) / 2

        return sim # default


    def get_closest_words(self, w, n=5):
        """Returns a list containing the n words that are the closest to the specified word.


        Args:
            w: the word to find close words to.
            n: the number of words to return. Defaults to 5.
        """
        closest_words = []

        if w not in self.word2index:
            return []
        for word in self.word2index.keys():
            if word == w:
                continue
            sim = self.compute_similarity(w, word)
            closest_words.append((word, sim))
            
        # Sort by similarity and get the top n words
        closest_words.sort(key=lambda x: x[1], reverse=True)
        closest_words = closest_words[:n]
        return closest_words

    def _initialize_embeddings(self):
        """Initializes the embedding matrices T and C.

        Returns:
            tuple: A tuple (T, C), where:
                - T (np.ndarray): Target word embedding matrix of shape (d, vocab_size).
                - C (np.ndarray): Context word embedding matrix of shape (vocab_size, d).
        """
        T = np.random.uniform(-0.5, 0.5, (self.d, self.vocab_size))
        C = np.random.uniform(-0.5, 0.5, (self.vocab_size, self.d))
        return T, C


    def _generate_training_pairs(self):
        """Generates all (center_idx, context_idx) training pairs from the corpus.

        Iterates over all sentences and collects word index pairs within the specified context window.

        Returns:
            list of tuple: A list of (center_word_index, context_word_index) pairs.
        """
        training_pairs = []
        for sentence in self.sentences:
            word_indices = [self.word2index[word] for word in sentence if word in self.word2index]
            for center_pos in range(len(word_indices)):
                center_idx = word_indices[center_pos]
                for offset in range(-self.context, self.context + 1):
                    if offset == 0 or center_pos + offset < 0 or center_pos + offset >= len(word_indices):
                        continue
                    context_idx = word_indices[center_pos + offset]
                    training_pairs.append((center_idx, context_idx))
        return training_pairs


    def _sample_negative(self, k, exclude_idx):
        """Samples k negative word indices from the vocabulary, excluding the positive index.

        Args:
            k (int): Number of negative samples to generate.
            exclude_idx (int): The index of the true context word to exclude.

        Returns:
            list of int: List of k sampled word indices not equal to exclude_idx.
        """
        negatives = []
        while len(negatives) < k:
            idx = np.random.choice(np.arange(self.vocab_size), p=self.word_probs)
            if idx != exclude_idx:
                negatives.append(idx)
        return negatives


    def _train_one_epoch(self, T, C, training_pairs, step_size):
        """Performs a single epoch of training using SGNS loss and updates embeddings.

        Args:
            T (np.ndarray): Target embedding matrix of shape (d, vocab_size).
            C (np.ndarray): Context embedding matrix of shape (vocab_size, d).
            training_pairs (list of tuple): List of (center_idx, context_idx) training pairs.
            step_size (float): Learning rate for gradient descent.

        Returns:
            float: Total loss over all training pairs in this epoch.
        """
        total_loss = 0
        random.shuffle(training_pairs)

        for center_idx, context_idx in training_pairs:
            v_target = T[:, center_idx]
            v_context = C[context_idx]
            score = sigmoid(np.dot(v_target, v_context))
            grad = score - 1

            T[:, center_idx] -= step_size * grad * v_context
            C[context_idx] -= step_size * grad * v_target

            total_loss += -np.log(score + 1e-10)

            negative_indices = self._sample_negative(self.neg_samples, context_idx)
            for neg_idx in negative_indices:
                v_neg = C[neg_idx]
                score_neg = sigmoid(np.dot(v_target, v_neg))
                grad_neg = score_neg

                T[:, center_idx] -= step_size * grad_neg * v_neg
                C[neg_idx] -= step_size * grad_neg * v_target

                total_loss += -np.log(1.0 - score_neg + 1e-10)

        return total_loss


    def learn_embeddings(self, step_size=0.001, epochs=50, early_stopping=3, model_path=None):
        """Returns a trained embedding models and saves it in the specified path


        Args:
            step_size: step size for  the gradient descent. Defaults to 0.0001
            epochs: number or training epochs. Defaults to 50
            early_stopping: stop training if the Loss was not improved for this number of epochs
            model_path: full path (including file name) to save the model pickle at.
        """

        #tips:
        # 1. have a flag that allows printing to standard output so you can follow timing, loss change etc.
        # 2. print progress indicators every N (hundreds? thousands? an epoch?) samples
        # 3. save a temp model after every epoch
        # 4.1 before you start - have the training examples ready - both positive and negative samples
        # 4.2. it is recommended to train on word indices and not the strings themselves.


        T, C = self._initialize_embeddings()
        training_pairs = self._generate_training_pairs()

        best_loss = float('inf')
        no_improvement = 0

        for epoch in range(epochs):
            start = time.time()
            loss = self._train_one_epoch(T, C, training_pairs, step_size)

            if loss < best_loss:
                best_loss = loss
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= early_stopping:
                    break

            if model_path:
                with open(model_path.replace(".pkl", f"_epoch{epoch + 1}.pkl"), "wb") as f:
                    pickle.dump(self, f)


        # save the trained model
        self.T = T
        self.C = C

        if model_path:
            with open(model_path, "wb") as file:
                pickle.dump(self, file)

        return T,C


    def combine_vectors(self, T, C, combo=0, model_path=None):
        """Returns a single embedding matrix and saves it to the specified path


        Args:
            T: The learned targets (T) embeddings (as returned from learn_embeddings())
            C: The learned contexts (C) embeddings (as returned from learn_embeddings())
            combo: indicates how wo combine the T and C embeddings (int)
                   0: use only the T embeddings (default)
                   1: use only the C embeddings
                   2: return a pointwise average of C and T
                   3: return the sum of C and T
                   4: concat C and T vectors (effectively doubling the dimention of the embedding space)
            model_path: full path (including file name) to save the model pickle at.
        """


        if combo == 0:
            V = T.T
        elif combo == 1:
            V = C
        elif combo == 2:
            V = (T.T + C) / 2
        elif combo == 3:
            V = T.T + C
        elif combo == 4:
            V = np.concatenate((T.T, C), axis=1)
        else:
            raise ValueError("Invalid combo value. Must be 0, 1, 2, 3, or 4.")

        if model_path:
            with open(model_path, "wb") as file:
                pickle.dump(V, file)

        self.V = V
        return V


    def find_analogy(self, w1,w2,w3):
        """Returns a word (string) that matches the analogy test given the three specified words.
           Required analogy: w1 to w2 is like ____ to w3.


        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
        """


        if not all(w in self.word2index for w in [w1, w2, w3]):
            return None  # Handle OOV words gracefully

        idx1 = self.word2index[w1]
        idx2 = self.word2index[w2]
        idx3 = self.word2index[w3]

        vec1 = self.V[idx1]
        vec2 = self.V[idx2]
        vec3 = self.V[idx3]

        target_vec = vec2 - vec1 + vec3
        target_vec /= np.linalg.norm(target_vec)

        best_word = None
        best_similarity = -1
        for word, idx in self.word2index.items():
            if word in [w1, w2, w3]:
                continue
            vec = self.V[idx]
            sim = np.dot(target_vec, vec) / (np.linalg.norm(vec) + 1e-8)
            if sim > best_similarity:
                best_similarity = sim
                best_word = word

        return best_word


    def test_analogy(self, w1, w2, w3, w4, n=1):
        """Returns True if sim(w1-w2+w3, w4)@n; Otherwise return False.
            That is, returning True if w4 is one of the n closest words to the vector w1-w2+w3.
            Interpretation: 'w1 to w2 is like w4 to w3'


        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
             w4: forth word in the analogy (string)
             n: the distance (work rank) to be accepted as similarity
            """


        w_analogy = self.find_analogy(w1, w2, w3)

        if w_analogy == w4:
            return True
        else:
            # Get the closest words to the analogy vector
            closest_words = self.get_closest_words(w_analogy, n=n)
            if w4 in closest_words:
                return True


        return False