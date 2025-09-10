import re, random, math
from collections import defaultdict, Counter
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')


class Spell_Checker:
    """Context-sensitive spell checker using a noisy channel model.

    Corrects spelling mistakes using a statistical language model and
    an error distribution matrix.
    """


    def __init__(self,  lm=None):
        """Initializing a spell checker object with a language model as an
        instance  variable.


        Args:
            lm: a language model object. Defaults to None.
        """
        self.lm = lm
        self.error_tables: dict[str, dict[str, float]] = {}


    def add_language_model(self, lm):
        """Adds the specified language model as an instance variable.
            (Replaces an older LM dictionary if set)


            Args:
                lm: a Spell_Checker.Language_Model object
        """
        self.lm = lm

    def add_error_tables(self, error_tables):
        """ Adds the specified dictionary of error tables as an instance variable.
            (Replaces an older value dictionary if set)


            Args:
            error_tables (dict): a dictionary of error tables in the format
            of the provided confusion matrices:
            https://www.dropbox.com/s/ic40soda29emt4a/spelling_confusion_matrices.py?dl=0
        """
        self.error_tables = error_tables

    def evaluate_text(self, text):
        """Returns the log-likelihood of the specified text given the language
            model in use. Smoothing should be applied on texts containing OOV words
    
           Args:
               text (str): Text to evaluate.
    
           Returns:
               Float. The float should reflect the (log) probability.
        """
        return self.lm.evaluate_text(text)
    
    def _known(self, words: set[str]) -> set[str]:
        """
        Filters the given set of words, returning only those that are present in the language model's vocabulary.

        Args:
            words (set[str]): A set of words to be checked against the language model's vocabulary.

        Returns:
            set[str]: A set containing only the words that exist in the language model's vocabulary.
        """
        return set(w for w in words if w in self.lm.vocabulary)

    def _edits1(self, word):
        """Generate all one-edit-distance words from the input.
    
        Supports deletion, insertion, substitution, and transposition.
        
        Args:
            word (str): The input word.
        
        Returns:
            set: Set of candidate words with edit operations.
        """
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        deletes = [(L + R[1:], ("deletion", R[0])) for L, R in splits if R]
        transposes = [(L + R[1] + R[0] + R[2:],("substitution", R[0] + R[1]))for L, R in splits if len(R) > 1]
        replaces = [(L + c + R[1:], ("substitution", R[0] + c)) for L, R in splits if R for c in letters]
        inserts = [(L + c + R, ("insertion", c + R[0] if R else c)) for L, R in splits for c in letters]

        return set(deletes + transposes + replaces + inserts)
    
    def _edits2(self, word):
        """
        Generates all candidate words that are two single-character edits away.

        This function builds on `_edits1` to generate words with two edits. Only the
        first edit's information is retained for use in the noisy-channel probability.

        Args:
            word (str): The word to generate second-level edit candidates for.

        Returns:
            set[tuple[str, tuple[str, str]]]: A set of tuples (candidate, edit_info), 
            where `edit_info` comes from the first edit step.
        """
        edits2 = set()
        for cand1, edit_info in self._edits1(word):
            for cand2, _ in self._edits1(cand1):
                edits2.add((cand2, edit_info))          # keep the first edit’s info
        return edits2
    
    def _generate_candidates(self, word):
        """Returns a list of possible candidates for the specified word.
            The candidates are generated according to the error tables.

            Args:
                word (str): the word to generate candidates for.

            Returns:
                List of candidate words.
        """
        candidates = [(word, None)]                     
        if word not in self.lm.vocabulary:              
            edits1 = self._edits1(word)
            edits2 = self._edits2(word)                
            pool = edits1 | edits2                  
            candidates += [c for c in pool if c[0] in self.lm.vocabulary]
        return candidates


    def _error_logprob(self, candidate_tuple, original_word, alpha):
        """
        Estimates the log-probability of an observed error using a noisy channel model.

        Applies a heuristic that supports a single-character edit. If the word was
        not changed (i.e., candidate is the same as the original), it returns log(alpha).
        Otherwise, it looks up the error probability from the confusion matrix and
        combines it with (1 - alpha).

        Args:
            candidate_tuple (tuple[str, tuple[str, str] or None]): A tuple containing the
                candidate word and its edit info (type and characters involved).
            original_word (str): The original, possibly misspelled word.
            alpha (float): The prior probability of a word being correct.

        Returns:
            float: The log-probability of the edit (or non-edit) under the error model.
        """
        cand, edit_info = candidate_tuple
        if edit_info is None:              
            return math.log(alpha)
    
        edit_type, chars = edit_info
        prob = self.error_tables.get(edit_type, {}).get(chars)
        if prob is None:                         
            for tbl in self.error_tables.values():
                if chars in tbl:
                    prob = tbl[chars]
                    break
        prob = prob if prob is not None else 0.2
        return math.log((1 - alpha) * prob + 1e-12)


    def spell_check(self, text, alpha):
        """ Returns the most probable fix for the specified text. Use a simple
            noisy channel model if the number of tokens in the specified text is
            smaller than the length (n) of the language model.


            Args:
                text (str): the text to spell check.
                alpha (float): the probability of keeping a lexical word as is.


            Return:
                A modified string (or a copy of the original if no corrections are made.)
        """

        tokens = normalize_text(text).split()
        use_context = len(tokens) >= self.lm.n
        best_sentence = tokens
        best_score = self.lm.evaluate_text(" ".join(tokens)) if use_context else 0.0

        for idx, wrong in enumerate(tokens):
            if not wrong.isalpha():          # fast‑forward over numbers, punctuation, etc.
                continue
            for cand_tuple in self._generate_candidates(wrong):
                cand, _ = cand_tuple
                if cand == wrong:
                    err = math.log(alpha)
                else:
                    err = self._error_logprob(cand_tuple, wrong, alpha)

                new_tokens = tokens[:]
                new_tokens[idx] = cand
                lm_score = self.lm.evaluate_text(" ".join(new_tokens)) if use_context else 0.0
                score    = lm_score + err

                if score > best_score:
                    best_score, best_sentence = score, new_tokens

        return " ".join(best_sentence)




    #####################################################################
    #                   Inner class                                     #
    #####################################################################


    class Language_Model:
        """The class implements a Markov Language Model that learns a model from a given text.
            It supports language generation and the evaluation of a given string.
            The class can be applied on both word level and character level.
        """


        def __init__(self, n=3, chars=False):
            """Initializing a language model object.
            Args:
                n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
                chars (bool): True iff the model consists of ngrams of characters rather than word tokens.
                              Defaults to False
            """
            self.n = n
            self.model_dict: dict[tuple, Counter] = defaultdict(Counter) #a dictionary of the form {ngram:count}, holding counts of all ngrams in the specified text.
            self.chars = chars
            self.context_totals: Counter = Counter()
            self.vocabulary: set[str] = set()

        def build_model(self, text):  # should be called build_model
            """populates the instance variable model_dict.


                Args:
                    text (str): the text to construct the model from.
            """
            normalized_text = normalize_text(text)  # Normalize the text
            tokens = list(normalized_text.replace(' ', '')) if self.chars else normalized_text.split()  # Tokenize text

            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i : i + self.n - 1])
                target  = tokens[i + self.n - 1]
                self.model_dict[context][target] += 1
                self.context_totals[context] += 1

            self.vocabulary.update(tokens)


        def get_model_dictionary(self):
            """Returns the dictionary class object
            """
            return self.model_dict
        


        def get_model_window_size(self):
            """Returning the size of the context window (the n in "n-gram")
            """
            return self.n
        
        def _prob(self, context: tuple, token: str) -> float:
            """
            Laplace smoothed probability P(token | context)
            """
            V = len(self.vocabulary) or 1
            num = self.model_dict[context][token] + 1
            den = self.context_totals[context] + V
            return num / den

        def generate(self, context=None, n=20):
            """Returns a string of the specified length, generated by applying the language model
            to the specified seed context. If no context is specified the context should be sampled
            from the models' contexts distribution. Generation should stop before the n'th word if the
            contexts are exhausted. If the length of the specified context exceeds (or equal to)
            the specified n, the method should return a prefix of length n of the specified context.

            Args:
                context (str): a seed context to start the generated string from. Defaults to None
                n (int): the length of the string to be generated.

            Return:
                String. The generated text.
            """
            if context:
                prefix = normalize_text(context)
                out    = list(prefix.replace(" ", "")) if self.chars else prefix.split()
            else:
                # sample a random context
                context = random.choice(list(self.model_dict.keys()))
                out = list(context)

            while len(out) < n:
                ctx = tuple(out[-(self.n - 1):]) if len(out) >= self.n - 1 else tuple(out)
                choices, weights = zip(*self.model_dict[ctx].items()) if ctx in self.model_dict else ((), ())
                if not choices:
                    break
                nxt = random.choices(choices, weights)[0]
                out.append(nxt)

            return "".join(out) if self.chars else " ".join(out[:n])


        def evaluate_text(self, text):
            """Returns the log-likelihood of the specified text to be a product of the model.
               Laplace smoothing should be applied if necessary.


               Args:
                   text (str): Text to evaluate.


               Returns:
                   Float. The float should reflect the (log) probability.
            """
            clean  = normalize_text(text)
            tokens = list(clean.replace(" ", "")) if self.chars else clean.split()

            if len(tokens) < self.n:
                return float("-inf")

            score = 0.0
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i : i + self.n - 1])
                prob = tokens[i + self.n - 1]
                score += math.log(self._prob(ngram, prob))
                
            return score / (len(tokens) - self.n + 1)


        def smooth(self, ngram):
            """Returns the smoothed (Laplace) probability of the specified ngram.


                Args:
                    ngram (str): the ngram to have its probability smoothed


                Returns:
                    float. The smoothed probability.
            """
            context = ngram[:-1]
            vocab_size = len(self.vocabulary)
            count_ngram = self.model_dict.get(ngram, 0)
            count_context = self.context_totals.get(context, 0)

            return (count_ngram + 1) / (count_context + vocab_size)


def normalize_text(text):
    """Returns a normalized version of the specified string.
      You can add default parameters as you like (they should have default values!)
      You should explain your decisions in the header of the function.


      Args:
        text (str): the text to normalize


      Returns:
        string. the normalized text.
    """
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [tok for tok in tokens if re.search(r"[a-z0-9]", tok)]
    return " ".join(tokens)

def who_am_i():  # this is not a class method
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Ayelet Hashahar Cohen', 'id': '206533895', 'email': 'ayelethc@post.bgu.ac.il'}



