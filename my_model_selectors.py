import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection based on BIC scores
        best_score = float("inf")
        best_model = self.base_model(self.n_constant)
        for n_state in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n_state)
                score = model.score(self.X, self.lengths)
                pN = n_state * n_state + 2 * n_state * len(self.X[0]) - 1
                bic = (-2) * score + math.log(len(self.X)) * pN
                if bic < best_score:
                    best_score = bic
                    best_model = model
            except:
                continue
        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection based on DIC scores
        best_score = float("inf")
        best_model = self.base_model(self.n_constant)
        m = len(self.words.keys())
        for n_state in range(self.min_n_components, self.max_n_components + 1):
            total_score = 0.0
            try:
                model = self.base_model(n_state)
                score = model.score(self.X, self.lengths)
                for a_word in self.hwords:
                    if a_word == self.this_word:
                        continue
                    X_word, a_word_lengths = self.hwords[a_word]
                    total_score += model.score(X_word, a_word_lengths)
                total_score = np.mean(total_score)
                dic = score - (1 / m - 1) * total_score
                if dic < best_score:
                    best_score = dic
                    best_model = model
            except:
                continue
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection using CV
        best_score = float("inf")
        best_model = self.base_model(self.n_constant)
        k_fold = KFold(random_state=self.random_state, n_splits=min(3, len(self.sequences)))
        for n_state in range(self.min_n_components, self.max_n_components + 1):
            folds = 0
            total_score = 0
            if len(self.sequences) < k_fold.n_splits:
                break
            try:
                for cv_train_x, cv_test_x in k_fold.split(self.sequences):
                    folds += 1
                    X_test, lengths_test = combine_sequences(cv_test_x, self.sequences)
                    model = self.base_model(n_state)
                    score = model.score(X_test, lengths_test)
                    total_score += score
                    average_score = total_score / folds
            except:
                pass
            if best_score < average_score:
                best_score = average_score
                best_model = model
        return best_model
