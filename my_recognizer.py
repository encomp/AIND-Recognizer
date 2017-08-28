import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # Implement the recognizer
    # return probabilities, guesses
    for test in range(test_set.num_items):
        top_prob, top_word = float("-inf"), None
        probs = {}
        sequence, lengths = test_set.get_item_Xlengths(test)
        for word, model in models.items():
            try:
                probs[word] = model.score(sequence, lengths)
            except:
                probs[word] = float("-inf")
            if probs[word] > top_prob:
                top_prob, top_word = probs[word], word
        probabilities.append(probs)
        guesses.append(top_word)
    return probabilities, guesses
