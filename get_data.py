import numpy as np
import torch
from torch import nn
from torch import autograd

def prepeare_X(array, batch_size, seq_len, n_vocab):
    tensor = torch.FloatTensor(batch_size, seq_len, n_vocab).zero_()
    if (batch_size == 1):
        for i, c in enumerate(array):
            tensor[0, i, c] = 1
    else:
        for j in xrange(batch_size):
            for i, c in enumerate(array[j]):
                tensor[j, i, c] = 1
    prepreared_data = autograd.Variable(tensor)
    return prepreared_data

def get_data(raw_text, seq_len):
    chars = sorted(list(set(raw_text))) #making vocab (all unique characters)
    char_to_int = dict((c, i) for i, c in enumerate (chars)) #coding vocab from 0 to size of vocab
    int_to_char = dict((i, c) for i, c in enumerate(chars)) #encoding from int fo char
    n_chars = len(raw_text)
    n_vocab = len(char_to_int)
    n_examples = n_chars - seq_len
    
    X = []
    Y = []
    points = np.arange(n_examples)
    np.random.shuffle(points)
    for i in points:
        seq_in = raw_text[i : i + seq_len]
        seq_out = raw_text[(i+1) : (i + seq_len + 1)]
        X.append([char_to_int[char] for char in (seq_in)])
        Y.append([char_to_int[char] for char in (seq_out)])

    return n_chars, n_vocab, n_examples, X, Y, int_to_char, char_to_int