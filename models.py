#!/usr/bin/env python3

# Problem 1: Bigram HMMs

import json
import math
import time
import sys
import os

UNK_PROP = 0.1
START    = 'START\0'
START2   = 'START2\0'
UNK      = 'UNK\0'
STOP     = 'STOP\0'
STOP2    = 'STOP2\0'
MODE     = "default"
if os.getenv("MODE") is not None:
    MODE = os.getenv("MODE")

def load_data(group):
    return [json.loads(line) for line in open("data/twt." + group + ".json", "r").readlines()]

def data_to_tags(data):
    def line_to_tags(line):
        return [tag for word, tag in line ]
    return [ line_to_tags(line) for line in data ]

def data_to_words(data):
    def line_to_words(line):
        return [word for word, tag in line ]
    return [ line_to_words(line) for line in data ]

from collections import Counter

def unigram_counter(data):
    return Counter([word for line in data for word in line])

def bigram_counter(data):
    return Counter([ (line[i], line[i+1]) for line in data for i in range(len(line) - 1) ])

def trigram_counter(data):
    return Counter([ (line[i], line[i+1], line[i+2]) for line in data for i in range(len(line) - 2) ])

def get_accuracy(predicted_tags, true_tags):
    assert len(predicted_tags) == len(true_tags), "%s != %s" % (len(predicted_tags), len(true_tags))
    return sum([p == t for p, t in zip(predicted_tags, true_tags)]) / float(len(true_tags))

def print_data_stats(data, name):
    sample_size = len(data)
    data_tags = data_to_tags(data)
    data_words = data_to_words(data)
    tags_vocab_size = len(unigram_counter(data_tags).keys())
    words_vocab_size = len(unigram_counter(data_words).keys())
    avg_sample_length = sum(map(len, data)) / float(sample_size)
    print(name + " dataset stats:")
    print("- how many sentences: %s" % sample_size)
    print("- tags vocabulary size: %s" % tags_vocab_size)
    print("- words vocabulary size: %s" % words_vocab_size)
    print("- average sentence length: %.2f" % avg_sample_length)

print("Loading data...")
dev_data = load_data("dev")
print_data_stats(dev_data, "dev")
train_data = load_data("train")
print_data_stats(train_data, "train")
test_data = load_data("test")
print_data_stats(test_data, "test")

class Model():
    def __init__(self, data, ADD_K_CONST, DISCOUNT):
        data_bigram = [[(START, START)] + line + [(STOP, STOP)] for line in data ]
        data_tags = data_to_tags(data_bigram)
        uni_count_tags = unigram_counter(data_tags)
        tags_vocab = uni_count_tags.keys()
        bi_count_tags = bigram_counter(data_tags)

        # Define q(y_i | y_{i-1}), applying OOV and smoothing
        bi_cond_tags = {}
        for tag_prev in set(tags_vocab).union([UNK]):
            if tag_prev == STOP: continue
            bi_cond_tags[tag_prev] = {}
            tag_prev_count = 0
            for tag in tags_vocab:
                if tag == START: continue
                if tag_prev == UNK:
                    if tag in uni_count_tags:
                        bi_cond_tags[tag_prev][tag] = uni_count_tags[tag]
                    else:
                        bi_cond_tags[tag_prev][tag] = 0
                else:
                    if (tag_prev, tag) in bi_count_tags:
                        bi_cond_tags[tag_prev][tag] = bi_count_tags[(tag_prev, tag)]
                    else:
                        bi_cond_tags[tag_prev][tag] = 0
                tag_prev_count += bi_cond_tags[tag_prev][tag]
            for tag in bi_cond_tags[tag_prev]:
                bi_cond_tags[tag_prev][tag] = (bi_cond_tags[tag_prev][tag] + ADD_K_CONST * DISCOUNT) / float(tag_prev_count + ADD_K_CONST * len(tags_vocab))
            bi_cond_tags[tag_prev][UNK] = 1.0 - sum(bi_cond_tags[tag_prev].values())

        # validate
        for tag_prev in bi_cond_tags:
            assert abs(1.0 - sum(bi_cond_tags[tag_prev].values())) < 0.001

        data_words = data_to_words(data_bigram)
        uni_count_words = unigram_counter(data_words)
        words_vocab = uni_count_words.keys()
        emission_count = Counter([(word, tag) for line in data_bigram for word, tag in line])

        # Define e(x_i|y_i)
        emission_cond = {}
        for tag in set(tags_vocab).union([UNK]):
            emission_cond[tag] = {}
            tag_count = 0
            for word in words_vocab:
                if tag == UNK:
                    if word in uni_count_words:
                        emission_cond[tag][word] = uni_count_words[word]
                    else:
                        emission_cond[tag][word] = 0
                else:
                    if (word, tag) in emission_count:
                        emission_cond[tag][word] = emission_count[(word, tag)]
                    else:
                        emission_cond[tag][word] = 0
                tag_count += emission_cond[tag][word]
            for word in emission_cond[tag]:
                emission_cond[tag][word] = (emission_cond[tag][word] + ADD_K_CONST * DISCOUNT) / float(tag_count + ADD_K_CONST * len(words_vocab))
            emission_cond[tag][UNK] = 1.0 - sum(emission_cond[tag].values())

        # validate
        for tag in emission_cond:
            assert abs(1.0 - sum(emission_cond[tag].values())) == 0

        self.emission_cond = emission_cond
        self.bi_cond_tags = bi_cond_tags
        self.tags_vocab = tags_vocab

    # viterbi algorithm
    def inference(self, words):
        words = [START] + list(words) + [STOP]
        # \pi_{i, y_i}
        pi = {}
        pi[0] = { tag : 0.0 for tag in self.tags_vocab }
        pi[0][START] = 1.0
        bp = {}

        def f(tag_prev, tag, k):
            assert tag_prev in pi[k - 1], "tag_prev: %s, k-1: %s" % (tag_prev, k - 1)
            ret = pi[k - 1][tag_prev]
            assert ret >= 0, "pi[%s][%s]" % (k - 1, tag_prev)
            ret *= self.bi_cond_tags[tag_prev][tag]
            assert ret >= 0
            word = words[k]
            if tag in self.emission_cond:
                if word in self.emission_cond[tag]:
                    ret *= self.emission_cond[tag][word]
                else:
                    ret *= self.emission_cond[tag][UNK]
            else:
                if word in self.emission_cond[UNK]:
                    ret *= self.emission_cond[UNK][word]
                else:
                    ret *= self.emission_cond[UNK][UNK]
            return ret

        for k in range(1, len(words)):
            assert k not in pi
            pi[k] = { tag : 0.0 for tag in self.tags_vocab }
            assert k not in bp
            bp[k] = {}
            for tag in self.tags_vocab:
                if tag == START: continue
                bp_val, pi_val = max([(tag_prev, f(tag_prev, tag, k)) for tag_prev in self.tags_vocab if tag_prev != STOP], key=lambda x:x[1])
                pi[k][tag] = pi_val
                bp[k][tag] = bp_val

        y = [None] * len(words)
        y[len(words) - 1] = STOP
        for k in reversed(range(0, len(words) - 1)):
            y[k] = bp[k+1][y[k+1]]
        assert y[0] == START
        return y[1:-1]

class Model3():
    def __init__(self, data, ADD_K_CONST, DISCOUNT):
        data_bigram = [[(START, START), (START2, START2)] + line + [(STOP2, STOP2), (STOP, STOP)] for line in data ]
        data_tags = data_to_tags(data_bigram)
        uni_count_tags = unigram_counter(data_tags)
        tags_vocab = uni_count_tags.keys()
        tri_count_tags = trigram_counter(data_tags)

        # Define q(y_i | y_{i-1}, y_{i-2}), applying OOV and smoothing
        tri_cond_tags = {}
        for tag_pp in set(tags_vocab).union([UNK]): # tag_prev: y_{i-2}
            for tag_p in set(tags_vocab).union([UNK]): # tag_prev2: y_{i-1}
                if tag_pp == STOP or tag_p == STOP: continue
                if tag_pp == STOP2: continue
                base_key = (tag_pp, tag_p)
                tri_cond_tags[base_key] = {}
                tag_prev_count = 0
                for tag in tags_vocab:
                    if tag == START or tag == START2: continue
                    if tag_pp == UNK or tag_p == UNK:
                        if tag in uni_count_tags:
                            tri_cond_tags[base_key][tag] = uni_count_tags[tag]
                        else:
                            tri_cond_tags[base_key][tag] = 0
                    else:
                        if (tag_pp, tag_p, tag) in tri_count_tags:
                            tri_cond_tags[base_key][tag] = tri_count_tags[(tag_pp, tag_p, tag)]
                        else:
                            tri_cond_tags[base_key][tag] = 0
                    tag_prev_count += tri_cond_tags[base_key][tag]
                for tag in tri_cond_tags[base_key]:
                    tri_cond_tags[base_key][tag] = (tri_cond_tags[base_key][tag] + ADD_K_CONST * DISCOUNT) / float(tag_prev_count + ADD_K_CONST * len(tags_vocab))
                tri_cond_tags[base_key][UNK] = 1.0 - sum(tri_cond_tags[base_key].values())

        # validate
        for base_key in tri_cond_tags:
            assert abs(1.0 - sum(tri_cond_tags[base_key].values())) < 0.001

        data_words = data_to_words(data_bigram)
        uni_count_words = unigram_counter(data_words)
        words_vocab = uni_count_words.keys()
        emission_count = Counter([(word, tag) for line in data_bigram for word, tag in line])

        # Define e(x_i|y_i)
        emission_cond = {}
        for tag in set(tags_vocab).union([UNK]):
            emission_cond[tag] = {}
            tag_count = 0
            for word in words_vocab:
                if tag == UNK:
                    if word in uni_count_words:
                        emission_cond[tag][word] = uni_count_words[word]
                    else:
                        emission_cond[tag][word] = 0
                else:
                    if (word, tag) in emission_count:
                        emission_cond[tag][word] = emission_count[(word, tag)]
                    else:
                        emission_cond[tag][word] = 0
                tag_count += emission_cond[tag][word]
            for word in emission_cond[tag]:
                emission_cond[tag][word] = (emission_cond[tag][word] + ADD_K_CONST * DISCOUNT) / float(tag_count + ADD_K_CONST * len(words_vocab))
            emission_cond[tag][UNK] = 1.0 - sum(emission_cond[tag].values())

        # validate
        for tag in emission_cond:
            assert abs(1.0 - sum(emission_cond[tag].values())) == 0

        self.emission_cond = emission_cond
        self.tri_cond_tags = tri_cond_tags
        self.tags_vocab = tags_vocab

    # viterbi algorithm
    def inference(self, words):
        words = [START, START2] + list(words) + [STOP2, STOP]
        # pi[k][(tag_p, tag)]:
        # the highest probability that any sequence x_1 ... x_k ends with bigram (x_{k-1}: tag_p, x_k: tag)
        pi = {}
        pi[0] = { (tag, tag2) : 0.0 for tag in self.tags_vocab for tag2 in self.tags_vocab }
        pi[0][(START, START2)] = 1.0
        bp = {}

        # pi(k-1, tag_pp, tag_p) * q(tag | tag_pp, tag_p) * e(x_{k-1} | tag)
        def f(tag_pp, tag_p, tag, k):
            assert (tag_pp, tag_p) in pi[k - 1], "(tag_pp, tag_p): %s, k-1: %s" % ((tag_pp, tag_p), k - 1)
            ret = pi[k - 1][(tag_pp, tag_p)]
            assert ret >= 0, "pi[%s][%s]" % (k - 1, (tag_pp, tag_p))
            ret *= self.tri_cond_tags[(tag_pp, tag_p)][tag]
            assert ret >= 0
            word = words[k + 1]
            if tag in self.emission_cond:
                if word in self.emission_cond[tag]:
                    ret *= self.emission_cond[tag][word]
                else:
                    ret *= self.emission_cond[tag][UNK]
            else:
                if word in self.emission_cond[UNK]:
                    ret *= self.emission_cond[UNK][word]
                else:
                    ret *= self.emission_cond[UNK][UNK]
            return ret

        for k in range(1, len(words) - 1):
            assert k not in pi
            pi[k] = { (tag_pp, tag_p) : 0.0 for tag_pp in self.tags_vocab for tag_p in self.tags_vocab }
            assert k not in bp
            bp[k] = {}
            for tag_p in self.tags_vocab:
                if tag_p == START or tag_p == STOP: continue
                for tag in self.tags_vocab:
                    if tag == START or tag == START2: continue
                    bp_val, pi_val = max([(tag_pp, f(tag_pp, tag_p, tag, k)) for tag_pp in self.tags_vocab if tag_pp != STOP and tag_pp != STOP2], key=lambda x:x[1])
                    pi[k][(tag_p, tag)] = pi_val
                    bp[k][(tag_p, tag)] = bp_val

        y = [None] * len(words)
        y[len(words) - 2] = STOP2
        y[len(words) - 1] = STOP
        for k in reversed(range(0, len(words) - 2)):
            y[k] = bp[k+1][(y[k+1], y[k+2])]
        assert y[1] == START2, y
        assert y[0] == START, y
        return y[2:-2]
        
def run(model, data):
    now = time.time()
    print("Inference...")
    accuracies = []
    for line in data:
        words = [ word for word, tag in line ]
        tags  = [ tag  for word, tag in line ]
        predicted_tags = model.inference(words)
        acc   = get_accuracy(predicted_tags, tags)
        max_word_length = max(map(len, words))
        if MODE == "ERROR_ANALYSIS" and acc == 0.0:
            print(">>>>>>>>>>>>>>> Error case <<<<<<<<<<<<<")
            for p, pt in zip(line, predicted_tags):
                nspaces = max_word_length - len(p[0]) + 4
                spaces = ' ' * nspaces
                print("word: %s\ttag:%s\tpredicted:%s" % (p[0] + spaces, p[1], pt))
        accuracies.append(acc)
    accuracy = sum(accuracies) / len(accuracies)
    print("Average accuracy: %.4f" % accuracy)
    print("Inference time: %.4fs" % (time.time() - now))
    return accuracy

def experiment():
    ADD_K_CONST_val = float(sys.argv[1])
    DISCOUNT_val    = float(sys.argv[2])
    test_dataset    = sys.argv[3]
    print("Hyper-parameters:")
    print("- DISCOUNT=%f" % DISCOUNT_val)
    print("- ADD_K_CONST=%f" % ADD_K_CONST_val)
    tri = Model3(train_data, ADD_K_CONST_val, DISCOUNT_val)
    bi = Model(train_data, ADD_K_CONST_val, DISCOUNT_val)
    if test_dataset == "test":
        data = test_data
    elif test_dataset == "dev":
        data = dev_data
    tacc = run(tri, data)
    bacc = run(bi, data)
    if MODE != "ERROR_ANALYSIS":
        print("%s\t&%s\t&%.4f\t&%.4f\\\\" % (ADD_K_CONST_val, DISCOUNT_val, bacc, tacc))

if __name__ == "__main__":
    experiment()