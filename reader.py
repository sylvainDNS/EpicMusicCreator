import json
import collections
import os


def _read_words(filename):
    data = json.load(open(filename))

    parsedData = []
    for i in range(0, len(data["song"])):
        parsedData.append(
            (data["song"][i]["note"], data["song"][i]["duration"]))

    return parsedData


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (- x[1], x[0]))

    words, _ = list(zip(* count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_words_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def load_data(datapath):

    train_path = os.path.join(datapath, "TheHobbit.json")
    valid_path = os.path.join(datapath, "InDreams.json")
    test_path = os.path.join(datapath, "HesAPirate.json")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_words_ids(train_path, word_to_id)
    valid_data = _file_to_words_ids(valid_path, word_to_id)
    test_data = _file_to_words_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary
