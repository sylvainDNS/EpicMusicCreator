import reader
import generator


def main(_):
    raw_data = reader.load_data("data/")
    train_data, valid_data, test_data, vocabulary, reversed_dictionary = raw_data
