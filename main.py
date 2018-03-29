import reader as rdr
import generator as gnt
import Input as ipt
import Model as mdl
import tensorflow as tf
import numpy as np

MODEL_PATH = 'model'
INPUT_PATH = 'input'


def main():
    raw_data = rdr.load_data("input/")

    train_data, valid_data, test_data, vocabulary, reversed_dictionary = raw_data

    test(MODEL_PATH, test_data, reversed_dictionary, vocabulary)


def test(model_path, test_data, reversed_dictionary, vocabulary):
    test_input = ipt.Input(batch_size=20, num_steps=35, data=test_data)
    m = mdl.Model(test_input, is_training=False, hidden_size=650, vocab_size=vocabulary,
                  num_layers=2)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # start threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        current_state = np.zeros((2, 2, m.batch_size, m.hidden_size))
        # restore the trained model
        saver.restore(sess, model_path)
        # get an average accuracy over num_acc_batches
        num_acc_batches = 30
        check_batch_idx = 25
        acc_check_thresh = 5
        accuracy = 0
        for batch in range(num_acc_batches):
            if batch == check_batch_idx:
                true_vals, pred, current_state, acc = sess.run(
                    [m.input_obj.targets, m.predict, m.state, m.accuracy], feed_dict={m.init_state: current_state})
                pred_string = [reversed_dictionary[x]
                               for x in pred[:m.num_steps]]
                true_vals_string = [reversed_dictionary[x]
                                    for x in true_vals[0]]
                print("True values (1st line) vs predicted values (2nd line):")
                print(" ".join(true_vals_string))
                print(" ".join(pred_string))
            else:
                acc, current_state = sess.run([m.accuracy, m.state], feed_dict={
                                              m.init_state: current_state})
            if batch >= acc_check_thresh:
                accuracy += acc
        print("Average accuracy: {:.3f}".format(
            accuracy / (num_acc_batches-acc_check_thresh)))
        # close threads
        coord.request_stop()
        coord.join(threads)


main()
