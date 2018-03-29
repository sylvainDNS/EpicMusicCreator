import tensorflow as tf
import Model as mdl
import Input as ipt
import numpy as np

OUTPUT_PATH = 'output'


def batch_producer(raw_data, batch_size, num_steps):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(
        raw_data[0:batch_size * batch_len], [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = data[:, i * num_steps: (i + 1) * num_steps]
    x.set_shape([batch_size, num_steps])
    y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
    y.set_shape([batch_size, num_steps])

    return x, y


def train(train_data, vocabulary, num_layers, num_epochs, batch_size, model_save_name, learning_rate=1.0, max_lr_epoch=10, lr_decay=0.93):
    # setup data and models
    training_input = ipt.Input(batch_size=batch_size,
                               num_steps=35, data=train_data)
    m = mdl.Model(training_input, is_training=True, hidden_size=650, vocab_size=vocabulary,
                  num_layers=num_layers)
    init_op = tf.global_variables_initializer()

    orig_decay = lr_decay
    with tf.Session() as sess:
        # start threads
        sess.run([init_op])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver = tf.train.Saver()

        for epoch in range(num_epochs):
            new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0.0)
            m.assign_lr(sess, learning_rate * new_lr_decay)
            current_state = np.zeros(
                (num_layers, 2, batch_size, m.hidden_size))
            for step in range(training_input.epoch_size):
                if step % 50 != 0:
                    cost, _, current_state = sess.run([m.cost, m.train_op, m.state], feed_dict={
                                                      m.init_state: current_state})
                else:
                    cost, _, current_state, acc = sess.run(
                        [m.cost, m.train_op, m.state, m.accuracy], feed_dict={m.init_state: current_state})
                    print("Epoch {}, Step {}, cost: {:.3f}, accuracy: {:.3f}".format(
                        epoch, step, cost, acc))
            # save a model checkpoint
            saver.save(sess, OUTPUT_PATH + '\\' +
                       model_save_name, global_step=epoch)
        # do a final save
        saver.save(sess, OUTPUT_PATH + '\\' + model_save_name + '-final')
        # close threads
        coord.request_stop()
        coord.join(threads)
