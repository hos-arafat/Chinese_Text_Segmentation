import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def create_tensorflow_model(u_vocab_size, b_vocab_size, embedding_size, hidden_size):
    print("\nCreating Tensorflow model..\n")

    # Inputs have (batch_size, timesteps) shape.
    unigrams_in = tf.placeholder(tf.int32, shape=[None, None])
    # Inputs have (batch_size, timesteps) shape.
    bigrams_in = tf.placeholder(tf.int32, shape=[None, None])
    # Labels have (batch_size,) shape.
    labels = tf.placeholder(tf.int64, shape=[None, None])
    # Keep_prob is a scalar.
    keep_prob = tf.placeholder(tf.float32, shape=[])
    # Calculate sequence lengths to mask out the paddings later on.
    seq_length = tf.count_nonzero(unigrams_in, axis=-1)
    # Mask array is a scalar.
    loss_mask = tf.to_float(tf.not_equal(unigrams_in, 0))

    with tf.variable_scope("uni_embeddings"):
        u_embedding_matrix = tf.get_variable("embeddings", shape=[u_vocab_size, embedding_size])
        u_embeddings = tf.nn.embedding_lookup(u_embedding_matrix, unigrams_in)

    with tf.variable_scope("bi_embeddings"):
        b_embedding_matrix = tf.get_variable("embeddings", shape=[b_vocab_size, embedding_size])
        b_embeddings = tf.nn.embedding_lookup(b_embedding_matrix, bigrams_in)

    with tf.variable_scope("concat_embeddings"):
        embeddings = tf.concat([u_embeddings, b_embeddings], -1)

    with tf.variable_scope("rnn"):
        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
        lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell,
                                                 input_keep_prob=keep_prob,
                                                 output_keep_prob=keep_prob,
                                                 state_keep_prob=keep_prob)
        # Backward direction cell
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell,
                                                 input_keep_prob=keep_prob,
                                                 output_keep_prob=keep_prob,
                                                 state_keep_prob=keep_prob)

        # Get BiRNN cell output
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embeddings, sequence_length=seq_length, dtype=tf.float32)
        con = tf.concat(outputs, -1)
        # con = tf.concat(outputs, 2)

    with tf.variable_scope("dense"):
        # TensorFlow performs the sigmoid in the loss function for efficiency,
        # so don't use any activation on last dense.
        logits = tf.layers.dense(con, 4, activation=None)
        logits = tf.squeeze(logits)

    #loss
    with tf.variable_scope("loss"):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    # Masked loss
    with tf.variable_scope("masked_loss"):
        b_masked_loss = tf.boolean_mask(loss, loss_mask)
        mean_loss = tf.reduce_mean(b_masked_loss, name="final_loss")

    with tf.variable_scope("train"):
        # train_op = tf.train.AdamOptimizer(0.04).minimize(mean_loss)
        train_op = tf.train.MomentumOptimizer(0.04, 0.95).minimize(mean_loss)

    with tf.variable_scope("accuracy"):
        probs = tf.nn.softmax(logits)
        predictions = tf.math.argmax(probs, axis=-1, name="arg_maxing")
        m_out_predictions = tf.boolean_mask(predictions, loss_mask)
        m_out_labels = tf.boolean_mask(labels, loss_mask)
        # predictions = tf.math.multiply(predictions, tf.cast(loss_mask, tf.int64))

        eq = tf.cast(tf.equal(m_out_predictions, m_out_labels), tf.float32)
        acc = tf.reduce_mean(eq)

    return unigrams_in, bigrams_in, labels, keep_prob, loss_mask, mean_loss, train_op, m_out_predictions, eq, acc
