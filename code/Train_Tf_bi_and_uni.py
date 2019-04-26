import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from preprocess_Class import Pocessor
from model import create_tensorflow_model

p = Pocessor("")

print("Loading Datasets...")
uni_train_x = np.load("./Processed_Train/train_unis.npy")
uni_dev_x = np.load("./Processed_Dev/dev_unis.npy")

bi_train_x = np.load("./Processed_Train/train_bis.npy")
bi_dev_x = np.load("./Processed_Dev/dev_bis.npy")

pad_train_y = np.load("./Processed_Train/train_labels.npy")
pad_dev_y = np.load("./Processed_Dev/dev_labels.npy")

print("Done!")

assert uni_train_x.shape == bi_train_x.shape == pad_train_y.shape, "Train Shapes are not equal"
assert uni_dev_x.shape == bi_dev_x.shape == pad_dev_y.shape, "Dev Shapes are not equal"

print("Training set {:} = Training set Bigrams {:} = Labels {:}: ".format(uni_train_x.shape, pad_train_y.shape, pad_train_y.shape))
print("Dev set Unigrams {:} = Dev set Bigrams {:} = Labels {:}: ".format(uni_dev_x.shape, bi_dev_x.shape, pad_dev_y.shape))

# DEFINE SOME COSTANTS
MAX_LENGTH = 80

U_VOCAB_SIZE = len(p.load_unigram_dict())
B_VOCAB_SIZE = len(p.load_bigram_dict())
# U_VOCAB_SIZE = 6592
# B_VOCAB_SIZE = 1042976

print("\nUnigram Vocab size is ", U_VOCAB_SIZE)
print("Bigram Vocab size is ", B_VOCAB_SIZE)
EMBEDDING_SIZE = 32
HIDDEN_SIZE = 100

def batch_generator(uni_X, bi_X, Y, batch_size, shuffle=False):
    if not shuffle:
        for start in range(0, len(uni_X), batch_size):
            end = start + batch_size
            uni_batch = uni_X[start:end]
            bi_batch = bi_X[start:end]
            yield uni_batch, bi_batch, Y[start:end]
    else:
        perm = np.random.permutation(len(uni_X))
        for start in range(0, len(uni_X), batch_size):
            end = start + batch_size
            uni_batch = uni_X[perm[start:end]]
            bi_batch = bi_X[perm[start:end]]
            yield uni_batch, bi_batch, Y[perm[start:end]]

def add_summary(writer, name, value, global_step):
     summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
     writer.add_summary(summary, global_step=global_step)


epochs = 5
batch_size = 300
print("\nLength of my TRAINING set is {:} & Batch size is {:}: ".format(len(uni_train_x), batch_size))
n_iterations = int(np.ceil(len(uni_train_x)/batch_size))
print("Therefore the Number of Training iterations is", n_iterations)

print("\nLength of my DEV set is {:} & Batch size is {:}: ".format(len(uni_dev_x), batch_size))
n_dev_iterations = int(np.ceil(len(uni_dev_x)/batch_size))
print("Therefore the Number of Development iterations is", n_dev_iterations)


uni_inputs, bi_inputs, labels, keep_prob, loss_mask, mean_l, train_op, preds, eq, acc = create_tensorflow_model(U_VOCAB_SIZE, B_VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE)
saver = tf.train.Saver()
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('logging/tensorflow_model', sess.graph)
    print("\nStarting training...")
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        epoch_loss, epoch_acc = 0., 0.
        mb = 0.0

        train_gen = batch_generator(uni_train_x, bi_train_x, pad_train_y, batch_size, shuffle=True)
        tq_gen = tqdm(train_gen, total=n_iterations, desc='Epoch %2d/%2d' % (epoch+1, epochs), ncols=80, smoothing=0.)
        for u_batch_x, b_batch_x, batch_y in tq_gen:
            mb += 1.0
            lm, _, acc_val, pred, equ, me_l = sess.run([loss_mask, train_op, acc, preds, eq, mean_l], #removed "acc" & "acc_val" from here
                                            feed_dict={uni_inputs: u_batch_x, bi_inputs: b_batch_x, labels: batch_y, keep_prob: 0.8})
            # Accumulate loss and acc as we scan through the dataset
            epoch_loss += me_l
            epoch_acc += acc_val
            #print("{:.2f}% of Epoch {:}\tTrain Loss: {:.4f}\tTrain Accuracy: {:.4f} ".format(100.*mb/n_iterations, epoch, epoch_loss/mb, epoch_acc/mb), end="\r")
            tqdm.set_postfix(tq_gen, Loss=epoch_loss/mb, Acc=epoch_acc/mb, refresh=True)


        # Once the Epoch is done, calculate Epoch Loss and Acc
        epoch_loss /= n_iterations
        epoch_acc /= n_iterations
        print("Epoch Train Loss: {:}\tTrain Accuracy: {:}".format(epoch_loss, epoch_acc))
        s_path = "./checkpts/Epoch_" + str(epoch+1)
        save_path = saver.save(sess, s_path)
        print("Model saved in path: %s " % save_path)
        add_summary(train_writer, "epoch_loss", epoch_loss, epoch)
        add_summary(train_writer, "epoch_acc", epoch_acc, epoch)

        # DEV EVALUATION
        dev_loss, dev_acc = 0.0, 0.0
        for u_batch_x, b_batch_x, batch_y in batch_generator(uni_dev_x, bi_dev_x, pad_dev_y, batch_size):
             loss_val, acc_val = sess.run([mean_l, acc], feed_dict={uni_inputs: u_batch_x, bi_inputs: b_batch_x, labels: batch_y, keep_prob: 1.0})
             dev_loss += loss_val
             dev_acc += acc_val
        dev_loss /= n_dev_iterations
        dev_acc /= n_dev_iterations

        add_summary(train_writer, "epoch_val_loss", dev_loss, epoch)
        add_summary(train_writer, "epoch_val_acc", dev_acc, epoch)
        print("\nTrain Loss: {:}\tTrain Accuracy: {:}".format(epoch_loss, epoch_acc))
        print("Dev   Loss: {:}\tDev   Accuracy: {:}\n".format(dev_loss, dev_acc))
        train_writer.close()
