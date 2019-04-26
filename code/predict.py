from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import os

from model import create_tensorflow_model

def pre_process_test(input_path):
    print("Processing Test file....")
    var = input_path
    os.system(r'python preprocess.py Test ' + var)

    u_test_batch = np.load("./Processed_Test/test_unis.npy")

    count = np.count_nonzero(u_test_batch, axis=1)
    count = np.insert(count, 0, 0)

    b_test_batch = np.load("./Processed_Test/test_bis.npy")

    return u_test_batch, b_test_batch, count


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()


def predict(input_path, output_path, resources_path):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the BIES format.

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.


    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    n = 80
    restore_list = []
    splited = open(r"./cut_input.txt", "w", encoding="utf8")
    with open(input_path, "r", encoding="utf8") as f:
        f_content = f.read()
        for l_idx, line in enumerate(f_content.splitlines()):
            restore_list.append((l_idx, int(np.ceil(len(line)/n)-1)))
            line = list(line)
            line.append("</s>")

        list_line = [line[i:i+n] for line in f_content.splitlines() for i in range(0, len(line), n)]
        for l in list_line:
            splited.write(l)
            splited.write("\n")
        splited.close()

    u_test_batch, b_test_batch, count = pre_process_test(r"./cut_input.txt")

    MAX_LENGTH = 80
    U_VOCAB_SIZE = 6592
    B_VOCAB_SIZE = 1042976

    EMBEDDING_SIZE = 32
    HIDDEN_SIZE = 100

    uni_inputs, bi_inputs, labels, keep_prob, loss_mask, mean_l, train_op, preds, eq, acc = create_tensorflow_model(U_VOCAB_SIZE, B_VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join(resources_path, "checkpts")))
        print("\nModel restored!")
        print("Model is Predicting....")
        pred = sess.run([preds], feed_dict={uni_inputs: u_test_batch, bi_inputs: b_test_batch, keep_prob: 1.0})
        print("Done !")
        # print(pred)
        # print("Shape of prediction is ", pred[0].shape)

        print("Reconstructing the Network's Prediction....")
        read = {0: "B", 1: "I", 2: "E", 3: "S"}
        c_op_file = open("./cut_output.txt", "w")
        reconst = []
        for c_idx in range(len(count)-1):
            # print("Len of this line is ", count[idx+1])
            reconst.append(pred[0][count[c_idx]:count[c_idx+1]+count[c_idx]])
        for r_idx, line in enumerate(reconst):
            # c_op_file.write(str(r_idx))  ## COMMENT THIS WHEN DONE WITH DEBUG
            for elem in line:
                # print("Elem is" , elem)
                c_op_file.write(read[elem])
            c_op_file.write("\n")
    c_op_file.close()

    print("\n")
    op_file = open(output_path, "w")
    with open("./cut_output.txt", "r", encoding="utf8") as f:
        op_content = f.read().splitlines()
        for i in range(len(restore_list)):
                # print("{:} << We will add {:} lines to this".format(op_content[i], restore_list[i][1]))
                if restore_list[i][1] == 0:
                    op_file.write(op_content[i])
                elif restore_list[i][1] > 0:
                    # print("it is ", i+restore_list[i][1]+1)
                    reconstructed = op_content[i:i+restore_list[i][1]+1]
                    # print("Line reconstructed is ", reconstructed)
                    for r in reconstructed:
                        op_file.write(r)
                    # print("I now want to delete", op_content[i+1:i+restore_list[i][1]+1])
                    del(op_content[i+1:i+restore_list[i][1]+1])
                op_file.write("\n")
                # print("\n")
    print("All Done !")

if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)
