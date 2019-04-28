import sys
import os
import numpy as np
import pickle
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("mode", choices=['Train','Dev','Test'], help="Train, Dev, or Test")
    parser.add_argument("parent_path", help="The path of the icwb2 Data folder")

    return parser.parse_args()

class Pocessor:
    def __init__(self, m):

        self.mode = m

        self.uni_dict_pth = os.path.join("./Processed_Train", "unigrams_dictionary.pickle")
        self.bi_dict_pth  =  os.path.join("./Processed_Train", "bigrams_dictionary.pickle")

        if m == (""):
            pass
        else:
            print("Creating Necessary Files & Folders")
            folder = "./Processed_" + self.mode
            if not os.path.exists(folder):
                os.makedirs(folder)

            self.no_sp_pth = os.path.join(folder, (self.mode + "_no_spaces.utf8"))

            # self.uni_dict_pth = os.path.join("./Processed_Train", "unigrams_dictionary.pickle")
            # self.bi_dict_pth  =  os.path.join("./Processed_Train", "bigrams_dictionary.pickle")

            self.u2i_pth  =  os.path.join(folder, (self.mode + "_unis.txt"))
            self.uni_np_pth  =  os.path.join(folder, (self.mode + "_unis.npy"))

            self.b2i_pth  =  os.path.join(folder, (self.mode + "_bis.txt"))
            self.bi_np_pth  =  os.path.join(folder, (self.mode + "_bis.npy"))

            if self.mode == ("Train") or self.mode == ("Dev"):
                self.label_file_pth = os.path.join(folder, (self.mode + "_labels.txt"))
                self.num_lables_pth = os.path.join(folder, (self.mode + "_numerical_labels.txt"))
                self.lbl_np_pth  =  os.path.join(folder, (self.mode + "_labels.npy"))

    def create_groundtruth(self, parent):
        if self.mode == ("Train"):
            sub = "training"
            # files = ["pku_training.utf8"] # , "cityu_training.utf8", "msr_training.utf8", "as_training.utf8"]
            files = ["pku_training.utf8", "cityu_training.utf8", "msr_training.utf8", "as_training.utf8"]
        elif self.mode == ("Dev"):
            sub = "gold"
            files = ["pku_test_gold.utf8"] #, "msr_test_gold.utf8"]

        print("{:} Dataset will be built from the following files {:}...".format(self.mode, files))
        print()

        no_sp = open(self.no_sp_pth , "w", encoding="utf8")
        label_file = open(self.label_file_pth , "w")
        num_lables = open(self.num_lables_pth , "w")

        for idx, file in enumerate(files):
            with open((os.path.join(parent, sub, file)), 'r', encoding='utf-8') as f:
                print("Opening file {:}...{:}".format(idx+1, file))
                f_content_1 = f.read()
                print("Done reading content.")
                for line_idx, line in enumerate(f_content_1.splitlines()):
                    # if line == "":
                    if not line.rstrip():
                        print("Skipping Empty line {:}!".format(line_idx+1))
                        continue
                    for word in line.split():
                        no_sp.write(word)
                        for idx, char in enumerate(word):
                            if len(word) == 1:
                                label_file.write("S")
                                num_lables.write("3")
                            elif len(word) > 1 and idx == 0:
                                label_file.write("B")
                                num_lables.write("0")
                            elif idx > 0 and idx < len(word)-1:
                                label_file.write("I")
                                num_lables.write("1")
                            elif idx == len(word)-1:
                                label_file.write("E")
                                num_lables.write("2")
                    no_sp.write(u"\n")
                    label_file.write("\n")
                    num_lables.write("\n")
        label_file.close()
        num_lables.close()
        no_sp.close()


    def create_unigram_dict(self):
        uni_list = []
        with open(self.no_sp_pth, "r", encoding="utf8") as f:
            f_content = f.read()
            print("\nCreating the Unigram dictionary....")
            unigrams = [b for l in f_content.splitlines() for b in l]
        unigrams_d = {ni: indi for indi, ni in enumerate(set(unigrams), start=2)}
        print("Done Creating the Unigram dictionary!")
        unigrams_d["PAD"] = 0
        unigrams_d["UNK"] = 1
        print("We have " + str(len(unigrams_d)) + " unique Unigrams (including the PAD and UNK)")
        self.uni_vocab = len(unigrams_d)
        with open(self.uni_dict_pth, "wb") as handle:
            pickle.dump(unigrams_d, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done Saving the Unigram dictionary!")
        print("\n")
        return unigrams_d

    def create_bigram_dict(self):
        bi = []
        print("Creating the Bigram dictionary....")
        with open(self.no_sp_pth, "r", encoding="utf8") as f:
            f_content = f.read()
        for line in f_content.splitlines():
            line = list(line)
            line.append("</s>")
            for idx in range(len(line)-1):
                bi.append(line[idx]+line[idx+1])
            # print(bi)

        bigrams_d = {ni: indi for indi, ni in enumerate(set(bi), start=2)}
        print("Done Creating the Bigram dictionary!")
        bigrams_d["PAD"] = 0
        bigrams_d["UNK"] = 1
        print("We have " + str(len(bigrams_d)) + " unique Bigrams (including the PAD, UNK)")
        self.bi_vocab  = len(bigrams_d)
        with open(self.bi_dict_pth, 'wb') as handle:
            pickle.dump(bigrams_d, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done Saving the Bigram dictionary!")
        return bigrams_d

    def load_unigram_dict(self):
        assert os.path.exists(self.uni_dict_pth), "Training Dictionary does not exist"
        with open(self.uni_dict_pth, 'rb') as handle:
            unigrams_d = pickle.load(handle)
        print("Done Loading the Unigram dictionary!")
        uni_vocab  = len(unigrams_d)
        return unigrams_d

    def load_bigram_dict(self):
        assert os.path.exists(self.uni_dict_pth), "Training Dictionary does not exist"
        with open(self.bi_dict_pth, 'rb') as handle:
            bigrams_d = pickle.load(handle)
        print("Done Loading the Bigram dictionary!")
        bi_vocab  = len(bigrams_d)
        return bigrams_d


    def map_n_grams(self):
        if self.mode == ("Train"):
            unigrams_d = self.create_unigram_dict()
            bigrams_d = self.create_bigram_dict()
        elif self.mode == ("Dev") or ("Test"):
            unigrams_d = self.load_unigram_dict()
            bigrams_d  = self.load_bigram_dict()

        u2i = open(self.u2i_pth, "w")
        b2i = open(self.b2i_pth, "w")

        with open(self.no_sp_pth, 'r', encoding='utf8') as f:
            f_content_3 = f.read()
            for u_line in f_content_3.splitlines():
                for idx in range(len(u_line)):
                    try:
                        uni_to_write = str(unigrams_d[u_line[idx]])
                    except KeyError:
                        uni_to_write = str(unigrams_d["UNK"])
                    u2i.write(uni_to_write + ",")
                u2i.write("\n")
            u2i.close()

            for line in f_content_3.splitlines():
                line = list(line)
                line.append("</s>")
                for idx in range(len(line)-1):
                    try:
                        bi_to_write = str(bigrams_d[line[idx]+line[idx+1]])
                    except KeyError:
                        bi_to_write = str(bigrams_d["UNK"])
                    b2i.write(bi_to_write + ",")
                b2i.write("\n")
        b2i.close()
        print("\nDone Creating N-grams (Unigram & Bigram) map!")


    def create_unigram_data(self):
        to_b_pad_uni_data = []

        with open(self.u2i_pth, 'r') as file:
            for sentence in file.readlines():
                line = sentence.rstrip().split(',')
                del(line[-1])
                if self.mode == ("Train"):
                    if len(line) >= 80:
                        line = line[:80]
                elif self.mode == ("Dev"):
                    if len(line) >= 80:
                        line = line[:80]
                line = list(map(int, line))
                to_b_pad_uni_data.append(line)

        length = max(map(len, to_b_pad_uni_data))
        print("\nCreating {:} Unigrams Numpy array....".format(self.mode))
        padded_uni_training = np.array([xi+[0]*(length-len(xi)) for xi in to_b_pad_uni_data])
        print("Done Creating {:} Unigrams Numpy array!".format(self.mode))
        print("Saving {:} Unigrams Numpy array....".format(self.mode))
        np.save(self.uni_np_pth, padded_uni_training)
        print("Done !\nPadded UNIGRAM Input {:} data is: {:}".format(self.mode, padded_uni_training.shape))

    def create_bigram_data(self):
        to_b_pad_bi_data = []

        with open(self.b2i_pth, 'r') as file:
            for sentence in file.readlines():
                line = sentence.rstrip().split(',')
                del(line[-1])
                if self.mode == ("Train"):
                    if len(line) >= 80:
                        line = line[:80]
                elif self.mode == ("Dev"):
                    if len(line) >= 80:
                        line = line[:80]
                line = list(map(int, line))
                to_b_pad_bi_data.append(line)


        length = max(map(len, to_b_pad_bi_data))
        print("\nCreating {:} Bigrams Numpy array....".format(self.mode))
        padded_bi_training = np.array([xi+[0]*(length-len(xi)) for xi in to_b_pad_bi_data])
        print("Done Creating {:} Bigrams Numpy array!".format(self.mode))

        print("Saving {:} Bigrams Numpy array....".format(self.mode))
        np.save(self.bi_np_pth, padded_bi_training)
        print("Done !\nPadded BIGRAM Input {:} data is: {:}".format(self.mode, padded_bi_training.shape))

    def create_label_data(self):
        padded_cls = []
        with open(self.num_lables_pth, 'r') as file:
            for line in file.readlines():
                if len(line) >= 80:
                    line = line[:80]
                line = line.rstrip()
                line = list(map(int, line))
                padded_cls.append(line)

        length = max(map(len, padded_cls))
        padded_cls = np.array([xi+[0]*(length-len(xi)) for xi in padded_cls])

        print("\nNON One-hot, {:} Label data's shape is: {:}".format(self.mode, padded_cls.shape))

        print("Saving Unigrams Numpy array....")
        np.save(self.lbl_np_pth, padded_cls)
        print("Done !\nPadded {:} Labels data is: {:}".format(self.mode, padded_cls.shape))


if __name__ == '__main__':
    args = parse_args()
    # mode = args.mode
    # parent = args.parent_path

    p = Pocessor(args.mode)

    if args.mode == ("Train") or args.mode == ("Dev"):

        p.create_groundtruth(args.parent_path)
        p.map_n_grams()
        p.create_unigram_data()
        p.create_bigram_data()
        p.create_label_data()
