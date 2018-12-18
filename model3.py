import numpy as np
import utils

# the model doesn't need the entire dataset
from sympy import Matrix


def filterData(dataset):
    data = []
    for obj in dataset:
        # we need to work with index and not a letter
        l = utils.charToIndex[obj["letter"]]
        data.append((obj["data"], l))

    return data


class Model3(object):
    def __init__(self, dataset):
        self.trainset = dataset
        self.W = np.zeros((len(utils.charToIndex), 128))  # instead of 26 Ws of size 128

    def train(self, epochs, log=True):
        correct = incorrect = 0

        prev_y = '$'

        # epochs
        for e in range(1, epochs + 1):
            np.random.shuffle(self.trainset)
            for x, y in self.trainset:

                y_hat = np.argmax(self.predict(x))

                phi_y = self.build_phi(x, (prev_y, y))
                phi_y_hat = self.build_phi(x, (prev_y, y_hat))

                if y != y_hat:
                    self.W += phi_y + phi_y_hat
                    incorrect += 1
                else:
                    correct += 1

                prev_y = y

            if log:
                acc = 100.0 * correct / (correct + incorrect)
                print("# {} train accuracy: {}".format(e, acc))

    def inference(self, x):
        res = np.dot(self.W, x)

        return np.argmax(res)

    def predict(self, x):
        num_of_eng_char = len(utils.charToIndex)
        score_matrix = np.zeros((num_of_eng_char, num_of_eng_char))
        prev_index_track_matrix = np.zeros((num_of_eng_char, num_of_eng_char))

        prev_char = '$'
        for i in range(1, num_of_eng_char):
            curr_char = utils.indexToChar[i]
            y_hat = (prev_char, curr_char)
            phi = self.build_phi(x, y_hat)
            score = utils.matrix_inner_product(self.W, phi)
            score_matrix[0][i] = score
            prev_index_track_matrix[0][i] = 0

        for i in range(1, num_of_eng_char):
            for j in range(1, num_of_eng_char):
                curr_char = utils.indexToChar[j+1]
                best_score, best_score_index = self.argmax(x, curr_char, score_matrix, i)
                score_matrix[i][j] = best_score
                prev_index_track_matrix[i][j] = best_score_index

        y_hat = np.zeros((num_of_eng_char))
        best_score = -1
        for i in range(num_of_eng_char):
            if best_score < score_matrix[num_of_eng_char-1][i]:
                y_hat[num_of_eng_char-1] = i
                best_score = score_matrix[num_of_eng_char-1][i]

        for i in (num_of_eng_char-2, 0):
            y_hat[i] = prev_index_track_matrix[i+1][y_hat[i+1]]

        return y_hat

    def build_phi(self, x, y_hat):
        num_of_eng_char = len(utils.charToIndex)
        y_index = utils.charToIndex(x["letter"])
        prev_char, curr_char = y_hat
        phi = np.zeros((num_of_eng_char, num_of_eng_char))
        phi[y_index] = x.data
        phi[utils.charToIndex[prev_char]][y_index] = 1
        return phi

    def argmax(self, x, curr_char, score_matrix, index):
        max_value_y_hat = -1
        max_y_hat_index = -1
        for y_hat in utils.charToIndex:
            phi = self.build_phi(x, (y_hat, curr_char))
            potential_y_hat = utils.matrix_inner_product(self.W, phi) + score_matrix[index-1][y_hat]
            if potential_y_hat > max_value_y_hat:
                max_value_y_hat = potential_y_hat
                max_y_hat_index = utils.charToIndex[y_hat]

        return max_value_y_hat, max_y_hat_index


def main():
    trainset = utils.load_data_bigrams('./data/letters.train.data')
    testset = utils.load_data_bigrams('./data/letters.test.data')

    model = Model3(trainset)
    model.train(3)

    correct = incorrect = 0
    for x, y in testset:
        y_tag = model.inference(x)
        correct += (y == y_tag)
        incorrect += (y != y_tag)

    acc = 100.0 * correct / (correct + incorrect)

    print("test accuracy: {}".format(acc))


if __name__ == "__main__":
    main()
