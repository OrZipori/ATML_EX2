import numpy as np
import utils

# the model doesn't need the entire dataset
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
        #self.W = np.zeros((len(utils.charToIndex), 155))  # instead of 26 Ws of size 128
        self.W = np.zeros((27*128 + 27*27))

    def train(self, epochs, log=True):        
        # epochs
        for e in range(1, epochs + 1):
            np.random.shuffle(self.trainset)
            correct = incorrect = 0
            for i, word in enumerate(self.trainset):
                y_hat = self.predict(word) 

                prev_y = '$'
                prev_y_h = '$'
                for pred, xi in zip(y_hat, word):
                    l = utils.charToIndex[xi["letter"]]

                    if (pred != l):
                        # convert to char
                        y_h = utils.indexToChar[pred]
                        y_i = xi["letter"]
                        phi_y = self.create_phi(xi, (prev_y, y_i))
                        phi_y_hat = self.create_phi(xi, (prev_y_h ,y_h))

                        # update
                        self.W += phi_y - phi_y_hat
                        incorrect += 1
                    else:
                        correct += 1

                    prev_y = xi["letter"]
                    prev_y_h = y_h

                if (i % 500 == 499):
                    acct = 100.0 * correct / (correct + incorrect)
                    print("#{} examples accuracy: {} ({} / {})".format(i, acct, correct, correct + incorrect))
                    
            if log:
                acc = 100.0 * correct / (correct + incorrect)
                print("# {} train accuracy: {} ({} / {})".format(e, acc, correct, correct + incorrect))

    def inference(self, x):
        res = self.predict(x)

        return np.argmax(res)

    def predict(self, word):
        num_of_eng_char = len(utils.charToIndex)
        score_matrix = np.zeros((len(word), num_of_eng_char))
        prev_index_track_matrix = np.zeros((len(word), num_of_eng_char), dtype=np.int)

        prev_char = '$'
        for i in range(1, num_of_eng_char):
            curr_char = utils.indexToChar[i]
            y_hat = (prev_char, curr_char)
            phi = self.create_phi(word[0], y_hat)
            score = utils.matrix_inner_product(self.W, phi)
            score_matrix[0][i] = score
            prev_index_track_matrix[0][i] = 0

        for i in range(1, len(word)):
            for j in range(0, num_of_eng_char - 1):
                curr_char = utils.indexToChar[j+1]
                best_score, best_score_index = self.argmax(word[i], curr_char, score_matrix, i)
                score_matrix[i][j] = best_score
                prev_index_track_matrix[i][j] = best_score_index

        y_hat = np.zeros((len(word)), dtype=np.int)
        best_score = -1
        for i in range(num_of_eng_char):
            if best_score < score_matrix[len(word) - 1][i]:
                y_hat[len(word) - 1] = i
                best_score = score_matrix[len(word) - 1][i]

        for i in range(len(word) - 2, 0, -1):
            y_hat[i] = prev_index_track_matrix[i+1][y_hat[i+1]]
        
        # print("*" * 10)
        # print(score_matrix)
        # print(prev_index_track_matrix)
        # print(y_hat)

        return y_hat

    def build_phi(self, x, y_hat):
        num_of_eng_char = len(utils.charToIndex)
        prev_char, curr_char = y_hat
        y_index = utils.charToIndex[curr_char]
        y_prev = utils.charToIndex[prev_char]

        phi = np.zeros((num_of_eng_char, 155))
        phi[y_index, :128] = x["data"]
        phi[y_prev][128 + y_index] = 1
        return phi

    def create_phi(self, x, y_hat):
        y = utils.charToIndex[y_hat[1]]
        y_prev = utils.charToIndex[y_hat[0]]
        PHI = np.zeros((27 * 128))
        PHI[y * 128:(y + 1) * 128] = x["data"]
        PHI_prev = np.zeros((27*27))
        PHI_prev[y_prev * y] = 1
        return np.concatenate((PHI, PHI_prev))

    def argmax(self, x, curr_char, score_matrix, index):
        max_value_y_hat = -np.inf
        max_y_hat_index = -1
        for y_hat in utils.charToIndex:
            phi = self.create_phi(x, (y_hat, curr_char))
            potential_y_hat = utils.matrix_inner_product(self.W, phi) + score_matrix[index-1][utils.charToIndex[y_hat]]
            if potential_y_hat > max_value_y_hat:
                max_value_y_hat = potential_y_hat
                max_y_hat_index = utils.charToIndex[y_hat]

        return max_value_y_hat, max_y_hat_index


def main():
    trainset = utils.loadDataPart3('./data/letters.train.data')
    testset = utils.loadDataPart3('./data/letters.test.data')

    model = Model3(trainset)
    model.train(3)

    correct = incorrect = 0
    for word in testset:
        y_hat = model.predict(word) 

        for pred, xi in zip(y_hat, word):
            l = utils.charToIndex[xi["letter"]]

            if (pred != l):
                # convert to char
                incorrect += 1
            else:
                correct += 1

    acc = 100.0 * correct / (correct + incorrect)

    print("test accuracy: {}".format(acc))


if __name__ == "__main__":
    main()