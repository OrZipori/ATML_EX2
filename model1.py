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

class Model1(object):
    def __init__(self, dataset):
        self.trainset = dataset
        self.W = np.zeros((len(utils.charToIndex), 128)) # instead of 26 Ws of size 128

    def train(self, epochs, log=True):
        correct = incorrect = 0

        # epochs
        for e in range(1, epochs + 1):
            np.random.shuffle(self.trainset)
            for x, y in self.trainset:
                res = np.dot(self.W, x)

                y_tag = np.argmax(res)

                # update the vectors 
                if (y != y_tag):
                    self.W[y] += x
                    self.W[y_tag] -= x
                    incorrect += 1
                else:
                    correct += 1
            
            if (log):
                acc = 100.0 * correct / (correct + incorrect)
                print("# {} train accuracy: {}".format(
                    e, acc
                ))

    def inference(self, x):
        res = np.dot(self.W, x)

        return np.argmax(res)

def main():
    train = utils.loadData('./data/letters.train.data')
    test = utils.loadData('./data/letters.test.data')

    trainset = filterData(train)
    testset = filterData(test)

    model = Model1(trainset)
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
    