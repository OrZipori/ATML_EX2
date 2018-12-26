import numpy as np 
import matplotlib.pyplot as plt

# mapping for predictions
charToIndex = {c:i for i, c in enumerate('$abcdefghijklmnopqrstuvwxyz')}
indexToChar = {i:c for i, c in enumerate('$abcdefghijklmnopqrstuvwxyz')}

def loadData(filename):
    dataset = []
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    
    # organize it according to readme.md
    for line in lines:
        obj = {}
        v = line.split() # split by white space

        obj["id"] = v[0]
        obj["letter"] = v[1] 
        obj["next_id"] = v[2]
        obj["word_id"] = v[3]
        obj["position"] = v[4]
        obj["fold"] = v[5]
        obj["data"] = np.asarray(v[6:], dtype=np.int)

        # get the previous letter in the same word
        if (int(v[4]) == 1):
            obj["prevLetter"] = '$'
        else: 
            obj["prevLetter"] = dataset[len(dataset) - 1]["letter"]

        dataset.append(obj)

    return dataset

def loadDataPart3(filename):
    dataset = []
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    
    word = []
    # organize it according to readme.md
    for line in lines:
        obj = {}
        v = line.split() # split by white space

        obj["letter"] = v[1] 
        obj["data"] = np.asarray(v[6:], dtype=np.int)
        
        # end of word
        word.append(obj)
        if (int(v[2]) == -1):
            dataset.append(word)
            word = []

    return dataset

def matrix_inner_product2(mat_a, mat_b):
    score = 0
    for v1, v2 in zip(mat_a, mat_b):
        score = score + np.dot(v1, v2)

    return score

def matrix_inner_product1(mat_a, mat_b):
    return np.dot(mat_a, mat_b)

def matrix_inner_product(mat_a, mat_b):
    return np.dot(mat_a, mat_b.transpose()).trace()

def plotHeatMap(W):
    # https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
    pass