import numpy as np 

# mapping for predictions
charToIndex = {c:i for i, c in enumerate('abcdefghijklmnopqrstuvwxyz')}
indexToChar = {i:c for i, c in enumerate('abcdefghijklmnopqrstuvwxyz')}

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

        dataset.append(obj)

    return dataset
