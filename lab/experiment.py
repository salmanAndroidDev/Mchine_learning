import numpy as np
np.random.seed(1)

def sigmoid(x):
    return 1/ (1 + np.exp(-x))

raw_data = [
    'terrible',
    'bad thing',
    'good',
    'not bad',
    'not good',
    'amazing',
    'bullshit and terible and bad',
]

label_data = np.array([0,0,1,1,0,1,0])

token = list(map(lambda sentence: set(sentence.split(' ')) ,raw_data))

vocabs = set()

for sent in token:
    for word in sent:
        if len(word) > 0:
            vocabs.add(word)
vocabs = list(vocabs)

word2index = {}

for index, word in enumerate(vocabs):
    word2index[word]= index

input_dataset = list()
for sent in token:
    raw_sent = list()
    for word in sent:
        try:
            raw_sent.append(word2index[word])
        except e:
            ""    
    input_dataset.append(list(set(raw_sent)))


alpha, iteration = (0.1, 300)
hidden_size = 5

weights_0_1 = 0.2*np.random.random((len(vocabs),hidden_size)) - 0.1
weights_1_2 = 0.2*np.random.random((hidden_size,1)) - 0.1

total, best_rond = (0,0)

for iter in range(iteration):
    correct= 0
    for i in range(len(input_dataset)):
        
        x = input_dataset[i]
        y = label_data[i]

        layer_1 = sigmoid(np.sum(weights_0_1[x], axis= 0))
        layer_2 = sigmoid(np.dot(layer_1, weights_1_2))
        
        layer_2_delta = layer_2 - y
        layer_1_delta = layer_2_delta.dot(weights_1_2.T)

        weights_0_1[x] -= layer_1_delta * alpha 
        weights_1_2 -= np.outer(layer_1,layer_2_delta) * alpha

        if (np.abs(layer_2_delta) < 0.5):
            correct += 1

        if iter == 110:
            print(layer_2 , ' == ', y)

    if correct > total:
        total = correct
        best_rond = iter
    print("Best iteration is: ", best_rond,' Acuraccy: ', (total/ len(label_data)) * 100,'%')         