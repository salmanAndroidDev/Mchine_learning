import numpy as np
np.random.seed(1)

def sigmoid(x):
    return 1/ (1 + np.exp(-x))

raw_data = [
    'so terrible dont buy',
    'terrible and bad and bullshit',
    'bad idea behind this game',
    'good idea behind this game',
    'not bad actually it is good and amazing',
    'not good actually it is bad and terrible and bullshit',
    'amazing idea with the best',
    'bullshit and terible and bad',
    'it is super good and the',
    'what the fuck with this bad and bullshit and terrible idea',
    'it got a git golden buzzar for this nice and good and amazing idea',
    'not that much bad, actually it is good',
    'not good at all, actually it is bad never buy it',
    'amazing idea it worth buying its so good so clever so amazing'
]

label_data = np.array([0,0,0,1,1,0,1,0,0,0,1,1,0,1])

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
hidden_size = 3

weights_0_1 = 0.2*np.random.random((len(vocabs),hidden_size)) - 0.1
weights_1_2 = 0.2*np.random.random((hidden_size,1)) - 0.1

total, best_rond = (0,0)
best_weight_0_1 = weights_0_1
best_weight_1_2 = weights_1_2
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

    if correct > total:
        total = correct
        best_rond = iter
        # initializing the weight
        best_weight_0_1 = weights_0_1
        best_weight_1_2 = weights_1_2
    print("Best iteration is: ", best_rond,' Acuraccy: ', (total/ len(label_data)) * 100,'%', ' current ', iter)
    

weights_0_1 = best_weight_0_1
weights_1_2 = best_weight_1_2
while(True):
    sentence = input("Enter the sentence: ")

    if sentence  == 'q':
        break
    else:
        sent = sentence.split(' ')
        raw_sentence = list()
        for word in sent:
            try:
                raw_sentence.append(word2index[word])
            except e:
                ""   
        x = list(set(raw_sentence))

        layer_1 = sigmoid(np.sum(weights_0_1[x], axis= 0))
        layer_2 = sigmoid(np.dot(layer_1, weights_1_2))
            
        message = "prediction:{} weight: {}".format(layer_2, weights_0_1[x])
        
        print(message)
