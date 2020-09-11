import math
from collections import Counter
import numpy as np
import sys

np.random.seed(1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


alpha, iterations = (0.01, 2)
hidden_size = 100


file = open('reviews.txt')
text = file.readlines()
file.close()

file = open('labels.txt')
labels = file.readlines()
file.close()


token = list(map(lambda x: set(x.split(' ')), text))

all_words = set()
for sentence in token:
    for word in sentence:
        if len(word) > 0:
            all_words.add(word)

all_words = list(all_words)

word_to_index = {}
for i, word in enumerate(all_words):
    word_to_index[word] = i

input_dataset = list()
i = 0
for sent in token:
    sent_index = list()
    for word in sent:
        try:
            sent_index.append(word_to_index[word])
        except:
            """"""

    input_dataset.append(sent_index)

target_dataset = list()

for i in labels:
    if i == 'positive\n':
        target_dataset.append(1)
    else:
        target_dataset.append(0)

weights_0_1 = 0.2*np.random.random((len(all_words), hidden_size)) - 0.1
weights_1_2 = 0.2*np.random.random((hidden_size, 1)) - 0.1

correct, total = 0, 0

for iter in range(iterations):
    for i in range(len(input_dataset) - 1000):
        x, y = (input_dataset[i], target_dataset[i])
        layer1 = sigmoid(np.sum(weights_0_1[x], axis=0))
        layer2 = sigmoid(np.dot(layer1, weights_1_2))

        layer_2_delta = layer2 - y
        layer_1_delta = layer_2_delta.dot(weights_1_2.T)

        weights_0_1[x] -= layer_1_delta * alpha
        weights_1_2 -= np.outer(layer1, layer_2_delta) * alpha

        if(np.abs(layer_2_delta) < 0.5):
            correct += 1
        total += 1
        if(i % 10 == 9):
            progress = str(i/float(len(input_dataset)))
            sys.stdout.write('\rIter:'+str(iter)
                             + ' Progress:'+progress[2:4]
                             + '.'+progress[4:6]
                             + '% Training Accuracy:'
                             + str(correct/float(total)) + '%')
    print()

correct, total = (0, 0)
for i in range(len(input_dataset)-1000, len(input_dataset)):
    x = input_dataset[i]
    y = target_dataset[i]
    layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
    layer_2 = sigmoid(np.dot(layer_1, weights_1_2))

    if(np.abs(layer_2 - y) < 0.5):
        correct += 1
    total += 1
print("Test Accuracy:" + str(correct / float(total)))


def similar(target='beautiful'):
    target_index = word_to_index[target]
    scores = Counter()
    for word, index in word_to_index.items():
        raw_difference = weights_0_1[index] - (weights_0_1[target_index])
        squared_difference = raw_difference * raw_difference
        scores[word] = -math.sqrt(sum(squared_difference))
    return scores.most_common(10)


while 1:
    word = input('enter word:')
    if word == 'q':
        break
    print(similar(word))
    print("***************************")
