require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCsv = require('../load-csv');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');

const {features, labels, testFeatures, testLabels} = loadCsv('../data/cars.csv', {
    dataColumns:['horsepower', 'displacement', 'weight'],
    labelColumns:['passedemissions'],
    shuffle: true,
    splitTest: 50,
    converters: {
        passedemissions: (e) => e === 'TRUE'? 1 : 0
    }    
})


const regression = new LogisticRegression(features, labels, {
    learningRate: 0.5,
    iterations: 10,
    batchSize: 10,
    decisionBoundary: 0.1
});


regression.train();
console.log(regression.test(testFeatures, testLabels));


plot({
    x: regression.costHistory.reverse()
})