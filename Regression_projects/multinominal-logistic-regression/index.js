require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');
const _ = require('lodash');
const mnist = require('mnist-data');

const mnistData = mnist.training(0,10);

const features = mnistData.images.values.map(image => _.flatMap(image));
const labels = mnist.training(0,10).labels.values.map(label => {
    const row = new Array(10).fill(0);  // this will create array, size of 10;
    row[label] = 1;
    return row;
});

console.log(labels);