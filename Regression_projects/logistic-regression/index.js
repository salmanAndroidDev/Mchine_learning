require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCsv = require('../load-csv')


const {features, labels, testFeatures, testLabels} = loadCsv('../data/cars.csv', {
    dataColumns:['horsepower', 'displacement', 'weight'],
    labelColumns:['passedemissions'],
    shuffle: true,
    splitTest: 50,
    converters: {
        passedemissions: (e) => e === 'TRUE'? 1 : 0
    }    
})


console.log(testLabels);

