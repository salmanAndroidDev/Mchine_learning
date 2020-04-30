require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const linearRegression = require('./linear-regrassion');



let {features, labels, testFeatures, testLabels} = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower'],
    labelColumns: ['mpg']
});


const regression = new linearRegression(features, labels, {
    learningRate: 0.01,
    iterations: 100
})

regression.train()
//const r2 = regression.test(testFeatures, testLabels);

regression.predict([[95]]).print();
