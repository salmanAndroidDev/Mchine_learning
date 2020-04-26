require('@tensorflow/tfjs-node');
const tf = require("@tensorflow/tfjs");
const loadCsv = require("./load-csv");

function knn(features, labels, predictionPoint, k){

  const {mean, variance} = tf.moments(features, 0);
  
  scaledPoint = predictionPoint.sub(mean).div(variance.pow(0.5));

  return features
  .sub(mean)
  .div(variance.pow(0.5))
  .sub(scaledPoint)
    .pow(2)
    .sum(1)
    .pow(0.5)
    .expandDims(1)
    .concat(labels,1)
    .unstack()
    .sort((a,b) => a.get(0) > b.get(0) ? 1 : -1)
    .slice(0,k)
    .reduce((acc, pair) => acc + pair.get(1) ,0) / k;

}


let {features, labels, testFeatures, testLabels} = loadCsv('kc_house_data.csv', {
  shuffle: true,
  splitTest: 10,
  dataColumns:['sqft_lot', 'sqft_living', 'Floors' ,'Bedrooms'],
  labelColumns: ['price']
});

features = tf.tensor(features);
labels = tf.tensor(labels);
// testFeatures = tf.tensor(testFeatures);
// testLabels = tf.tensor(testLabels);

  testFeatures.forEach((testPoint, i) => {
    const result = knn(features, labels, tf.tensor(testPoint), 5)
    const err = (testLabels[i][0] - result) / testLabels[i][0];
    console.log(" ------> " , err * 100);
  });