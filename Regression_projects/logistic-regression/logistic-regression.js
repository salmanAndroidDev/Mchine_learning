const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LogisticRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.costHistory = [];

    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000, decisionBoundary: 0.5 },
      options
    );

    this.weights = tf.zeros([this.features.shape[1], 1]);
  }

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights).sigmoid();
    const differences = currentGuesses.sub(labels);

    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    );

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * this.options.batchSize;
        const { batchSize } = this.options;

        const featureSlice = this.features.slice(
          [startIndex, 0],
          [batchSize, -1]
        );
        const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);

        this.gradientDescent(featureSlice, labelSlice);
      }
      console.log(this.options.learningRate)
      this.recordCost();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    return this.processFeatures(observations)
    .matMul(this.weights)
    .sigmoid()
    .greater(this.options.decisionBoundary) // this will return true or false.
    .cast('float32'); // cuz it returns true or false we have to convert it to floast32 to manipulate it.
  }

  test(testFeatures, testLabels) {
    const predictins = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels);
      // we use abs to convert negative values -1 into positive +1 values.
    const incorrects = predictins.sub(testLabels).abs().sum().get();  // we gotta get the value outside of tensor using get().
  
   return (predictins.shape[0] - incorrects) / predictins.shape[0];
  }

  processFeatures(features) {
    features = tf.tensor(features);
    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }

    return features;
  }

  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);

    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5));
  }

  recordCost() {
    const guesses = this.features.matMul(this.weights).sigmoid();
  
    const termOne = this.labels.transpose().matMul(guesses.log());


    const termTwo = this.labels
          .mul(-1)
          .add(1)
          .transpose()
          .matMul(
            guesses
            .mul(-1)
            .add(1)
            .log()
            )
      
      
     const cost = termOne.add(termTwo)
            .div(this.features.shape[0])
            .mul(-1)
            .get(0,0);
    
    this.costHistory.unshift(cost);
          }

  updateLearningRate() {
    if (this.costHistory.length < 2) {
      return;
    }

    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LogisticRegression;
