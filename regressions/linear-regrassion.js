const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegrassion{
    // passino features and labels arguments must be instances of Tensor Object;
    constructor(features, labels, options) {
        const {mean, variance} = tf.moments(features);
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);
                
        // this will assign default values for options, if we don't specify options param
        //else it will copy options object into instance variable;
        this.options = Object.assign({learningRate: 1.5, iterations: 1000}, options);
        // by convention it's better to initial m and b with 0 or 1;
  
        this.weight = tf.zeros([2,1]);
    }

    gradientDecent() {
        const currentGuesses = this.features.matMul(this.weight)
        const differences = currentGuesses.sub(this.labels);
        
        const slopes = this.features
                .transpose()
                .matMul(differences)
                .div(this.features.shape[0])   
                // .mul(2)// i should test this thing to see the result.

       this.weight =  this.weight.sub(slopes.mul(this.options.learningRate));

    }

    gradientDecentOldVersion(){
        const currentGuessesForMPG = this.features.map(row => {
            return this.m * row[0] + this.b;
        });

        let bSlope = _.sum(currentGuessesForMPG.map((guess, index) => {
                    return guess - this.labels[index][0];
                    })) * 2 / this.features.length;            
    
    
        let mSlope = _.sum(currentGuessesForMPG.map((guess, index) =>{
            return (-1 * this.features[index][0]) * (this.labels[index][0] - guess);
        })) * 2 / this.features.length;                 
    
        this.m = this.m - (mSlope * this.options.learningRate);
        this.b = this.b - (bSlope * this.options.learningRate);
    
    }

    train() {
        for(let i = 0; i < this.options.iterations; i++){
            this.gradientDecent();
        }
    }    

    test(testFeatures, testLabels) {
         testFeatures = this.processFeatures(testFeatures);
         testLabels = tf.tensor(testLabels);
         
        const predictions = testFeatures.matMul(this.weight);

        const res = testLabels.sub(predictions)
                    .pow(2)
                    .sum()
                    .get();

        const tot = testLabels.sub(testLabels.mean())
                        .pow(2)
                        .sum()
                        .get();
        
        return 1 - res / tot;                
    }

    processFeatures(features){
        features = tf.tensor(features);

        if(this.mean && this.variance){
            features = features.sub(this.mean).div(this.variance.pow(0.5));
        } else {
            features = this.standardize(features);
        }

        features = tf.ones([features.shape[0], 1]).concat(features,1)    

        return features;
    }

    standardize(features) {
        const {mean, variance} = tf.moments(features, 0);

        this.mean = mean;
        this.variance = variance;
        
        return features.sub(mean).div(variance.pow(0.5));
    }

    predict(features){
        features = this.processFeatures(features);
        return features.matMul(this.weight);
    }

}

module.exports = LinearRegrassion;