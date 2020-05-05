const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegrassion{
    // passino features and labels arguments must be instances of Tensor Object;
    constructor(features, labels, options) {
        const {mean, variance} = tf.moments(features);
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);
        this.mseHistory = [];
        this.bHistory = [];        
        // this will assign default values for options, if we don't specify options param
        //else it will copy options object into instance variable;
        this.options = Object.assign({learningRate: 1.5, iterations: 1000}, options);
        // by convention it's better to initial m and b with 0 or 1;
  
        this.weight = tf.zeros([this.features.shape[1],1]);

    }

    gradientDecent(features, labels) {
        const currentGuesses = features.matMul(this.weight)
        const differences = currentGuesses.sub(labels);
        
        const slopes = features
                .transpose()
                .matMul(differences)
                .div(features.shape[0])   
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

        const batchQuantity = Math.floor(
             this.features.shape[0] / this.options.batchSize
        );
            
        for(let i = 0; i < this.options.iterations; i++){
            for(let j = 0; j < batchQuantity; j++){
                const {batchSize} = this.options;
                const startIndex = j * batchSize;
                
                // it slices the entire features into batches
                const featureSlice = this.features.slice(
                    [startIndex, 0],    // starts at [startIndex, 0 column]
                    [batchSize, -1]);    // we wanna take batchSize record and -1 means entire features

                const labelSlice = this.labels.slice(
                    [startIndex, 0],
                    [batchSize, -1]);

                this.gradientDecent(featureSlice, labelSlice);
            }
            
            this.recordMSE();
            this.updateLearningRate();
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
        console.log(features.shape)
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
        return this.processFeatures(features).matMul(this.weight);
    }

    recordMSE() {
        const mse = this.features.matMul(this.weight)
            .sub(this.labels)
            .pow(2)
            .sum()
            .div(this.features.shape[0])
            .get();
    
            // unshift will make order from new to old rather than from old to new;
        this.mseHistory.unshift(mse);
        }

    updateLearningRate() {

        if(this.mseHistory.length < 2)
            return;

            // [0] is newer one;
        if(this.mseHistory[0] > this.mseHistory[1]){
            this.options.learningRate /= 2;
        }else { // it'll increase learningRate by 5%
            this.options.learningRate *= 1.05;
        }
    }
}

module.exports = LinearRegrassion;