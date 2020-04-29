const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegrassion{
    // passino features and labels arguments must be instances of Tensor Object;
    constructor(features, labels, options) {
        this.features = tf.tensor(features);
        this.labels = tf.tensor(labels);
        
        this.features=  tf.ones([this.features.shape[0], 1]).concat(features,1)
        
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
         testFeatures = tf.tensor(testFeatures);
         testLabels = tf.tensor(testLabels);
         
        testFeatures=  tf.ones([testFeatures.shape[0], 1]).concat(testFeatures,1)

        const predictions = testFeatures.matMul(this.weight);

        predictions.print();
    }
}

module.exports = LinearRegrassion;