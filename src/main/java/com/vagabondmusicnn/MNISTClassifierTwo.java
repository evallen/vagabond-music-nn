package com.vagabondmusicnn;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Created by kingds321 on 11/13/16.
 */
public class MNISTClassifierTwo {

    private static Logger log = LoggerFactory.getLogger(MNISTClassifierTwo.class);

    public static void main(String[] args) throws IOException {

        final int numRows = 28; // The number of rows of a matrix

        final int numColumns = 28; // The number of rows of a column

        int outputNum = 10; // The number of possible outcomes (e.g. labels 0 through 9)

        int batchSize = 128; // How many examples to fetch with each step (How many times to go before you see how you did and change weights)

        int rngSeed = 123; // This random-number generator applies a seed to ensure that the same initial weights are used when training.

        int numEpochs = 15; //



            DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);


            DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, true, rngSeed);


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.006)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .regularization(true).l2(1e-4)
                .list()

                .layer(0, new DenseLayer.Builder()
                        .nIn(numRows * numColumns)
                        //Number of input datapoints.


                        .nOut(1000)
                        //Number of output datapoints.


                        .activation("relu")
                        //Activation Function.


                        .weightInit(WeightInit.XAVIER)
                        //Weight Initialization

                        .build())
                    .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)

                            .nIn(1000)
                            .nOut(outputNum)
                            .activation("softmax")
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .pretrain(false).backprop(true)
                    .build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        //print the score with every 1 iteration
        model.setListeners(new ScoreIterationListener(1));

        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
            model.fit(mnistTrain);
        }


        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
        while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");






    }


}
