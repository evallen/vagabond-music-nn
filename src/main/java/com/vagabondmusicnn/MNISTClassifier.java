package com.vagabondmusicnn;

import org.bytedeco.javacv.ImageTransformer;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;

/**
 * Created by Evan on 11/13/2016.
 */
public class MNISTClassifier {
    public static void main (String[] args) {
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10;
        int batchSize = 128;
        int rngSeed = 123;
        int numEpochs = 15;

        try {
            DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
            DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);
        } catch (IOException e) {
            e.printStackTrace();
        }

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.006)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .regularization(true).l2(1e-4)
                .list().build();
    }
}
