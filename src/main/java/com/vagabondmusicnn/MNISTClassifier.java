package com.vagabondmusicnn;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
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
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
