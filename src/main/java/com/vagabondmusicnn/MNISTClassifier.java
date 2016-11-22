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
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * Created by Evan on 11/13/2016.
 */
public class MNISTClassifier {

    private static Logger log = LoggerFactory.getLogger(MNISTClassifier.class);
    private static boolean saving = false;

    public static void main (String[] args) throws IOException {
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10;
        int batchSize = 128;
        int rngSeed = 123;
        int numEpochs = 1;

        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);



        log.info("Build model...");
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
                    .nOut(1000)
                    .activation("relu")
                    .weightInit(WeightInit.XAVIER)
                    .build())

                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                      .nIn(1000)
                    .nOut(outputNum)
                    .activation("softmax")
                    .weightInit(WeightInit.XAVIER)
                    .build())

                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork (conf);
        model.init();

        model.setListeners(new ScoreIterationListener(1));

        log.info("Train model...");
        for (int i = 0; i < numEpochs; i++) {
            model.fit(mnistTrain);
        }

        log.info("Evaluate model ...");
        Evaluation eval = new Evaluation(outputNum);
        while (mnistTest.hasNext()) {
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(), output);
        }

        if (saving) {
            File saveLocation = new File("products/evanMNISTClassifier");
            saveLocation.getParentFile().mkdirs();
            boolean saveUpdater = true;
            ModelSerializer.writeModel(model, saveLocation, saveUpdater);
        }

        log.info(eval.stats());
        log.info("***********Example finished**********");

    }
}
