package com.vagabondmusicnn;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Created by Evan on 11/17/2016.
 */
public class MNISTTester {

    private static Logger log = LoggerFactory.getLogger(MNISTTester.class);

    public static void main(String[] args) throws IOException {
        int width = 28;
        int height = 28;
        int channels = 1;
        int batchSize = 1;
        int outputNum = 10;
        Random rand = new Random(123);

        File testData = new File("src/main/resources/numbers");
        File modelFile = new File("products/evanMNISTClassifier");

        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, rand);

        ParentPathLabelGenerator labeller = new ParentPathLabelGenerator();

        ImageRecordReader reader = new ImageRecordReader(height, width, channels, labeller);

        reader.initialize(test);

        DataSetIterator dataIterator =  new RecordReaderDataSetIterator(reader, batchSize, 1, outputNum);

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelFile);

        log.info("Evaluating set...");

        Evaluation eval = new Evaluation();

        /*DataNormalization normalizer = new ImagePreProcessingScaler(0, 1);
        normalizer.fit(dataIterator);
        dataIterator.setPreProcessor(normalizer);*/

        while (dataIterator.hasNext()) {
            DataSet img = dataIterator.next();
            INDArray out = model.output(img.getFeatures());
            eval.eval(img.getLabels(), out);
        }

        log.info(eval.stats());
        log.info("- - - Finished! - - -");
    }

}
