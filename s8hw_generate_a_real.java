import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class s8hw_generate_a_real {

    // Simulator configuration
    private int dataPoints = 1000;
    private int features = 10;
    private int hiddenLayers = 2;
    private int hiddenLayerSize = 50;
    private double learningRate = 0.01;
    private int epochs = 100;

    // Data storage
    private INDArray trainData;
    private INDArray trainLabels;
    private INDArray testData;
    private INDArray testLabels;

    // Neural Network
    private INDArray weights1;
    private INDArray weights2;
    private INDArray bias1;
    private INDArray bias2;

    public static void main(String[] args) {
        s8hw_generate_a_real simulator = new s8hw_generate_a_real();
        simulator.generateData();
        simulator.initModel();
        simulator.trainModel();
        simulator.testModel();
    }

    // Generate random data for demonstration
    private void generateData() {
        Random rand = new Random();
        List<Double> data = new ArrayList<>();
        List<Double> labels = new ArrayList<>();

        for (int i = 0; i < dataPoints; i++) {
            for (int j = 0; j < features; j++) {
                data.add(rand.nextDouble() * 10);
            }
            labels.add(rand.nextDouble() * 10);
        }

        double[] dataArr = new double[data.size()];
        double[] labelsArr = new double[labels.size()];
        for (int i = 0; i < data.size(); i++) {
            dataArr[i] = data.get(i);
        }
        for (int i = 0; i < labels.size(); i++) {
            labelsArr[i] = labels.get(i);
        }

        trainData = Nd4j.create(dataArr, new int[] { dataPoints, features });
        trainLabels = Nd4j.create(labelsArr, new int[] { dataPoints, 1 });
        testData = Nd4j.create(dataArr, new int[] { dataPoints, features });
        testLabels = Nd4j.create(labelsArr, new int[] { dataPoints, 1 });
    }

    // Initialize neural network model
    private void initModel() {
        weights1 = Nd4j.rand(features, hiddenLayerSize);
        weights2 = Nd4j.rand(hiddenLayerSize, 1);
        bias1 = Nd4j.zeros(hiddenLayerSize);
        bias2 = Nd4j.zeros(1);
    }

    // Train the model
    private void trainModel() {
        for (int i = 0; i < epochs; i++) {
            INDArray output = forwardPass(trainData);
            INDArray error = output.sub(trainLabels);
            backwardPass(error);
        }
    }

    // Forward pass
    private INDArray forwardPass(INDArray input) {
        INDArray hiddenLayer = input.mmul(weights1).add(bias1);
        hiddenLayer = Transforms.sigmoid(hiddenLayer);
        INDArray output = hiddenLayer.mmul(weights2).add(bias2);
        output = Transforms.sigmoid(output);
        return output;
    }

    // Backward pass
    private void backwardPass(INDArray error) {
        INDArray hiddenLayerError = error.mmul(weights2.transpose());
        hiddenLayerError = hiddenLayerError.mul(Transforms.sigmoidDerivative(hiddenLayerError));

        INDArray weights2Delta = hiddenLayerError.mmul(weights2).mul(learningRate);
        INDArray bias2Delta = error.mul(learningRate);

        INDArray weights1Delta = hiddenLayerError.mmul(weights1).mul(learningRate);
        INDArray bias1Delta = hiddenLayerError.mul(learningRate);

        weights2.subi(weights2Delta);
        bias2.subi(bias2Delta);
        weights1.subi(weights1Delta);
        bias1.subi(bias1Delta);
    }

    // Test the model
    private void testModel() {
        INDArray output = forwardPass(testData);
        System.out.println("Model prediction: " + output);
    }
}