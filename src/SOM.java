import java.util.*;
import java.io.*;

/**
 * KRISH PATEL 6949671
 * SOM implementation
 */
public class SOM {
    private int gridSize;
    private int inputDim;
    private double[][][] weights; // 3D array for weights [gridSize][gridSize][inputDim]

    public SOM(int gridSize, int inputDim){
        this.gridSize = gridSize;
        this.inputDim = inputDim;
        this.weights = new double[gridSize][gridSize][inputDim];

        initializeWeights();
    }

    private void initializeWeights(){
        Random rand = new Random(42); // Fixed seed for reproducibility
        for(int i=0; i<gridSize; i++){
            for(int j=0; j<gridSize; j++){
                for(int k=0; k<inputDim; k++){
                    weights[i][j][k] = rand.nextDouble(0.0, 1.0);
                }
            }
        }
    }

    private double euclideanDistance(double[] input, double[] weight){
        double sum = 0.0;
        for(int i=0; i<input.length; i++){
            sum += Math.pow(input[i] - weight[i], 2);
        }
        return Math.sqrt(sum);
    }

    private double toroidalDistance(int x1, int y1, int x2, int y2){
        int dx = Math.abs(x1 - x2);
        int dy = Math.abs(y1 - y2);
        //wrap around
        dx = Math.min(dx, gridSize - dx);
        dy = Math.min(dy, gridSize - dy);

        return Math.sqrt(dx * dx + dy * dy);
    }

    private double radialBasisFunction(double distance, double r){
        return Math.exp(-distance * distance/(2 * r * r));
    }

    private double calculateRadius(double currentT, double finalT){
        double initialRadius = 2.0;
        double finalRadius = 0.5;
        return initialRadius - (initialRadius - finalRadius) * (currentT / finalT);
    }

    private int[] findClosestVector(double[] input){
        double minDist = Double.MAX_VALUE;
        int[] closestVectorIndex = new int[2];

        for(int i=0; i<gridSize; i++){
            for(int j=0; j<gridSize; j++){
                double dist = euclideanDistance(input, weights[i][j]);
                if(dist < minDist){
                    minDist = dist;
                    closestVectorIndex[0] = i;
                    closestVectorIndex[1] = j;
                }
            }
        }
        return closestVectorIndex;
    }

    public void train(double[][] inputs, int epochs, double initialLearningRate){
        Random rand = new Random();

        for (int t=0; t<epochs; t++){
            //dynamic learning rate, based on current epoch
            double learningRate = initialLearningRate * (1.0 - (double)t / epochs);

            //calculate radius for this epoch, based on linear decay
            double r = calculateRadius(t, epochs);

            // Select a random input vector
            double[] input = inputs[rand.nextInt(inputs.length)];

            //find the closest vector to randomly selected input
            int[] closestIndex = findClosestVector(input);

            for(int i=0; i<gridSize; i++){
                for(int j=0; j<gridSize; j++){
                    //calculate distance to closest vector with wrap-around
                    double distToClosest = toroidalDistance(closestIndex[0], closestIndex[1] , i, j);

                    //calculate influence based on radial basis function (neighbourhood function)
                    double influence = radialBasisFunction(distToClosest, r);

                    //update weights
                    for(int k=0; k<inputDim; k++) {
                        double delta = learningRate * influence * (input[k] - weights[i][j][k]);
                        weights[i][j][k] += delta;
                    }
                }
            }

            if((t+1) % 1000 == 0){
                System.out.println("Completed epoch: " + (t+1) + "/ " + epochs);
                System.out.println("Learning Rate: " + learningRate + ", Radius: " + r);

            }
        }
    }

    private double[][] generateHeatMap(double[][] inputs, int[] labels){
        double heatMap[][] = new double[gridSize][gridSize];
        double countMap[][] = new double[gridSize][gridSize];

        double r = 0.5; //fixed radius for heatmap generation

        for(int s=0; s< inputs.length; s++){
            double[] input = inputs[s];
            int label = labels[s];

            //calculate activation for each node
            for(int i=0; i<gridSize; i++){
                for(int j=0; j<gridSize; j++){
                    double dist = euclideanDistance(input, weights[i][j]);
                    double activation = radialBasisFunction(dist, r);

                    double signedActivation = (label==0) ? activation : -activation;
                    heatMap[i][j] += signedActivation;
                    countMap[i][j] += activation;
                }
            }
        }

        //normalize heatmap
        for(int i=0; i<gridSize; i++){
            for(int j=0; j<gridSize; j++){
                if(countMap[i][j] > 0){
                    heatMap[i][j] /= countMap[i][j];
                }
            }
        }

        return heatMap;
    }

    public void exportHeatMap(){

    }

    public static void main(String[] args) {
        SOM som = new SOM(10, 3);
    }
}
