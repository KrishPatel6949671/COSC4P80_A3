import java.util.*;

/**
 * KRISH PATEL 6949671
 * SOM implementation
 */
public class SOM {
    private int gridSize;
    private int inputDim;
    private double[][][] weights; // 3D array for weights [gridSize][gridSize][inputDim]
    private double neighborhoodWidth;


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


    }

    public static void main(String[] args) {
        SOM som = new SOM(10, 3);
    }
}
