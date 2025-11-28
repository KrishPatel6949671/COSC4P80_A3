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
    }

    private void initializeWeights(double[][] inputs){
        Random rand = new Random(42);
        for (int i=0; i<gridSize; i++){
            for (int j=0; j<gridSize; j++){
                double[] sample = inputs[rand.nextInt(inputs.length)];
                System.arraycopy(sample, 0, weights[i][j], 0, inputDim);
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
        double initialRadius = 20;
        double finalRadius = 2;
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
        Random rand = new Random(42); // Fixed seed for reproducibility
        initializeWeights(inputs);

        for (int t=0; t<epochs; t++){
            //dynamic learning rate, based on current epoch
            double learningRate = initialLearningRate * Math.exp(-2.0 * t / epochs);

            //calculate radius for this epoch, based on linear decay
            double r = calculateRadius(t, epochs);

            double[] input;
            int[] closestIndex;

            List<Integer> inputIndices = new ArrayList<>();
            //shuffle input indices to run training on all inputs per epoch
            for(int i=0; i<inputs.length; i++) inputIndices.add(i);
            Collections.shuffle(inputIndices, rand);

            for(int idx : inputIndices){
                input = inputs[idx];
                closestIndex = findClosestVector(input);

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
            }

            if((t+1) % 1000 == 0){
                System.out.println("Completed epoch: " + (t+1) + "/" + epochs);
                System.out.println("Learning Rate: " + learningRate + ", Radius: " + r);
            }
        }
    }

    private double[][] generateHeatMap(double[][] inputs, int[] labels){
        double[][] goodMap = new double[gridSize][gridSize];
        double[][] badMap = new double[gridSize][gridSize];
        double[][] heatMap = new double[gridSize][gridSize];

        //double r = 0.5/ Math.sqrt(2*Math.log(2));
        double r = 10;

        for(int s=0; s< inputs.length; s++){
            double[] input = inputs[s];
            int label = labels[s];

            double sign = (label == 0) ? 1.0 : -1.0;

            //calculate activation per node for every input
            for(int i=0; i<gridSize; i++){
                for(int j=0; j<gridSize; j++){
                    double dist = euclideanDistance(input, weights[i][j]);
                    double activation = radialBasisFunction(dist, r);

                    heatMap[i][j] += sign * activation;
                }
            }
        }

        //normalize heatmap to range [-1, 1]
        double maxAbsHeat = 0.0;
        for(int i=0; i<gridSize; i++){
            for(int j=0; j<gridSize; j++){
                maxAbsHeat = Math.max(maxAbsHeat, Math.abs(heatMap[i][j]));
            }
        }

        if(maxAbsHeat > 0){
            for(int i=0; i<gridSize; i++){
                for(int j=0; j<gridSize; j++){
                    heatMap[i][j] /= maxAbsHeat;
                }
            }
        }

        System.out.println("Sample heatmap values:");
        for(int i=0; i<Math.min(3, gridSize); i++){
            for(int j=0; j<Math.min(3, gridSize); j++){
                System.out.printf("  [%d][%d]: heat=%.4f%n",
                        i, j, heatMap[i][j]);
            }
        }

        return heatMap;
    }

    public void exportHeatMap(double[][] heatmap, String filename) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            for (int i = 0; i < gridSize; i++) {
                for (int j = 0; j < gridSize; j++) {
                    writer.write(String.format("%.6f", heatmap[i][j]));
                    if (j < gridSize - 1) {
                        writer.write(",");
                    }
                }
                writer.newLine();
            }
        }
    }

    public static void main(String[] args) {
        try{
            String[] dataFiles = {"L30fft16.out", "L30fft_32.out"};
            int[] gridSizes = {5,6,7};
            int epochs = 10000;
            double learningRate = 1.0;

            for(String dataFile : dataFiles){
                System.out.println("\n========================================");
                System.out.println("Processing: " + dataFile);
                System.out.println("========================================");

                Dataset dataset = new Dataset(dataFile);
                int inputDim = dataset.inputs[0].length;

                for(int gridSize : gridSizes){
                    System.out.println("\n--- Training SOM with grid size: " + gridSize + " ---");
                    SOM som = new SOM(gridSize, inputDim);
                    som.train(dataset.inputs, epochs, learningRate);

                    System.out.println("Training Complete, Generating heatmap...");
                    double[][] heatmap = som.generateHeatMap(dataset.inputs, dataset.labels);
                    String heatmapFile = "heatmap_" + dataFile + "_" + gridSize + "x" + gridSize + ".csv";
                    som.exportHeatMap(heatmap, heatmapFile);
                    System.out.println("Heatmap exported to: " + heatmapFile);
                }

            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
