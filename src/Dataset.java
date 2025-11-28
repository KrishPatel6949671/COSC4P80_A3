import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * Loads data from file (mostly reused from A2)
 */
public class Dataset {
    public double[][] inputs;
    public int[] labels;

    public Dataset(String path) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String[] header = br.readLine().trim().split("\\s+");
            int numSamples = Integer.parseInt(header[0]);
            int numBins = Integer.parseInt(header[1]);

            inputs = new double[numSamples][numBins];
            labels = new int[numSamples];

            for (int i = 0; i < numSamples; i++) {
                String[] line = br.readLine().trim().split("\\s+");
                int label = Integer.parseInt(line[0]);
                labels[i] = label;

                for (int j = 0; j < numBins; j++) {
                    inputs[i][j] = Double.parseDouble(line[j + 1]);
                }
            }
        }
        //normalizeVectors(inputs);
    }

    public void normalizeVectors(double[][] vectors) {
        for (int i = 0; i < vectors.length; i++) {
            double norm = 0.0;
            for (double val : vectors[i]) {
                norm += val * val;
            }
            norm = Math.sqrt(norm);
            for (int j = 0; j < vectors[i].length; j++) {
                vectors[i][j] /= norm;
            }
        }
    }
}
