package nhom7.diabetes.model;

import java.util.stream.IntStream;

/**
 * K láng giềng gần nhất (KNN): khoảng cách Euclidean hoặc Manhattan,
 * trọng số đều (uniform) hoặc theo khoảng cách (distance).
 */
public class KNNModel implements Classifier {

    private final int k;
    private final String weights;
    private final String metric;

    private double[][] Xtrain;
    private int[] ytrain;

    public KNNModel() {
        this(5, "uniform", "euclidean");
    }

    public KNNModel(int k, String weights, String metric) {
        this.k = k;
        this.weights = weights;
        this.metric = metric;
    }

    @Override
    public void fit(double[][] X, int[] y) {
        this.Xtrain = X;
        this.ytrain = y;
    }

    @Override
    public double[] predictProba(double[][] X) {
        Neighbors nb = findNeighbors(Xtrain, X, k, metric);
        return probaFromNeighbors(nb, ytrain, k, weights);
    }

    public static final class Neighbors {
        public final int[][] index;
        public final double[][] dist;

        Neighbors(int[][] index, double[][] dist) {
            this.index = index;
            this.dist = dist;
        }
    }


    public static Neighbors findNeighbors(double[][] train, double[][] query, int maxK, String metric) {
        int m = query.length;
        int[][] idx = new int[m][];
        double[][] dst = new double[m][];
        boolean manhattan = "manhattan".equals(metric);
        int d = train[0].length;

        IntStream.range(0, m).parallel().forEach(q -> {
            double[] x = query[q];
            int[] bi = new int[maxK];
            double[] bd = new double[maxK];
            java.util.Arrays.fill(bd, Double.MAX_VALUE);
            double worst = Double.MAX_VALUE;
            for (int i = 0; i < train.length; i++) {
                double[] t = train[i];
                double s = 0;
                if (manhattan) {
                    for (int j = 0; j < d && s < worst; j++) s += Math.abs(x[j] - t[j]);
                } else {
                    for (int j = 0; j < d && s < worst; j++) { double df = x[j] - t[j]; s += df * df; }
                }
                if (s < worst) {
                    int p = maxK - 1; 
                    while (p > 0 && bd[p - 1] > s) { bd[p] = bd[p - 1]; bi[p] = bi[p - 1]; p--; }
                    bd[p] = s;
                    bi[p] = i;
                    worst = bd[maxK - 1];
                }
            }
            if (!manhattan) for (int j = 0; j < maxK; j++) bd[j] = Math.sqrt(bd[j]);
            idx[q] = bi;
            dst[q] = bd;
        });
        return new Neighbors(idx, dst);
    }


    public static double[] probaFromNeighbors(Neighbors nb, int[] ytrain, int k, String weights) {
        int m = nb.index.length;
        double[] p = new double[m];
        boolean byDistance = "distance".equals(weights);
        for (int q = 0; q < m; q++) {
            int[] bi = nb.index[q];
            double[] bd = nb.dist[q];
            if (byDistance) {
                int zero = 0, zeroPos = 0;
                for (int j = 0; j < k; j++) if (bd[j] == 0) { zero++; zeroPos += ytrain[bi[j]]; }
                if (zero > 0) { p[q] = (double) zeroPos / zero; continue; }
                double sw = 0, sp = 0;
                for (int j = 0; j < k; j++) {
                    double w = 1.0 / bd[j];
                    sw += w;
                    sp += w * ytrain[bi[j]];
                }
                p[q] = sp / sw;
            } else {
                int s = 0;
                for (int j = 0; j < k; j++) s += ytrain[bi[j]];
                p[q] = (double) s / k;
            }
        }
        return p;
    }

    @Override
    public String name() { return "KNN"; }

    @Override
    public String params() { return "n_neighbors=" + k + ", weights=" + weights + ", metric=" + metric; }
}
