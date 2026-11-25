package nhom7.diabetes.model;

import java.util.Random;

/**
 * Hồi quy Logistic: p = sigmoid(w·x + b).
 * Huấn luyện bằng mini-batch gradient descent (Adam), có chính quy hóa L2 (tham số C)
 * và tùy chọn class_weight = balanced cho dữ liệu mất cân bằng.
 */
public class LogisticRegressionModel implements Classifier {

    private final double C;
    private final String classWeight;
    private final int epochs;
    private final double learningRate;
    private final int batchSize;
    private final long seed;

    private double[] w;
    private double b;

    public LogisticRegressionModel() {
        this(1.0, "none", 30, 0.01, 256, 42L);
    }

    public LogisticRegressionModel(double C, String classWeight) {
        this(C, classWeight, 30, 0.01, 256, 42L);
    }

    public LogisticRegressionModel(double C, String classWeight, int epochs, double learningRate, int batchSize, long seed) {
        this.C = C;
        this.classWeight = classWeight;
        this.epochs = epochs;
        this.learningRate = learningRate;
        this.batchSize = batchSize;
        this.seed = seed;
    }

    @Override
    public void fit(double[][] X, int[] y) {
        int n = X.length, d = X[0].length;
        w = new double[d];
        b = 0;

        int pos = 0;
        for (int v : y) pos += v;
        double w1 = 1, w0 = 1;
        if ("balanced".equals(classWeight)) {
            w1 = n / (2.0 * pos);
            w0 = n / (2.0 * (n - pos));
        }
        double lambda = 1.0 / (C * n); 

        double[] mW = new double[d], vW = new double[d];
        double mB = 0, vB = 0, beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
        int t = 0;

        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Random rnd = new Random(seed);
        double[] gW = new double[d];

        for (int ep = 0; ep < epochs; ep++) {
            shuffle(order, rnd);
            for (int start = 0; start < n; start += batchSize) {
                int end = Math.min(n, start + batchSize);
                int m = end - start;
                java.util.Arrays.fill(gW, 0);
                double gB = 0;
                for (int k = start; k < end; k++) {
                    int i = order[k];
                    double p = sigmoid(dot(w, X[i]) + b);
                    double cw = y[i] == 1 ? w1 : w0;
                    double err = cw * (p - y[i]);
                    for (int j = 0; j < d; j++) gW[j] += err * X[i][j];
                    gB += err;
                }
                t++;
                double lrT = learningRate * Math.sqrt(1 - Math.pow(beta2, t)) / (1 - Math.pow(beta1, t));
                for (int j = 0; j < d; j++) {
                    double g = gW[j] / m + lambda * w[j];
                    mW[j] = beta1 * mW[j] + (1 - beta1) * g;
                    vW[j] = beta2 * vW[j] + (1 - beta2) * g * g;
                    w[j] -= lrT * mW[j] / (Math.sqrt(vW[j]) + eps);
                }
                double g = gB / m;
                mB = beta1 * mB + (1 - beta1) * g;
                vB = beta2 * vB + (1 - beta2) * g * g;
                b -= lrT * mB / (Math.sqrt(vB) + eps);
            }
        }
    }

    @Override
    public double[] predictProba(double[][] X) {
        double[] p = new double[X.length];
        for (int i = 0; i < X.length; i++) p[i] = sigmoid(dot(w, X[i]) + b);
        return p;
    }

    public double[] getCoefficients() { return w.clone(); }

    public double getIntercept() { return b; }

    @Override
    public String name() { return "Logistic Regression"; }

    @Override
    public String params() { return "C=" + C + ", class_weight=" + classWeight; }

    private static double sigmoid(double z) {
        if (z >= 0) return 1.0 / (1.0 + Math.exp(-z));
        double e = Math.exp(z);
        return e / (1.0 + e);
    }

    private static double dot(double[] a, double[] x) {
        double s = 0;
        for (int j = 0; j < a.length; j++) s += a[j] * x[j];
        return s;
    }

    private static void shuffle(int[] a, Random rnd) {
        for (int i = a.length - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int tmp = a[i]; a[i] = a[j]; a[j] = tmp;
        }
    }
}
