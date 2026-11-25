package nhom7.diabetes.model;

import java.util.Arrays;

/**
 * Chia giá trị mỗi đặc trưng thành các khoảng (bin) theo phân vị.
 * Dùng cho Decision Tree và Random Forest để tìm điểm chia nút nhanh hơn.
 */
public class FeatureBinner {

    private final double[][] thresholds;

    private FeatureBinner(double[][] thresholds) {
        this.thresholds = thresholds;
    }

    public static FeatureBinner fit(double[][] X, int maxBins) {
        int n = X.length, d = X[0].length;
        double[][] th = new double[d][];
        double[] col = new double[n];
        for (int f = 0; f < d; f++) {
            for (int i = 0; i < n; i++) col[i] = X[i][f];
            Arrays.sort(col);
            double[] uniq = new double[n];
            int u = 0;
            for (int i = 0; i < n; i++) if (i == 0 || col[i] != col[i - 1]) uniq[u++] = col[i];
            if (u <= maxBins) {
                th[f] = new double[Math.max(0, u - 1)];
                for (int i = 0; i < u - 1; i++) th[f][i] = (uniq[i] + uniq[i + 1]) / 2.0;
            } else {
                double[] tmp = new double[maxBins - 1];
                int k = 0;
                for (int q = 1; q < maxBins; q++) {
                    double v = col[(int) ((long) q * n / maxBins)];
                    if (v >= col[n - 1]) break;
                    if (k == 0 || v > tmp[k - 1]) tmp[k++] = v;
                }
                th[f] = Arrays.copyOf(tmp, k);
            }
        }
        return new FeatureBinner(th);
    }

    public int numBins(int f) { return thresholds[f].length + 1; }

    public double threshold(int f, int b) { return thresholds[f][b]; }

    public int bin(int f, double v) {
        double[] t = thresholds[f];
        int lo = 0, hi = t.length; 
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (v <= t[mid]) hi = mid; else lo = mid + 1;
        }
        return lo;
    }

    public short[][] transform(double[][] X) {
        int n = X.length, d = thresholds.length;
        short[][] out = new short[d][n];
        for (int f = 0; f < d; f++)
            for (int i = 0; i < n; i++) out[f][i] = (short) bin(f, X[i][f]);
        return out;
    }
}
