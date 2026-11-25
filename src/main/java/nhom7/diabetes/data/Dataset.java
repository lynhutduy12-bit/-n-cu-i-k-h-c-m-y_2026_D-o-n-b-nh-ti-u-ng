package nhom7.diabetes.data;

/**
 * Cấu trúc dữ liệu dùng chung cho các mô hình:
 * ma trận đặc trưng X (n mẫu x d đặc trưng) và vector nhãn y (0/1).
 */
public class Dataset {

    public final double[][] X;
    public final int[] y;
    public final String[] featureNames;

    public Dataset(double[][] X, int[] y, String[] featureNames) {
        this.X = X;
        this.y = y;
        this.featureNames = featureNames;
    }

    public int size() { return y.length; }

    public int numFeatures() { return featureNames.length; }

    public int countPositive() {
        int c = 0;
        for (int v : y) c += v;
        return c;
    }

    // Lấy tập con theo danh sách chỉ số (dùng khi chia fold)
    public Dataset subset(int[] idx) {
        double[][] sx = new double[idx.length][];
        int[] sy = new int[idx.length];
        for (int i = 0; i < idx.length; i++) {
            sx[i] = X[idx[i]];
            sy[i] = y[idx[i]];
        }
        return new Dataset(sx, sy, featureNames);
    }
}
