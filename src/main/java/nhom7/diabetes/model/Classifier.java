package nhom7.diabetes.model;

import nhom7.diabetes.config.AppConfig;


/**
 * Giao diện chung cho 4 mô hình: fit (huấn luyện), predictProba (dự đoán xác suất), predict (dự đoán nhãn).
 */
public interface Classifier extends java.io.Serializable {

    void fit(double[][] X, int[] y);

    double[] predictProba(double[][] X);

    default int[] predict(double[][] X) {
        double[] p = predictProba(X);
        int[] out = new int[p.length];
        for (int i = 0; i < p.length; i++) out[i] = p[i] >= AppConfig.THRESHOLD ? 1 : 0;
        return out;
    }

    String name();

    String params();

    default double[] featureImportance() { return null; }
}
