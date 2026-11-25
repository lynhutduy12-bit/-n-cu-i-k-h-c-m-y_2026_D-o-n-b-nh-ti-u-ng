package nhom7.diabetes.model;

import nhom7.diabetes.preprocessing.StandardScaler;

import java.io.*;

/**
 * Lưu / nạp mô hình tốt nhất (kèm bộ chuẩn hóa và ngưỡng) ra file output/best_model.ser
 * để website dùng lại mà không cần huấn luyện lại.
 */
public class SavedModel implements Serializable {

    private static final long serialVersionUID = 1L;

    public final Classifier model;
    public final StandardScaler scaler;
    public final String[] featureNames;
    public final double threshold;

    public SavedModel(Classifier model, StandardScaler scaler, String[] featureNames, double threshold) {
        this.model = model;
        this.scaler = scaler;
        this.featureNames = featureNames;
        this.threshold = threshold;
    }

    // Dự đoán xác suất cho 1 người (vector đặc trưng CHƯA chuẩn hóa, đúng thứ tự featureNames)
    public double predictProba(double[] rawFeatures) {
        double[][] x = scaler.transform(new double[][]{rawFeatures});
        return model.predictProba(x)[0];
    }

    public void save(String path) throws IOException {
        try (ObjectOutputStream out = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(path)))) {
            out.writeObject(this);
        }
    }

    public static SavedModel load(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream in = new ObjectInputStream(new BufferedInputStream(new FileInputStream(path)))) {
            return (SavedModel) in.readObject();
        }
    }
}
