package nhom7.diabetes.preprocessing;

import nhom7.diabetes.db.DatabaseManager;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.Statement;

/**
 * Chuẩn hóa Z-score: x' = (x - mean) / std cho các cột liên tục (age, bmi, HbA1c, đường huyết).
 * mean/std chỉ tính trên tập train để tránh rò rỉ dữ liệu.
 */
public class StandardScaler implements java.io.Serializable {

    private final int[] columns;
    private double[] mean;
    private double[] std;

    public StandardScaler(int[] columns) {
        this.columns = columns.clone();
    }

    public StandardScaler fit(double[][] X) {
        int n = X.length;
        mean = new double[columns.length];
        std = new double[columns.length];
        for (int j = 0; j < columns.length; j++) {
            int col = columns[j];
            double s = 0;
            for (double[] row : X) s += row[col];
            double m = s / n;
            double v = 0;
            for (double[] row : X) v += (row[col] - m) * (row[col] - m);
            double sd = Math.sqrt(v / n);
            mean[j] = m;
            std[j] = sd < 1e-12 ? 1.0 : sd;
        }
        return this;
    }

    // Trả về bản sao đã chuẩn hóa (không sửa dữ liệu gốc)
    public double[][] transform(double[][] X) {
        double[][] out = new double[X.length][];
        for (int i = 0; i < X.length; i++) {
            double[] r = X[i].clone();
            for (int j = 0; j < columns.length; j++) {
                r[columns[j]] = (r[columns[j]] - mean[j]) / std[j];
            }
            out[i] = r;
        }
        return out;
    }

    public double[] getMean() { return mean.clone(); }

    public double[] getStd() { return std.clone(); }

    // Lưu mean/std vào bảng scaler_params
    public void saveToDb(String[] featureNames) throws Exception {
        try (Connection c = DatabaseManager.getConnection()) {
            c.setAutoCommit(false);
            try (Statement st = c.createStatement()) { st.executeUpdate("DELETE FROM scaler_params"); }
            try (PreparedStatement ps = c.prepareStatement("INSERT INTO scaler_params(feature,mean,std) VALUES(?,?,?)")) {
                for (int j = 0; j < columns.length; j++) {
                    ps.setString(1, featureNames[columns[j]]);
                    ps.setDouble(2, mean[j]);
                    ps.setDouble(3, std[j]);
                    ps.addBatch();
                }
                ps.executeBatch();
            }
            c.commit();
        }
    }

    // Đọc lại scaler đã lưu (dùng khi triển khai web / dự đoán mẫu mới)
    public static StandardScaler loadFromDb(int[] columns, String[] featureNames) throws Exception {
        StandardScaler s = new StandardScaler(columns);
        s.mean = new double[columns.length];
        s.std = new double[columns.length];
        try (Connection c = DatabaseManager.getConnection();
             PreparedStatement ps = c.prepareStatement("SELECT mean,std FROM scaler_params WHERE feature=?")) {
            for (int j = 0; j < columns.length; j++) {
                ps.setString(1, featureNames[columns[j]]);
                try (ResultSet rs = ps.executeQuery()) {
                    if (!rs.next()) throw new IllegalStateException("Chưa có tham số chuẩn hóa cho " + featureNames[columns[j]]);
                    s.mean[j] = rs.getDouble(1);
                    s.std[j] = rs.getDouble(2);
                }
            }
        }
        return s;
    }
}
