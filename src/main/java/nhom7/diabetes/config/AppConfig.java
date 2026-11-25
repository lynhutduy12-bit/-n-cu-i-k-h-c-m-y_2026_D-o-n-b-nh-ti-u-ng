package nhom7.diabetes.config;

/**
 * Cấu hình chung: đường dẫn dữ liệu, CSDL, thư mục output, seed, tỷ lệ chia train/test, số fold, ngưỡng.
 */
public final class AppConfig {

    private AppConfig() {}

    // Đường dẫn file CSV tải từ Kaggle (Diabetes Prediction Dataset - iammustafatz)
    public static final String CSV_PATH =
            System.getProperty("csv.path", "data/diabetes_prediction_dataset.csv");

    // Chuỗi kết nối CSDL
    public static final String DB_URL =
            System.getProperty("db.url", "jdbc:sqlite:data/diabetes.db");
    public static final String DB_USER = System.getProperty("db.user", "");
    public static final String DB_PASSWORD = System.getProperty("db.password", "");

    // Thư mục xuất kết quả (CSV, hình)
    public static final String OUTPUT_DIR = System.getProperty("output.dir", "output");

    // Seed cố định để kết quả lặp lại được giữa các lần chạy
    public static final long SEED = 42L;

    // Tỷ lệ tập kiểm tra (test) khi chia dữ liệu
    public static final double TEST_RATIO = 0.2;

    // Số fold cho K-Fold Cross-Validation
    public static final int K_FOLDS = Integer.getInteger("cv.folds", 5);

    // Ngưỡng xác suất để dự đoán nhãn 1 (có nguy cơ tiểu đường)
    public static final double THRESHOLD = 0.5;
}
