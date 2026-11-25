package nhom7.diabetes.db;

import nhom7.diabetes.config.AppConfig;

import java.io.File;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;

/**
 * Quản lý kết nối CSDL SQLite và tạo các bảng:
 * raw_data, processed_data, scaler_params, data_log, cv_results, model_results, feature_importance.
 */
public final class DatabaseManager {

    private DatabaseManager() {}

    public static Connection getConnection() throws SQLException {
        if (AppConfig.DB_URL.startsWith("jdbc:sqlite:")) {
            String path = AppConfig.DB_URL.substring("jdbc:sqlite:".length());
            File parent = new File(path).getAbsoluteFile().getParentFile();
            if (parent != null && !parent.exists()) parent.mkdirs();
        }
        return DriverManager.getConnection(AppConfig.DB_URL, AppConfig.DB_USER, AppConfig.DB_PASSWORD);
    }

    // Tạo các bảng nếu chưa tồn tại
    public static void initSchema() throws SQLException {
        try (Connection c = getConnection(); Statement st = c.createStatement()) {
            st.executeUpdate("""
                CREATE TABLE IF NOT EXISTS raw_data (
                    id                  INTEGER PRIMARY KEY,
                    gender              TEXT,
                    age                 REAL,
                    hypertension        INTEGER,
                    heart_disease       INTEGER,
                    smoking_history     TEXT,
                    bmi                 REAL,
                    hba1c_level         REAL,
                    blood_glucose_level REAL,
                    diabetes            INTEGER
                )""");
            st.executeUpdate("""
                CREATE TABLE IF NOT EXISTS processed_data (
                    id                  INTEGER PRIMARY KEY,
                    raw_id              INTEGER,
                    split               TEXT,
                    gender_male         REAL,
                    age                 REAL,
                    hypertension        REAL,
                    heart_disease       REAL,
                    bmi                 REAL,
                    hba1c_level         REAL,
                    blood_glucose_level REAL,
                    smoking_never       REAL,
                    smoking_no_info     REAL,
                    smoking_current     REAL,
                    smoking_former      REAL,
                    smoking_ever        REAL,
                    smoking_not_current REAL,
                    diabetes            INTEGER
                )""");
            st.executeUpdate("""
                CREATE TABLE IF NOT EXISTS scaler_params (
                    feature  TEXT PRIMARY KEY,
                    mean     REAL,
                    std      REAL
                )""");
            st.executeUpdate("""
                CREATE TABLE IF NOT EXISTS data_log (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    step       TEXT,
                    item       TEXT,
                    value      TEXT,
                    created_at TEXT DEFAULT (datetime('now','localtime'))
                )""");
            st.executeUpdate("""
                CREATE TABLE IF NOT EXISTS cv_results (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    model       TEXT,
                    search_type TEXT,
                    params      TEXT,
                    k_folds     INTEGER,
                    acc_mean    REAL,
                    prec_mean   REAL,
                    rec_mean    REAL,
                    f1_mean     REAL,
                    f1_std      REAL,
                    auc_mean    REAL,
                    time_ms     INTEGER,
                    is_best     INTEGER DEFAULT 0,
                    created_at  TEXT DEFAULT (datetime('now','localtime'))
                )""");
            st.executeUpdate("""
                CREATE TABLE IF NOT EXISTS model_results (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    stage       TEXT,
                    model       TEXT,
                    params      TEXT,
                    accuracy    REAL,
                    precision_  REAL,
                    recall      REAL,
                    f1          REAL,
                    roc_auc     REAL,
                    tn INTEGER, fp INTEGER, fn INTEGER, tp INTEGER,
                    train_ms    INTEGER,
                    created_at  TEXT DEFAULT (datetime('now','localtime'))
                )""");
            st.executeUpdate("""
                CREATE TABLE IF NOT EXISTS feature_importance (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    stage      TEXT,
                    model      TEXT,
                    feature    TEXT,
                    importance REAL
                )""");
        }
    }

    // Ghi một dòng nhật ký thống kê
    public static void log(Connection c, String step, String item, Object value) throws SQLException {
        try (var ps = c.prepareStatement("INSERT INTO data_log(step,item,value) VALUES(?,?,?)")) {
            ps.setString(1, step);
            ps.setString(2, item);
            ps.setString(3, String.valueOf(value));
            ps.executeUpdate();
        }
    }

    public static void clearLog(Connection c, String step) throws SQLException {
        try (var ps = c.prepareStatement("DELETE FROM data_log WHERE step=?")) {
            ps.setString(1, step);
            ps.executeUpdate();
        }
    }
}
