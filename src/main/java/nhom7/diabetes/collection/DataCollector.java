package nhom7.diabetes.collection;

import nhom7.diabetes.config.AppConfig;
import nhom7.diabetes.db.DatabaseManager;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.Statement;
import java.sql.Types;

/**
 * Thu thập dữ liệu: đọc file CSV Diabetes Prediction Dataset (Kaggle),
 * nạp vào bảng raw_data và thống kê mô tả ban đầu.
 */
public class DataCollector {

    private static final String[] EXPECTED_HEADER = {
            "gender", "age", "hypertension", "heart_disease", "smoking_history",
            "bmi", "HbA1c_level", "blood_glucose_level", "diabetes"
    };

    public void run() throws Exception {
        System.out.println("\n================= BƯỚC 1: THU THẬP DỮ LIỆU =================");
        Path csv = Path.of(AppConfig.CSV_PATH);
        if (!Files.exists(csv)) {
            throw new IOException("Không tìm thấy file dữ liệu: " + csv.toAbsolutePath()
                    + "\nHãy tải diabetes_prediction_dataset.csv từ Kaggle và đặt vào thư mục data/");
        }
        DatabaseManager.initSchema();
        int inserted = importCsv(csv);
        System.out.printf("Đã nạp %,d bản ghi từ %s vào bảng raw_data%n", inserted, csv.getFileName());
        describe();
    }

    // Đọc CSV và chèn theo lô (batch) vào bảng raw_data trong 1 transaction
    private int importCsv(Path csv) throws Exception {
        long t0 = System.currentTimeMillis();
        int count = 0, badLines = 0;
        try (Connection c = DatabaseManager.getConnection();
             BufferedReader br = Files.newBufferedReader(csv, StandardCharsets.UTF_8)) {
            c.setAutoCommit(false);
            try (Statement st = c.createStatement()) {
                st.executeUpdate("DELETE FROM raw_data");
            }
            String header = br.readLine();
            if (header == null) throw new IOException("File CSV rỗng");
            String[] cols = header.replace("﻿", "").trim().split(",");
            if (cols.length != EXPECTED_HEADER.length) {
                throw new IOException("Sai số cột. Mong đợi " + EXPECTED_HEADER.length + ", nhận " + cols.length);
            }
            for (int i = 0; i < cols.length; i++) {
                if (!cols[i].trim().equalsIgnoreCase(EXPECTED_HEADER[i])) {
                    throw new IOException("Cột thứ " + (i + 1) + " phải là '" + EXPECTED_HEADER[i] + "' nhưng là '" + cols[i] + "'");
                }
            }

            String sql = "INSERT INTO raw_data(id,gender,age,hypertension,heart_disease,smoking_history,"
                    + "bmi,hba1c_level,blood_glucose_level,diabetes) VALUES(?,?,?,?,?,?,?,?,?,?)";
            try (PreparedStatement ps = c.prepareStatement(sql)) {
                String line;
                while ((line = br.readLine()) != null) {
                    if (line.isBlank()) continue;
                    String[] v = line.split(",", -1);
                    if (v.length != EXPECTED_HEADER.length) { badLines++; continue; }
                    count++;
                    ps.setInt(1, count);
                    setText(ps, 2, v[0]);
                    setNumber(ps, 3, v[1]);
                    setNumber(ps, 4, v[2]);
                    setNumber(ps, 5, v[3]);
                    setText(ps, 6, v[4]);
                    setNumber(ps, 7, v[5]);
                    setNumber(ps, 8, v[6]);
                    setNumber(ps, 9, v[7]);
                    setNumber(ps, 10, v[8]);
                    ps.addBatch();
                    if (count % 5000 == 0) ps.executeBatch();
                }
                ps.executeBatch();
            }
            DatabaseManager.clearLog(c, "1_thu_thap");
            DatabaseManager.log(c, "1_thu_thap", "nguon", "Kaggle - Diabetes Prediction Dataset (iammustafatz)");
            DatabaseManager.log(c, "1_thu_thap", "so_ban_ghi", count);
            DatabaseManager.log(c, "1_thu_thap", "dong_loi_bo_qua", badLines);
            c.commit();
        }
        System.out.printf("Thời gian nạp: %d ms. Dòng sai định dạng bị bỏ qua: %d%n",
                System.currentTimeMillis() - t0, badLines);
        return count;
    }

    private static void setText(PreparedStatement ps, int i, String s) throws Exception {
        s = s.trim();
        if (s.isEmpty()) ps.setNull(i, Types.VARCHAR); else ps.setString(i, s);
    }

    private static void setNumber(PreparedStatement ps, int i, String s) throws Exception {
        s = s.trim();
        if (s.isEmpty() || s.equalsIgnoreCase("nan")) { ps.setNull(i, Types.REAL); return; }
        try {
            ps.setDouble(i, Double.parseDouble(s));
        } catch (NumberFormatException e) {
            ps.setNull(i, Types.REAL); // giá trị không hợp lệ -> coi như thiếu, xử lý ở bước 2
        }
    }

    // Thống kê mô tả dữ liệu thô bằng SQL
    private void describe() throws Exception {
        try (Connection c = DatabaseManager.getConnection(); Statement st = c.createStatement()) {
            System.out.println("\n--- Phân bố nhãn diabetes ---");
            try (ResultSet rs = st.executeQuery(
                    "SELECT diabetes, COUNT(*) n, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM raw_data),2) pct "
                            + "FROM raw_data GROUP BY diabetes ORDER BY diabetes")) {
                while (rs.next()) {
                    String lbl = rs.getInt(1) == 1 ? "1 (Có tiểu đường)" : "0 (Không)";
                    System.out.printf("  %-20s %,8d  (%.2f%%)%n", lbl, rs.getInt(2), rs.getDouble(3));
                    DatabaseManager.log(c, "1_thu_thap", "nhan_" + rs.getInt(1), rs.getInt(2));
                }
            }

            System.out.println("\n--- Thống kê thuộc tính số ---");
            System.out.printf("  %-22s %10s %10s %10s%n", "Thuộc tính", "Min", "Max", "Mean");
            String[] numCols = {"age", "bmi", "hba1c_level", "blood_glucose_level"};
            for (String col : numCols) {
                try (ResultSet rs = st.executeQuery(
                        "SELECT MIN(" + col + "), MAX(" + col + "), AVG(" + col + ") FROM raw_data")) {
                    rs.next();
                    System.out.printf("  %-22s %10.2f %10.2f %10.2f%n", col, rs.getDouble(1), rs.getDouble(2), rs.getDouble(3));
                }
            }

            System.out.println("\n--- Thuộc tính phân loại ---");
            for (String col : new String[]{"gender", "smoking_history", "hypertension", "heart_disease"}) {
                StringBuilder sb = new StringBuilder();
                try (ResultSet rs = st.executeQuery(
                        "SELECT " + col + ", COUNT(*) FROM raw_data GROUP BY " + col + " ORDER BY 2 DESC")) {
                    while (rs.next()) sb.append(rs.getString(1)).append('=').append(String.format("%,d", rs.getInt(2))).append("  ");
                }
                System.out.printf("  %-16s: %s%n", col, sb);
            }

            System.out.println("\n--- 5 bản ghi đầu tiên ---");
            try (ResultSet rs = st.executeQuery("SELECT * FROM raw_data ORDER BY id LIMIT 5")) {
                System.out.printf("  %-4s %-7s %5s %4s %4s %-12s %6s %5s %6s %3s%n",
                        "id", "gender", "age", "HT", "HD", "smoking", "bmi", "HbA1c", "gluc", "y");
                while (rs.next()) {
                    System.out.printf("  %-4d %-7s %5.1f %4d %4d %-12s %6.2f %5.1f %6.0f %3d%n",
                            rs.getInt(1), rs.getString(2), rs.getDouble(3), rs.getInt(4), rs.getInt(5),
                            rs.getString(6), rs.getDouble(7), rs.getDouble(8), rs.getDouble(9), rs.getInt(10));
                }
            }
        }
    }
}
