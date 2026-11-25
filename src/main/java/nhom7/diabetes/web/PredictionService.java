package nhom7.diabetes.web;

import nhom7.diabetes.db.DatabaseManager;
import nhom7.diabetes.model.SavedModel;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Xử lý dự đoán cho website: kiểm tra dữ liệu nhập, mã hóa thành 13 đặc trưng,
 * dự đoán bằng mô hình đã lưu và lưu kết quả vào bảng prediction_history.
 */
public class PredictionService {

    // 6 nhóm tiền sử hút thuốc, đúng thứ tự cột one-hot trong DataPreprocessor
    private static final String[] SMOKING_CATS = {"never", "No Info", "current", "former", "ever", "not current"};

    // Dưới mức này là nguy cơ thấp
    private static final double LOW_RISK_LIMIT = 0.15;

    private final SavedModel saved;

    public PredictionService(SavedModel saved) {
        this.saved = saved;
    }

    public SavedModel getSavedModel() { return saved; }

    // Kết quả trả về cho giao diện
    public static class Result {
        public double probability;                      // xác suất có nguy cơ (0..1)
        public boolean atRisk;                          // xác suất >= ngưỡng
        public String level;                            // "thap" | "theo_doi" | "cao"
        public String message;                          // câu kết luận
        public List<String[]> indicators = new ArrayList<>(); // {tên chỉ số, giá trị, đánh giá, mức: ok|warn|bad}
        public List<String> warnings = new ArrayList<>();      // cảnh báo về dữ liệu nhập
    }

    // Lỗi do người dùng nhập sai -> trả về HTTP 400 kèm thông báo
    public static class InputException extends Exception {
        public InputException(String msg) { super(msg); }
    }


    public Result predict(Map<String, String> form) throws Exception {
        // ---------- 1. Kiểm tra dữ liệu nhập ----------
        String gender = required(form, "gender", "Giới tính");
        if (!gender.equals("Male") && !gender.equals("Female")) throw new InputException("Giới tính không hợp lệ.");
        double age = number(form, "age", "Tuổi", 1, 120);
        int hypertension = yesNo(form, "hypertension", "Tăng huyết áp");
        int heartDisease = yesNo(form, "heart_disease", "Bệnh tim");
        String smoking = required(form, "smoking_history", "Tiền sử hút thuốc");
        int smokingIdx = indexOf(SMOKING_CATS, smoking);
        if (smokingIdx < 0) throw new InputException("Tiền sử hút thuốc không hợp lệ.");
        double bmi = number(form, "bmi", "BMI", 10, 100);
        double hba1c = number(form, "hba1c_level", "HbA1c", 3, 20);
        double glucose = number(form, "blood_glucose_level", "Đường huyết", 40, 600);

        // ---------- 2. Mã hóa thành vector 13 đặc trưng (chưa chuẩn hóa) ----------
        double[] x = new double[13];
        x[0] = gender.equals("Male") ? 1 : 0;
        x[1] = age;
        x[2] = hypertension;
        x[3] = heartDisease;
        x[4] = bmi;
        x[5] = hba1c;
        x[6] = glucose;
        x[7 + smokingIdx] = 1;   // one-hot: chỉ 1 trong 6 cột smoking bằng 1

        // ---------- 3 + 4. Dự đoán & kết luận ----------
        Result r = new Result();
        r.probability = saved.predictProba(x);
        r.atRisk = r.probability >= saved.threshold;
        if (r.atRisk) {
            r.level = "cao";
            r.message = "Mô hình đánh giá bạn CÓ NGUY CƠ mắc bệnh tiểu đường. Bạn nên đến cơ sở y tế để được xét nghiệm và tư vấn.";
        } else if (r.probability >= LOW_RISK_LIMIT) {
            r.level = "theo_doi";
            r.message = "Nguy cơ chưa vượt ngưỡng nhưng không thấp. Bạn nên duy trì lối sống lành mạnh và kiểm tra sức khỏe định kỳ.";
        } else {
            r.level = "thap";
            r.message = "Mô hình đánh giá nguy cơ mắc bệnh tiểu đường của bạn THẤP.";
        }

        // Đánh giá tham khảo từng chỉ số theo ngưỡng y khoa
        r.indicators.add(rateHba1c(hba1c));
        r.indicators.add(rateGlucose(glucose));
        r.indicators.add(rateBmi(bmi));

        // Cảnh báo khi giá trị nằm ngoài khoảng dữ liệu huấn luyện
        if (age > 80) r.warnings.add("Tuổi > 80 nằm ngoài khoảng dữ liệu huấn luyện (0 - 80), kết quả có thể kém chính xác.");
        if (hba1c < 3.5 || hba1c > 9) r.warnings.add("HbA1c nằm ngoài khoảng dữ liệu huấn luyện (3.5 - 9.0%).");
        if (glucose < 80 || glucose > 300) r.warnings.add("Đường huyết nằm ngoài khoảng dữ liệu huấn luyện (80 - 300 mg/dL).");

        // ---------- 5. Lưu lịch sử ----------
        saveHistory(gender, age, hypertension, heartDisease, smoking, bmi, hba1c, glucose, r);
        return r;
    }

    // ---------- Đánh giá tham khảo từng chỉ số ----------

    private static String[] rateHba1c(double v) {
        String val = String.format("%.1f %%", v);
        if (v >= 6.5) return new String[]{"HbA1c", val, "≥ 6.5%: ngưỡng chẩn đoán tiểu đường", "bad"};
        if (v >= 5.7) return new String[]{"HbA1c", val, "5.7 - 6.4%: tiền tiểu đường", "warn"};
        return new String[]{"HbA1c", val, "< 5.7%: bình thường", "ok"};
    }

    private static String[] rateGlucose(double v) {
        String val = String.format("%.0f mg/dL", v);
        if (v >= 200) return new String[]{"Đường huyết", val, "≥ 200 mg/dL: rất cao", "bad"};
        if (v >= 140) return new String[]{"Đường huyết", val, "140 - 199 mg/dL: cao hơn bình thường", "warn"};
        return new String[]{"Đường huyết", val, "< 140 mg/dL: bình thường", "ok"};
    }

    private static String[] rateBmi(double v) {
        String val = String.format("%.1f", v);
        if (v >= 30) return new String[]{"BMI", val, "≥ 30: béo phì", "bad"};
        if (v >= 25) return new String[]{"BMI", val, "25 - 29.9: thừa cân", "warn"};
        if (v < 18.5) return new String[]{"BMI", val, "< 18.5: thiếu cân", "warn"};
        return new String[]{"BMI", val, "18.5 - 24.9: bình thường", "ok"};
    }

    // ---------- Lịch sử dự đoán (CSDL) ----------

    public static void createHistoryTable() throws Exception {
        try (Connection c = DatabaseManager.getConnection(); Statement st = c.createStatement()) {
            st.executeUpdate("""
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at          TEXT DEFAULT (datetime('now','localtime')),
                    gender              TEXT,
                    age                 REAL,
                    hypertension        INTEGER,
                    heart_disease       INTEGER,
                    smoking_history     TEXT,
                    bmi                 REAL,
                    hba1c_level         REAL,
                    blood_glucose_level REAL,
                    probability         REAL,
                    at_risk             INTEGER
                )""");
        }
    }

    private void saveHistory(String gender, double age, int ht, int hd, String smoking,
                             double bmi, double hba1c, double glucose, Result r) throws Exception {
        String sql = "INSERT INTO prediction_history(gender,age,hypertension,heart_disease,smoking_history,"
                + "bmi,hba1c_level,blood_glucose_level,probability,at_risk) VALUES(?,?,?,?,?,?,?,?,?,?)";
        try (Connection c = DatabaseManager.getConnection(); PreparedStatement ps = c.prepareStatement(sql)) {
            ps.setString(1, gender);
            ps.setDouble(2, age);
            ps.setInt(3, ht);
            ps.setInt(4, hd);
            ps.setString(5, smoking);
            ps.setDouble(6, bmi);
            ps.setDouble(7, hba1c);
            ps.setDouble(8, glucose);
            ps.setDouble(9, r.probability);
            ps.setInt(10, r.atRisk ? 1 : 0);
            ps.executeUpdate();
        }
    }

    // Lấy n lần dự đoán gần nhất (mới nhất trước)
    public List<String[]> recentHistory(int n) throws Exception {
        List<String[]> rows = new ArrayList<>();
        String sql = "SELECT created_at, gender, age, hypertension, heart_disease, smoking_history, bmi, "
                + "hba1c_level, blood_glucose_level, probability, at_risk FROM prediction_history ORDER BY id DESC LIMIT ?";
        try (Connection c = DatabaseManager.getConnection(); PreparedStatement ps = c.prepareStatement(sql)) {
            ps.setInt(1, n);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    rows.add(new String[]{
                            rs.getString(1),
                            rs.getString(2).equals("Male") ? "Nam" : "Nữ",
                            String.format("%.0f", rs.getDouble(3)),
                            rs.getInt(4) == 1 ? "Có" : "Không",
                            rs.getInt(5) == 1 ? "Có" : "Không",
                            rs.getString(6),
                            String.format("%.1f", rs.getDouble(7)),
                            String.format("%.1f", rs.getDouble(8)),
                            String.format("%.0f", rs.getDouble(9)),
                            String.format("%.1f%%", rs.getDouble(10) * 100),
                            rs.getInt(11) == 1 ? "Có nguy cơ" : "Không"
                    });
                }
            }
        }
        return rows;
    }

    // ---------- Hàm kiểm tra dữ liệu nhập ----------

    private static String required(Map<String, String> f, String key, String label) throws InputException {
        String v = f.get(key);
        if (v == null || v.isBlank()) throw new InputException("Vui lòng nhập: " + label);
        return v.trim();
    }

    private static double number(Map<String, String> f, String key, String label, double min, double max) throws InputException {
        String s = required(f, key, label).replace(',', '.');   // cho phép nhập dấu phẩy thập phân
        double v;
        try {
            v = Double.parseDouble(s);
        } catch (NumberFormatException e) {
            throw new InputException(label + " phải là số.");
        }
        if (v < min || v > max) throw new InputException(label + " phải nằm trong khoảng " + fmt(min) + " - " + fmt(max) + ".");
        return v;
    }

    private static int yesNo(Map<String, String> f, String key, String label) throws InputException {
        String v = required(f, key, label);
        if (v.equals("1")) return 1;
        if (v.equals("0")) return 0;
        throw new InputException(label + " không hợp lệ.");
    }

    private static int indexOf(String[] arr, String v) {
        for (int i = 0; i < arr.length; i++) if (arr[i].equals(v)) return i;
        return -1;
    }

    private static String fmt(double v) {
        return v == Math.floor(v) ? String.valueOf((long) v) : String.valueOf(v);
    }
}
