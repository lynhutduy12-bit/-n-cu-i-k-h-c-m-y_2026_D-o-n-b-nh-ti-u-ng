package nhom7.diabetes.preprocessing;

import nhom7.diabetes.config.AppConfig;
import nhom7.diabetes.data.Dataset;
import nhom7.diabetes.db.DatabaseManager;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.*;

/**
 * Tiền xử lý dữ liệu:
 *   - Xử lý dữ liệu thiếu, giá trị bất thường, bản ghi trùng lặp, loại gender = "Other"
 *   - Mã hóa: gender -> 0/1, smoking_history -> One-Hot 6 cột
 *   - Chia train/test 80/20 có phân tầng, lưu vào bảng processed_data
 */
public class DataPreprocessor {

    public static final String[] FEATURE_NAMES = {
            "gender_male", "age", "hypertension", "heart_disease", "bmi",
            "hba1c_level", "blood_glucose_level",
            "smoking_never", "smoking_no_info", "smoking_current",
            "smoking_former", "smoking_ever", "smoking_not_current"
    };

    // Chỉ số các cột liên tục cần chuẩn hóa
    public static final int[] CONTINUOUS_COLS = {1, 4, 5, 6};

    private static final String[] SMOKING_CATS = {"never", "No Info", "current", "former", "ever", "not current"};

    // Bản ghi thô đọc từ bảng raw_data
    private static class RawRow {
        int id;
        String gender, smoking;
        Double age, bmi, hba1c, glucose;
        Integer hypertension, heartDisease, diabetes;

        String key() {
            return gender + "|" + age + "|" + hypertension + "|" + heartDisease + "|" + smoking + "|"
                    + bmi + "|" + hba1c + "|" + glucose + "|" + diabetes;
        }
    }

    public void run() throws Exception {
        System.out.println("\n================= BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU =================");
        DatabaseManager.initSchema();
        List<RawRow> rows = loadRaw();
        if (rows.isEmpty()) throw new IllegalStateException("Bảng raw_data trống. Hãy chạy Bước 1 (Thu thập) trước.");
        System.out.printf("Đọc %,d bản ghi từ bảng raw_data%n", rows.size());

        try (Connection c = DatabaseManager.getConnection()) {
            DatabaseManager.clearLog(c, "2_tien_xu_ly");
            DatabaseManager.log(c, "2_tien_xu_ly", "so_ban_ghi_dau_vao", rows.size());

            // ---------- 2.1 Dữ liệu thiếu & bất thường ----------
            System.out.println("\n[2.1] Kiểm tra dữ liệu thiếu (NULL) theo từng cột:");
            String[] cols = {"gender", "age", "hypertension", "heart_disease", "smoking_history",
                    "bmi", "hba1c_level", "blood_glucose_level", "diabetes"};
            int totalMissing = 0;
            try (Statement st = c.createStatement()) {
                for (String col : cols) {
                    try (ResultSet rs = st.executeQuery("SELECT COUNT(*) FROM raw_data WHERE " + col + " IS NULL")) {
                        rs.next();
                        int m = rs.getInt(1);
                        totalMissing += m;
                        System.out.printf("   %-22s : %d%n", col, m);
                        DatabaseManager.log(c, "2_tien_xu_ly", "thieu_" + col, m);
                    }
                }
                try (ResultSet rs = st.executeQuery("SELECT COUNT(*) FROM raw_data WHERE smoking_history='No Info'")) {
                    rs.next();
                    System.out.printf("   Ghi chú: smoking_history = 'No Info' có %,d bản ghi -> giữ lại như 1 nhóm riêng%n", rs.getInt(1));
                }
            }
            int invalid = markInvalidAsMissing(rows);
            System.out.printf("   Tổng ô thiếu: %d | Giá trị bất thường (coi như thiếu): %d%n", totalMissing, invalid);
            int droppedNoLabel = fillMissing(rows);
            System.out.printf("   Đã điền giá trị thiếu (median/mode). Loại %d bản ghi thiếu nhãn.%n", droppedNoLabel);
            DatabaseManager.log(c, "2_tien_xu_ly", "gia_tri_bat_thuong", invalid);
            DatabaseManager.log(c, "2_tien_xu_ly", "loai_do_thieu_nhan", droppedNoLabel);

            // ---------- 2.2 Trùng lặp ----------
            int before = rows.size();
            Set<String> seen = new HashSet<>();
            List<RawRow> dedup = new ArrayList<>(rows.size());
            for (RawRow r : rows) if (seen.add(r.key())) dedup.add(r);
            rows = dedup;
            int dup = before - rows.size();
            System.out.printf("%n[2.2] Bản ghi trùng lặp: %,d -> đã loại bỏ. Còn lại %,d bản ghi%n", dup, rows.size());
            DatabaseManager.log(c, "2_tien_xu_ly", "ban_ghi_trung_lap", dup);

            // ---------- 2.3 gender = Other ----------
            before = rows.size();
            rows.removeIf(r -> !"Male".equals(r.gender) && !"Female".equals(r.gender));
            int other = before - rows.size();
            System.out.printf("%n[2.3] Loại %d bản ghi gender = 'Other'. Còn lại %,d bản ghi%n", other, rows.size());
            DatabaseManager.log(c, "2_tien_xu_ly", "loai_gender_other", other);

            // ---------- 2.4 Mã hóa ----------
            double[][] X = new double[rows.size()][];
            int[] y = new int[rows.size()];
            int[] rawIds = new int[rows.size()];
            for (int i = 0; i < rows.size(); i++) {
                X[i] = encode(rows.get(i));
                y[i] = rows.get(i).diabetes;
                rawIds[i] = rows.get(i).id;
            }
            System.out.println("\n[2.4] Mã hóa biến phân loại:");
            System.out.println("   gender          -> gender_male (Male=1, Female=0)");
            System.out.println("   smoking_history -> One-Hot: smoking_never, smoking_no_info, smoking_current,");
            System.out.println("                      smoking_former, smoking_ever, smoking_not_current");
            System.out.printf("   Số đặc trưng sau mã hóa: %d%n", FEATURE_NAMES.length);
            DatabaseManager.log(c, "2_tien_xu_ly", "so_dac_trung", FEATURE_NAMES.length);

            // ---------- 2.5 Chia train/test có phân tầng ----------
            boolean[] isTest = stratifiedSplit(y, AppConfig.TEST_RATIO, AppConfig.SEED);
            int nTest = 0, posTest = 0, posTrain = 0;
            for (int i = 0; i < y.length; i++) {
                if (isTest[i]) { nTest++; posTest += y[i]; } else posTrain += y[i];
            }
            int nTrain = y.length - nTest;
            System.out.printf("%n[2.5] Chia dữ liệu (stratified, seed=%d):%n", AppConfig.SEED);
            System.out.printf("   Train: %,6d mẫu | dương tính %,5d (%.2f%%)%n", nTrain, posTrain, 100.0 * posTrain / nTrain);
            System.out.printf("   Test : %,6d mẫu | dương tính %,5d (%.2f%%)%n", nTest, posTest, 100.0 * posTest / nTest);
            DatabaseManager.log(c, "2_tien_xu_ly", "so_mau_train", nTrain);
            DatabaseManager.log(c, "2_tien_xu_ly", "so_mau_test", nTest);
            DatabaseManager.log(c, "2_tien_xu_ly", "duong_tinh_train", posTrain);
            DatabaseManager.log(c, "2_tien_xu_ly", "duong_tinh_test", posTest);

            // ---------- 2.6 Chuẩn hóa (fit trên train) ----------
            List<double[]> trainRows = new ArrayList<>();
            for (int i = 0; i < y.length; i++) if (!isTest[i]) trainRows.add(X[i]);
            StandardScaler scaler = new StandardScaler(CONTINUOUS_COLS).fit(trainRows.toArray(new double[0][]));
            scaler.saveToDb(FEATURE_NAMES);
            System.out.println("\n[2.6] Chuẩn hóa Z-score (fit trên tập train) - đã lưu vào bảng scaler_params:");
            double[] m = scaler.getMean(), s = scaler.getStd();
            for (int j = 0; j < CONTINUOUS_COLS.length; j++) {
                System.out.printf("   %-22s mean = %8.3f   std = %8.3f%n", FEATURE_NAMES[CONTINUOUS_COLS[j]], m[j], s[j]);
            }
            double[][] demo = scaler.transform(new double[][]{X[0]});
            System.out.println("   Ví dụ bản ghi đầu tiên trước/sau chuẩn hóa:");
            System.out.println("     trước: " + fmt(X[0]));
            System.out.println("     sau  : " + fmt(demo[0]));

            // ---------- Lưu processed_data ----------
            saveProcessed(c, X, y, rawIds, isTest);
            DatabaseManager.log(c, "2_tien_xu_ly", "so_ban_ghi_sau_xu_ly", y.length);
            System.out.printf("%nĐã lưu %,d bản ghi vào bảng processed_data%n", y.length);
        }
    }


    private List<RawRow> loadRaw() throws Exception {
        List<RawRow> list = new ArrayList<>(100_000);
        try (Connection c = DatabaseManager.getConnection(); Statement st = c.createStatement();
             ResultSet rs = st.executeQuery("SELECT id,gender,age,hypertension,heart_disease,smoking_history,"
                     + "bmi,hba1c_level,blood_glucose_level,diabetes FROM raw_data ORDER BY id")) {
            while (rs.next()) {
                RawRow r = new RawRow();
                r.id = rs.getInt(1);
                r.gender = rs.getString(2);
                r.age = getD(rs, 3);
                r.hypertension = getI(rs, 4);
                r.heartDisease = getI(rs, 5);
                r.smoking = rs.getString(6);
                r.bmi = getD(rs, 7);
                r.hba1c = getD(rs, 8);
                r.glucose = getD(rs, 9);
                r.diabetes = getI(rs, 10);
                list.add(r);
            }
        }
        return list;
    }

    private static Double getD(ResultSet rs, int i) throws Exception {
        double v = rs.getDouble(i);
        return rs.wasNull() ? null : v;
    }

    private static Integer getI(ResultSet rs, int i) throws Exception {
        double v = rs.getDouble(i);
        return rs.wasNull() ? null : (int) Math.round(v);
    }

    // Giá trị ngoài miền hợp lý được coi là thiếu
    private int markInvalidAsMissing(List<RawRow> rows) {
        int cnt = 0;
        Set<String> smokingOk = new HashSet<>(Arrays.asList(SMOKING_CATS));
        for (RawRow r : rows) {
            if (r.age != null && (r.age <= 0 || r.age > 120)) { r.age = null; cnt++; }
            if (r.bmi != null && (r.bmi < 10 || r.bmi > 100)) { r.bmi = null; cnt++; }
            if (r.hba1c != null && (r.hba1c < 3 || r.hba1c > 20)) { r.hba1c = null; cnt++; }
            if (r.glucose != null && (r.glucose < 40 || r.glucose > 500)) { r.glucose = null; cnt++; }
            if (r.hypertension != null && r.hypertension != 0 && r.hypertension != 1) { r.hypertension = null; cnt++; }
            if (r.heartDisease != null && r.heartDisease != 0 && r.heartDisease != 1) { r.heartDisease = null; cnt++; }
            if (r.diabetes != null && r.diabetes != 0 && r.diabetes != 1) { r.diabetes = null; cnt++; }
            if (r.smoking != null && !smokingOk.contains(r.smoking)) { r.smoking = null; cnt++; }
        }
        return cnt;
    }

    // Điền giá trị thiếu; trả về số bản ghi bị loại vì thiếu nhãn
    private int fillMissing(List<RawRow> rows) {
        int before = rows.size();
        rows.removeIf(r -> r.diabetes == null);
        double ageMed = median(rows, 0), bmiMed = median(rows, 1), hbMed = median(rows, 2), glMed = median(rows, 3);
        Map<String, Integer> genderCount = new HashMap<>();
        for (RawRow r : rows) if (r.gender != null) genderCount.merge(r.gender, 1, Integer::sum);
        String genderMode = genderCount.entrySet().stream().max(Map.Entry.comparingByValue()).map(Map.Entry::getKey).orElse("Female");
        for (RawRow r : rows) {
            if (r.age == null) r.age = ageMed;
            if (r.bmi == null) r.bmi = bmiMed;
            if (r.hba1c == null) r.hba1c = hbMed;
            if (r.glucose == null) r.glucose = glMed;
            if (r.hypertension == null) r.hypertension = 0;
            if (r.heartDisease == null) r.heartDisease = 0;
            if (r.gender == null) r.gender = genderMode;
            if (r.smoking == null) r.smoking = "No Info";
        }
        return before - rows.size();
    }

    private static double median(List<RawRow> rows, int which) {
        double[] v = rows.stream().map(r -> switch (which) {
            case 0 -> r.age; case 1 -> r.bmi; case 2 -> r.hba1c; default -> r.glucose;
        }).filter(Objects::nonNull).mapToDouble(Double::doubleValue).sorted().toArray();
        if (v.length == 0) return 0;
        return v.length % 2 == 1 ? v[v.length / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2.0;
    }

    private static double[] encode(RawRow r) {
        double[] x = new double[FEATURE_NAMES.length];
        x[0] = "Male".equals(r.gender) ? 1 : 0;
        x[1] = r.age;
        x[2] = r.hypertension;
        x[3] = r.heartDisease;
        x[4] = r.bmi;
        x[5] = r.hba1c;
        x[6] = r.glucose;
        for (int k = 0; k < SMOKING_CATS.length; k++) {
            x[7 + k] = SMOKING_CATS[k].equals(r.smoking) ? 1 : 0;
        }
        return x;
    }

    // Chia phân tầng: lấy đúng tỷ lệ testRatio từ mỗi lớp
    public static boolean[] stratifiedSplit(int[] y, double testRatio, long seed) {
        Random rnd = new Random(seed);
        boolean[] isTest = new boolean[y.length];
        for (int cls = 0; cls <= 1; cls++) {
            List<Integer> idx = new ArrayList<>();
            for (int i = 0; i < y.length; i++) if (y[i] == cls) idx.add(i);
            Collections.shuffle(idx, rnd);
            int nTest = (int) Math.round(idx.size() * testRatio);
            for (int i = 0; i < nTest; i++) isTest[idx.get(i)] = true;
        }
        return isTest;
    }

    private void saveProcessed(Connection c, double[][] X, int[] y, int[] rawIds, boolean[] isTest) throws Exception {
        c.setAutoCommit(false);
        try (Statement st = c.createStatement()) { st.executeUpdate("DELETE FROM processed_data"); }
        StringBuilder sql = new StringBuilder("INSERT INTO processed_data(id,raw_id,split");
        for (String f : FEATURE_NAMES) sql.append(',').append(f);
        sql.append(",diabetes) VALUES(?,?,?");
        for (int i = 0; i < FEATURE_NAMES.length; i++) sql.append(",?");
        sql.append(",?)");
        try (PreparedStatement ps = c.prepareStatement(sql.toString())) {
            for (int i = 0; i < y.length; i++) {
                int p = 1;
                ps.setInt(p++, i + 1);
                ps.setInt(p++, rawIds[i]);
                ps.setString(p++, isTest[i] ? "test" : "train");
                for (double v : X[i]) ps.setDouble(p++, v);
                ps.setInt(p, y[i]);
                ps.addBatch();
                if ((i + 1) % 5000 == 0) ps.executeBatch();
            }
            ps.executeBatch();
        }
        c.commit();
        c.setAutoCommit(true);
    }

    // Đọc tập train hoặc test (đã mã hóa, chưa chuẩn hóa) từ CSDL
    public static Dataset loadSplit(String split) throws Exception {
        List<double[]> xs = new ArrayList<>();
        List<Integer> ys = new ArrayList<>();
        String sql = "SELECT " + String.join(",", FEATURE_NAMES) + ",diabetes FROM processed_data WHERE split=? ORDER BY id";
        try (Connection c = DatabaseManager.getConnection(); PreparedStatement ps = c.prepareStatement(sql)) {
            ps.setString(1, split);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    double[] x = new double[FEATURE_NAMES.length];
                    for (int j = 0; j < x.length; j++) x[j] = rs.getDouble(j + 1);
                    xs.add(x);
                    ys.add(rs.getInt(FEATURE_NAMES.length + 1));
                }
            }
        }
        if (xs.isEmpty()) throw new IllegalStateException("Chưa có dữ liệu '" + split + "'. Hãy chạy Bước 2 (Tiền xử lý) trước.");
        return new Dataset(xs.toArray(new double[0][]), ys.stream().mapToInt(Integer::intValue).toArray(), FEATURE_NAMES);
    }

    private static String fmt(double[] x) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < x.length; i++) sb.append(i > 0 ? ", " : "").append(String.format(Locale.US, "%.2f", x[i]));
        return sb.append(']').toString();
    }
}
