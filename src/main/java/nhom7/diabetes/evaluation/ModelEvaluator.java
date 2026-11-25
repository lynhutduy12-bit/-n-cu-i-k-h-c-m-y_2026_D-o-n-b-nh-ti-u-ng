package nhom7.diabetes.evaluation;

import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import nhom7.diabetes.config.AppConfig;
import nhom7.diabetes.data.Dataset;
import nhom7.diabetes.db.DatabaseManager;
import nhom7.diabetes.model.Classifier;
import nhom7.diabetes.model.SavedModel;
import nhom7.diabetes.preprocessing.DataPreprocessor;
import nhom7.diabetes.preprocessing.StandardScaler;
import nhom7.diabetes.report.ChartDrawer;
import nhom7.diabetes.tuning.CrossValidator;
import nhom7.diabetes.tuning.HyperparameterTuner;

/**
 * Đánh giá, so sánh và chọn mô hình tốt nhất:
 * huấn luyện lại 4 mô hình với tham số tốt nhất, tính 6 chỉ số trên tập test, vẽ biểu đồ,
 * phân tích ngưỡng và lưu mô hình tốt nhất ra output/best_model.ser.
 */
public class ModelEvaluator {

    private static final String[] MODELS = {"Logistic Regression", "Decision Tree", "KNN", "Random Forest"};

    
    private static final double F1_TOLERANCE = 0.01;

    private static class Result {
        String name, params;
        double cvF1, cvF1Std, cvAuc;  
        Classifier model;
        double[] testProba;           
        Metrics test;                 
        long trainMs, predictMs;
    }

    public void run() throws Exception {
        System.out.println("\n========== BƯỚC 5: ĐÁNH GIÁ, SO SÁNH & CHỌN MÔ HÌNH TỐT NHẤT ==========");
        DatabaseManager.initSchema();
        String out = AppConfig.OUTPUT_DIR;
        new java.io.File(out).mkdirs();

        Dataset train = DataPreprocessor.loadSplit("train");
        Dataset test = DataPreprocessor.loadSplit("test");
        StandardScaler scaler = new StandardScaler(DataPreprocessor.CONTINUOUS_COLS).fit(train.X);
        double[][] xTrain = scaler.transform(train.X);
        double[][] xTest = scaler.transform(test.X);

        // ---------- Huấn luyện lại & đánh giá trên tập test ----------
        System.out.println("\n[5.1] Huấn luyện lại 4 mô hình với tham số tốt nhất & đánh giá trên tập test");
        List<Result> results = loadBestParamsFromCv();
        for (Result r : results) {
            r.model = HyperparameterTuner.build(r.name, parseParams(r.params));
            long t0 = System.currentTimeMillis();
            r.model.fit(xTrain, train.y);
            r.trainMs = System.currentTimeMillis() - t0;
            t0 = System.currentTimeMillis();
            r.testProba = r.model.predictProba(xTest);
            r.predictMs = System.currentTimeMillis() - t0;
            r.test = Metrics.evaluate(test.y, r.testProba);
            System.out.printf("%n>> %s [%s]%n   %s%n", r.name, r.params, r.test);
            System.out.println(r.test.confusionMatrix().indent(3).stripTrailing());
        }
        printComparisonTable(results);

        // ---------- Biểu đồ ----------
        System.out.println("\n[5.3] Vẽ biểu đồ");
        List<String> names = new ArrayList<>();
        List<Metrics> metrics = new ArrayList<>();
        List<double[][]> rocCurves = new ArrayList<>();
        List<Double> aucs = new ArrayList<>();
        for (Result r : results) {
            names.add(r.name);
            metrics.add(r.test);
            rocCurves.add(Metrics.rocCurve(test.y, r.testProba));
            aucs.add(r.test.rocAuc);
        }
        ChartDrawer.rocChart(names, rocCurves, aucs, out + "/hinh_roc_curve.png");
        ChartDrawer.confusionMatrixChart(names, metrics, out + "/hinh_confusion_matrix.png");
        ChartDrawer.metricBarChart(names, metrics, out + "/hinh_so_sanh_chi_so.png");
        exportRocCsv(names, rocCurves, out + "/roc_curve.csv");

        // ---------- Chọn mô hình tốt nhất ----------
        Result best = chooseBest(results);

        // ---------- Phân tích ngưỡng cho mô hình tốt nhất ----------
        System.out.println("\n[5.5] Phân tích ngưỡng dự đoán cho " + best.name + " (dựa trên 5-Fold CV trên tập train)");
        double[] th = new double[19];                       // 0.05, 0.10, ..., 0.95
        for (int i = 0; i < th.length; i++) th[i] = Math.round((i + 1) * 0.05 * 100) / 100.0;
        Metrics[] cvAtTh = thresholdAnalysisCv(best, train, th);
        double chosenTh = chooseThreshold(th, cvAtTh);
        Metrics testAtChosen = Metrics.evaluate(test.y, best.testProba, chosenTh);
        printThresholdTable(th, cvAtTh, chosenTh);
        System.out.printf("%n   Kiểm tra lại trên tập test với ngưỡng %.2f:%n   - Ngưỡng 0.50: %s%n   - Ngưỡng %.2f: %s%n",
                chosenTh, best.test, chosenTh, testAtChosen);
        System.out.println(testAtChosen.confusionMatrix().indent(3).stripTrailing());

        double[] p = new double[th.length], rc = new double[th.length], f = new double[th.length];
        for (int i = 0; i < th.length; i++) { p[i] = cvAtTh[i].precision; rc[i] = cvAtTh[i].recall; f[i] = cvAtTh[i].f1; }
        ChartDrawer.thresholdChart(best.name, th, p, rc, f, chosenTh, out + "/hinh_phan_tich_nguong.png");
        if (best.model.featureImportance() != null) {
            ChartDrawer.featureImportanceChart(best.name, train.featureNames, best.model.featureImportance(),
                    out + "/hinh_do_quan_trong_dac_trung.png");
        }

        // ---------- Lưu mô hình tốt nhất ----------
        String modelFile = out + "/best_model.ser";
        new SavedModel(best.model, scaler, train.featureNames, chosenTh).save(modelFile);
        System.out.printf("%n[5.6] Đã lưu mô hình tốt nhất (%s, ngưỡng %.2f) vào %s%n", best.name, chosenTh, modelFile);
        checkSavedModel(modelFile, test, best);

        // ---------- Xuất số liệu ra CSV ----------
        exportComparisonCsv(results, out + "/so_sanh_mo_hinh.csv");
        exportThresholdCsv(best.name, th, cvAtTh, out + "/phan_tich_nguong.csv");
    }

    // Đọc tham số tốt nhất của từng mô hình từ bảng cv_results
    private List<Result> loadBestParamsFromCv() throws Exception {
        Map<String, Result> found = new HashMap<>();
        try (Connection c = DatabaseManager.getConnection(); Statement st = c.createStatement();
             ResultSet rs = st.executeQuery("SELECT model, params, f1_mean, f1_std, auc_mean FROM cv_results WHERE is_best = 1")) {
            while (rs.next()) {
                Result r = new Result();
                r.name = rs.getString(1);
                r.params = rs.getString(2);
                r.cvF1 = rs.getDouble(3);
                r.cvF1Std = rs.getDouble(4);
                r.cvAuc = rs.getDouble(5);
                found.put(r.name, r);
            }
        }
        List<Result> list = new ArrayList<>();
        for (String m : MODELS) {
            if (!found.containsKey(m)) {
                throw new IllegalStateException("Chưa có tham số tốt nhất của " + m
                        + ". Hãy chạy Bước 4 (Cross-Validation & điều chỉnh tham số) trước.");
            }
            list.add(found.get(m));
        }
        return list;
    }

    // Đổi chuỗi tham số (vd "C=100.0, class_weight=none") thành Map
    static Map<String, Object> parseParams(String text) {
        Map<String, Object> p = new LinkedHashMap<>();
        for (String part : text.split(",")) {
            String[] kv = part.trim().split("=");
            String key = kv[0].trim(), val = kv[1].trim();
            if (val.equals("None")) p.put(key, 0);
            else if (key.equals("C")) p.put(key, Double.parseDouble(val));
            else if (key.equals("max_features")) p.put(key, val);
            else if (val.matches("-?\\d+")) p.put(key, Integer.parseInt(val));
            else p.put(key, val);
        }
        return p;
    }

    // Chọn mô hình tốt nhất: trong nhóm có CV F1 gần cao nhất (lệch <= 0.01), lấy mô hình có CV AUC cao nhất
    private Result chooseBest(List<Result> results) {
        System.out.println("\n[5.4] Lựa chọn mô hình tốt nhất (theo kết quả 5-Fold CV)");
        double maxF1 = results.stream().mapToDouble(r -> r.cvF1).max().orElse(0);
        Result best = null;
        for (Result r : results) {
            boolean nearTop = maxF1 - r.cvF1 <= F1_TOLERANCE;
            System.out.printf("   %-20s CV F1 = %.4f ± %.4f | CV AUC = %.4f %s%n",
                    r.name, r.cvF1, r.cvF1Std, r.cvAuc, nearTop ? "<- thuộc nhóm F1 cao nhất" : "");
            if (nearTop && (best == null || r.cvAuc > best.cvAuc)) best = r;
        }
        System.out.printf("   => MÔ HÌNH ĐƯỢC CHỌN: %s (CV F1 = %.4f, CV AUC = %.4f)%n", best.name, best.cvF1, best.cvAuc);
        return best;
    }

    // Phân tích ngưỡng: tính chỉ số ở từng ngưỡng trên 5-Fold CV (không dùng tập test)
    private Metrics[] thresholdAnalysisCv(Result best, Dataset train, double[] thresholds) {
        CrossValidator.Fold[] folds = CrossValidator.makeFolds(train, AppConfig.K_FOLDS, AppConfig.SEED);
        int n = train.size(), pos = 0;
        int[] yAll = new int[n];
        double[] pAll = new double[n];
        for (CrossValidator.Fold fd : folds) {
            Classifier m = HyperparameterTuner.build(best.name, parseParams(best.params));
            m.fit(fd.xTrain, fd.yTrain);
            double[] proba = m.predictProba(fd.xVal);
            for (int i = 0; i < proba.length; i++) {
                yAll[pos] = fd.yVal[i];
                pAll[pos] = proba[i];
                pos++;
            }
        }
        Metrics[] res = new Metrics[thresholds.length];
        for (int i = 0; i < thresholds.length; i++) res[i] = Metrics.evaluate(yAll, pAll, thresholds[i]);
        return res;
    }

    private double chooseThreshold(double[] th, Metrics[] m) {
        int best = 0;
        for (int i = 1; i < th.length; i++) if (m[i].f1 > m[best].f1) best = i;
        return th[best];
    }

    // In bảng so sánh
    private void printComparisonTable(List<Result> results) {
        System.out.println("\n" + "=".repeat(110));
        System.out.println(" [5.2] BẢNG SO SÁNH 4 MÔ HÌNH TRÊN TẬP TEST (ngưỡng 0.5)");
        System.out.println("=".repeat(110));
        System.out.printf(" %-20s %9s %9s %9s %9s %9s %7s %7s %7s %7s %10s %10s%n",
                "Mô hình", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "TN", "FP", "FN", "TP", "Train(ms)", "Dự đoán(ms)");
        System.out.println("-".repeat(110));
        for (Result r : results) {
            Metrics m = r.test;
            System.out.printf(" %-20s %9.4f %9.4f %9.4f %9.4f %9.4f %7d %7d %7d %7d %,10d %,10d%n",
                    r.name, m.accuracy, m.precision, m.recall, m.f1, m.rocAuc, m.tn, m.fp, m.fn, m.tp, r.trainMs, r.predictMs);
        }
        System.out.println("=".repeat(110));
        System.out.printf(" Cao nhất:  Accuracy -> %s | Precision -> %s | Recall -> %s | F1 -> %s | ROC-AUC -> %s%n",
                argMax(results, r -> r.test.accuracy), argMax(results, r -> r.test.precision),
                argMax(results, r -> r.test.recall), argMax(results, r -> r.test.f1), argMax(results, r -> r.test.rocAuc));
    }

    private void printThresholdTable(double[] th, Metrics[] m, double chosen) {
        System.out.printf("   %8s %9s %9s %9s %9s %7s %7s%n", "Ngưỡng", "Accuracy", "Precision", "Recall", "F1", "FN", "FP");
        for (int i = 0; i < th.length; i++) {
            String mark = th[i] == chosen ? "  <- F1 cao nhất (chọn)" : (th[i] == 0.5 ? "  <- mặc định" : "");
            System.out.printf("   %8.2f %9.4f %9.4f %9.4f %9.4f %7d %7d%s%n",
                    th[i], m[i].accuracy, m[i].precision, m[i].recall, m[i].f1, m[i].fn, m[i].fp, mark);
        }
    }

    // Nạp lại file mô hình vừa lưu và dự đoán thử để kiểm tra
    private void checkSavedModel(String file, Dataset test, Result best) throws Exception {
        int i = 0;
        while (i < test.size() - 1 && test.y[i] != 1) i++;   // bản ghi dương tính đầu tiên
        SavedModel sm = SavedModel.load(file);
        double p1 = sm.predictProba(test.X[i]);
        double p2 = best.testProba[i];
        System.out.printf("   Kiểm tra: nạp lại file, dự đoán 1 bản ghi test có nhãn 1 -> xác suất %.4f (trước khi lưu: %.4f) %s%n",
                p1, p2, Math.abs(p1 - p2) < 1e-9 ? "OK" : "KHÁC!");
        System.out.printf("   Kết luận với ngưỡng %.2f: %s%n", sm.threshold,
                p1 >= sm.threshold ? "CÓ nguy cơ tiểu đường" : "KHÔNG có nguy cơ");
    }

    private static String argMax(List<Result> rs, java.util.function.ToDoubleFunction<Result> f) {
        return rs.stream().max(Comparator.comparingDouble(f)).map(r -> r.name).orElse("");
    }

    // Xuất số liệu ra CSV
    private void exportComparisonCsv(List<Result> results, String file) throws Exception {
        try (PrintWriter pw = new PrintWriter(file, StandardCharsets.UTF_8)) {
            pw.print('\uFEFF'); // BOM để Excel đọc đúng tiếng Việt
            pw.println("model,params,cv_f1_mean,cv_f1_std,cv_auc_mean,test_accuracy,test_precision,test_recall,test_f1,test_roc_auc,tn,fp,fn,tp,train_ms,predict_ms");
            for (Result r : results) {
                Metrics m = r.test;
                pw.printf(Locale.US, "%s,\"%s\",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%d,%d,%d%n",
                        r.name, r.params, r.cvF1, r.cvF1Std, r.cvAuc, m.accuracy, m.precision, m.recall, m.f1, m.rocAuc,
                        m.tn, m.fp, m.fn, m.tp, r.trainMs, r.predictMs);
            }
        }
        System.out.println("   Đã xuất " + file);
    }

    private void exportRocCsv(List<String> names, List<double[][]> curves, String file) throws Exception {
        try (PrintWriter pw = new PrintWriter(file, StandardCharsets.UTF_8)) {
            pw.println("model,fpr,tpr");
            for (int m = 0; m < names.size(); m++) {
                for (double[] pt : curves.get(m)) pw.printf(Locale.US, "%s,%.6f,%.6f%n", names.get(m), pt[0], pt[1]);
            }
        }
        System.out.println("   Đã xuất " + file);
    }

    private void exportThresholdCsv(String model, double[] th, Metrics[] m, String file) throws Exception {
        try (PrintWriter pw = new PrintWriter(file, StandardCharsets.UTF_8)) {
            pw.println("model,threshold,accuracy,precision,recall,f1,tn,fp,fn,tp");
            for (int i = 0; i < th.length; i++) {
                pw.printf(Locale.US, "%s,%.2f,%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%d%n", model, th[i],
                        m[i].accuracy, m[i].precision, m[i].recall, m[i].f1, m[i].tn, m[i].fp, m[i].fn, m[i].tp);
            }
        }
        System.out.println("   Đã xuất " + file);
    }
}