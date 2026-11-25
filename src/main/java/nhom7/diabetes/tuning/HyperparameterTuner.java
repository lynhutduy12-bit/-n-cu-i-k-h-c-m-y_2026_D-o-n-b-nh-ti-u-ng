package nhom7.diabetes.tuning;

import nhom7.diabetes.config.AppConfig;
import nhom7.diabetes.data.Dataset;
import nhom7.diabetes.db.DatabaseManager;
import nhom7.diabetes.db.ResultRepository;
import nhom7.diabetes.evaluation.Metrics;
import nhom7.diabetes.model.*;
import nhom7.diabetes.model.ModelTrainer.ModelResult;
import nhom7.diabetes.preprocessing.DataPreprocessor;

import java.util.*;
import java.util.function.Function;

/**
 * Cross-Validation và điều chỉnh siêu tham số:
 * Grid Search cho LR, DT, KNN; Randomized Search cho RF.
 * Chọn bộ tham số có F1 trung bình (5-Fold CV) cao nhất.
 */
public class HyperparameterTuner {

    private static final int RF_N_ITER = Integer.getInteger("rf.iter", 12);

    // Kết quả tinh chỉnh 1 mô hình
    public record TuningResult(String model, Map<String, Object> bestParams, String bestParamsText,
                               CrossValidator.CVResult bestCv, int combosTried) {}

    private CrossValidator.Fold[] folds;

    public List<ModelResult> run(Set<String> which) throws Exception {
        System.out.println("\n========== BƯỚC 4: CROSS-VALIDATION & ĐIỀU CHỈNH THAM SỐ ==========");
        DatabaseManager.initSchema();
        Dataset train = DataPreprocessor.loadSplit("train");
        Dataset test = DataPreprocessor.loadSplit("test");
        int k = AppConfig.K_FOLDS;
        System.out.printf("Stratified %d-Fold CV trên tập train (%,d mẫu). Tiêu chí chọn: F1-score trung bình.%n", k, train.size());
        folds = CrossValidator.makeFolds(train, k, AppConfig.SEED);
        for (int f = 0; f < k; f++) {
            int pos = 0;
            for (int v : folds[f].yVal) pos += v;
            System.out.printf("   Fold %d: train=%,d  validation=%,d (dương tính %d)%n",
                    f + 1, folds[f].yTrain.length, folds[f].yVal.length, pos);
        }

        List<TuningResult> tuned = new ArrayList<>();
        if (which.contains("lr")) tuned.add(tuneLogistic());
        if (which.contains("dt")) tuned.add(tuneDecisionTree());
        if (which.contains("knn")) tuned.add(tuneKnn());
        if (which.contains("rf")) tuned.add(tuneRandomForest());

        // ---------- Huấn luyện lại với tham số tốt nhất & đánh giá trên tập test ----------
        System.out.println("\n========== HUẤN LUYỆN LẠI VỚI THAM SỐ TỐT NHẤT & ĐÁNH GIÁ TRÊN TẬP TEST ==========");
        ResultRepository.clearStage(ModelTrainer.STAGE_TUNED);
        List<ModelResult> results = new ArrayList<>();
        for (TuningResult tr : tuned) {
            Classifier best = build(tr.model(), tr.bestParams());
            results.add(ModelTrainer.trainAndEvaluate(best, train, test, ModelTrainer.STAGE_TUNED));
        }

        System.out.println("\n" + "=".repeat(100));
        System.out.println(" TỔNG HỢP THAM SỐ TỐT NHẤT (theo " + k + "-Fold CV)");
        System.out.println("=".repeat(100));
        for (TuningResult tr : tuned) {
            System.out.printf(" %-20s | %3d tổ hợp | CV F1 = %.4f ± %.4f | CV AUC = %.4f%n      -> %s%n",
                    tr.model(), tr.combosTried(), tr.bestCv().meanF1(), tr.bestCv().stdF1(), tr.bestCv().meanAuc(), tr.bestParamsText());
        }
        ModelTrainer.printSummary("KẾT QUẢ SAU TINH CHỈNH TRÊN TẬP TEST", results);
        return results;
    }

    // ---------- Không gian tham số từng mô hình ----------

    private TuningResult tuneLogistic() throws Exception {
        LinkedHashMap<String, List<Object>> space = new LinkedHashMap<>();
        space.put("C", List.of(0.001, 0.01, 0.1, 1.0, 10.0, 100.0));
        space.put("class_weight", List.of("none", "balanced"));
        return search("Logistic Regression", "GridSearch", grid(space));
    }

    private TuningResult tuneDecisionTree() throws Exception {
        LinkedHashMap<String, List<Object>> space = new LinkedHashMap<>();
        space.put("max_depth", List.of(3, 5, 7, 9, 12, 15, 0)); // 0 = None
        space.put("min_samples_leaf", List.of(1, 5, 20, 50));
        space.put("criterion", List.of("gini", "entropy"));
        return search("Decision Tree", "GridSearch", grid(space));
    }

    private TuningResult tuneKnn() throws Exception {
        LinkedHashMap<String, List<Object>> space = new LinkedHashMap<>();
        space.put("metric", List.of("euclidean", "manhattan"));
        space.put("n_neighbors", List.of(3, 5, 7, 9, 11, 15, 21, 31));
        space.put("weights", List.of("uniform", "distance"));
        return searchKnn(grid(space));
    }

    private TuningResult tuneRandomForest() throws Exception {
        LinkedHashMap<String, List<Object>> space = new LinkedHashMap<>();
        space.put("n_estimators", List.of(50, 100, 150, 200));
        space.put("max_depth", List.of(8, 12, 16, 20, 0));
        space.put("min_samples_leaf", List.of(1, 2, 5, 10));
        space.put("max_features", List.of("sqrt", "0.3", "0.5"));
        List<Map<String, Object>> all = grid(space);
        List<Map<String, Object>> sampled = randomSample(all, RF_N_ITER, AppConfig.SEED);
        System.out.printf("%n(Randomized Search: chọn ngẫu nhiên %d / %d tổ hợp)%n", sampled.size(), all.size());
        return search("Random Forest", "RandomizedSearch", sampled);
    }

    // Tạo mô hình từ tên + bộ tham số
    public static Classifier build(String model, Map<String, Object> p) {
        return switch (model) {
            case "Logistic Regression" -> new LogisticRegressionModel((Double) p.get("C"), (String) p.get("class_weight"));
            case "Decision Tree" -> new DecisionTreeModel((Integer) p.get("max_depth"), 2,
                    (Integer) p.get("min_samples_leaf"), (String) p.get("criterion"));
            case "KNN" -> new KNNModel((Integer) p.get("n_neighbors"), (String) p.get("weights"), (String) p.get("metric"));
            case "Random Forest" -> new RandomForestModel((Integer) p.get("n_estimators"), (Integer) p.get("max_depth"),
                    (Integer) p.get("min_samples_leaf"), (String) p.get("max_features"));
            default -> throw new IllegalArgumentException(model);
        };
    }

    // ---------- Grid / Randomized Search ----------

    // Tạo tất cả tổ hợp tham số
    public static List<Map<String, Object>> grid(LinkedHashMap<String, List<Object>> space) {
        List<Map<String, Object>> out = new ArrayList<>();
        out.add(new LinkedHashMap<>());
        for (Map.Entry<String, List<Object>> e : space.entrySet()) {
            List<Map<String, Object>> next = new ArrayList<>();
            for (Map<String, Object> partial : out) {
                for (Object v : e.getValue()) {
                    Map<String, Object> m = new LinkedHashMap<>(partial);
                    m.put(e.getKey(), v);
                    next.add(m);
                }
            }
            out = next;
        }
        return out;
    }

    public static List<Map<String, Object>> randomSample(List<Map<String, Object>> all, int n, long seed) {
        List<Map<String, Object>> copy = new ArrayList<>(all);
        Collections.shuffle(copy, new Random(seed));
        return copy.subList(0, Math.min(n, copy.size()));
    }

    private TuningResult search(String model, String type, List<Map<String, Object>> combos) throws Exception {
        return search(model, type, combos, p -> CrossValidator.crossValidate(() -> build(model, p), folds));
    }

    private TuningResult search(String model, String type, List<Map<String, Object>> combos,
                                Function<Map<String, Object>, CrossValidator.CVResult> evaluator) throws Exception {
        System.out.printf("%n--- %s: %s với %d tổ hợp x %d fold ---%n", model, type, combos.size(), folds.length);
        System.out.printf(" %4s  %-72s %8s %8s %8s %8s %8s %8s%n", "#", "Tham số", "Acc", "Prec", "Recall", "F1", "±F1", "AUC");
        ResultRepository.clearCv(model);
        Map<String, Object> bestP = null;
        CrossValidator.CVResult bestR = null;
        String bestText = null;
        int i = 0;
        for (Map<String, Object> p : combos) {
            i++;
            CrossValidator.CVResult r = evaluator.apply(p);
            String text = build(model, p).params();
            System.out.printf(" %4d  %-72s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f%n",
                    i, text, r.meanAcc(), r.meanPrec(), r.meanRec(), r.meanF1(), r.stdF1(), r.meanAuc());
            ResultRepository.saveCvResult(model, type, text, folds.length, r.meanAcc(), r.meanPrec(),
                    r.meanRec(), r.meanF1(), r.stdF1(), r.meanAuc(), r.timeMs);
            if (bestR == null || r.meanF1() > bestR.meanF1() + 1e-9
                    || (Math.abs(r.meanF1() - bestR.meanF1()) <= 1e-9 && r.meanAuc() > bestR.meanAuc())) {
                bestR = r;
                bestP = p;
                bestText = text;
            }
        }
        ResultRepository.markBestCv(model, bestText);
        System.out.printf(" => Tốt nhất: %s  (F1 = %.4f, AUC = %.4f)%n", bestText, bestR.meanF1(), bestR.meanAuc());
        return new TuningResult(model, bestP, bestText, bestR, combos.size());
    }

    // Grid Search cho KNN có tối ưu tính toán
    private TuningResult searchKnn(List<Map<String, Object>> combos) throws Exception {
        int maxK = combos.stream().mapToInt(p -> (Integer) p.get("n_neighbors")).max().orElse(5);
        Map<String, KNNModel.Neighbors[]> cache = new HashMap<>();
        long tStart = System.currentTimeMillis();
        for (String metric : new String[]{"euclidean", "manhattan"}) {
            if (combos.stream().noneMatch(p -> metric.equals(p.get("metric")))) continue;
            System.out.printf("%n(KNN) Đang tính khoảng cách cho %d fold, metric=%s ...%n", folds.length, metric);
            KNNModel.Neighbors[] arr = new KNNModel.Neighbors[folds.length];
            for (int f = 0; f < folds.length; f++) {
                arr[f] = KNNModel.findNeighbors(folds[f].xTrain, folds[f].xVal, maxK, metric);
            }
            cache.put(metric, arr);
        }
        long neighborMs = System.currentTimeMillis() - tStart;
        System.out.printf("(KNN) Hoàn tất tìm láng giềng trong %,d ms%n", neighborMs);

        return search("KNN", "GridSearch", combos, p -> {
            KNNModel.Neighbors[] arr = cache.get((String) p.get("metric"));
            List<Metrics> res = new ArrayList<>();
            long t0 = System.currentTimeMillis();
            for (int f = 0; f < folds.length; f++) {
                double[] proba = KNNModel.probaFromNeighbors(arr[f], folds[f].yTrain,
                        (Integer) p.get("n_neighbors"), (String) p.get("weights"));
                res.add(Metrics.evaluate(folds[f].yVal, proba));
            }
            return new CrossValidator.CVResult(res, System.currentTimeMillis() - t0);
        });
    }
}
