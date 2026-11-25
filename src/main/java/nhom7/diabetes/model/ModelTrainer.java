package nhom7.diabetes.model;

import nhom7.diabetes.data.Dataset;
import nhom7.diabetes.db.DatabaseManager;
import nhom7.diabetes.db.ResultRepository;
import nhom7.diabetes.evaluation.Metrics;
import nhom7.diabetes.preprocessing.DataPreprocessor;
import nhom7.diabetes.preprocessing.StandardScaler;

import java.util.ArrayList;
import java.util.List;

/**
 * Xây dựng mô hình baseline: huấn luyện 4 mô hình với tham số mặc định,
 * đánh giá trên tập test và lưu kết quả vào CSDL.
 */
public class ModelTrainer {

    public static final String STAGE_BASELINE = "baseline";
    public static final String STAGE_TUNED = "tuned";

    public record ModelResult(String model, String params, Metrics metrics, long trainMs) {}

    public List<ModelResult> run() throws Exception {
        System.out.println("\n========== BƯỚC 3: XÂY DỰNG MÔ HÌNH (BASELINE - THAM SỐ MẶC ĐỊNH) ==========");
        DatabaseManager.initSchema();
        Dataset train = DataPreprocessor.loadSplit("train");
        Dataset test = DataPreprocessor.loadSplit("test");
        System.out.printf("Train: %,d mẫu | Test: %,d mẫu | %d đặc trưng%n", train.size(), test.size(), train.numFeatures());

        ResultRepository.clearStage(STAGE_BASELINE);
        List<Classifier> models = List.of(
                new LogisticRegressionModel(),
                new DecisionTreeModel(),
                new KNNModel(),
                new RandomForestModel()
        );
        List<ModelResult> results = new ArrayList<>();
        for (Classifier m : models) {
            results.add(trainAndEvaluate(m, train, test, STAGE_BASELINE));
        }
        printSummary("KẾT QUẢ BASELINE TRÊN TẬP TEST", results);
        return results;
    }


    public static ModelResult trainAndEvaluate(Classifier model, Dataset train, Dataset test, String stage) throws Exception {
        StandardScaler scaler = new StandardScaler(DataPreprocessor.CONTINUOUS_COLS).fit(train.X);
        double[][] xtr = scaler.transform(train.X);
        double[][] xte = scaler.transform(test.X);

        System.out.printf("%n>> %s [%s]%n", model.name(), model.params());
        long t0 = System.currentTimeMillis();
        model.fit(xtr, train.y);
        long trainMs = System.currentTimeMillis() - t0;
        Metrics m = Metrics.evaluate(test.y, model.predictProba(xte));
        System.out.printf("   Thời gian huấn luyện: %,d ms%n   %s%n", trainMs, m);
        System.out.println("   Ma trận nhầm lẫn:");
        System.out.println(m.confusionMatrix().indent(3).stripTrailing());

        ResultRepository.saveModelResult(stage, model.name(), model.params(), m, trainMs);
        double[] imp = model.featureImportance();
        if (imp != null) {
            ResultRepository.saveImportance(stage, model.name(), train.featureNames, imp);
            printTopFeatures(train.featureNames, imp, "Gini importance");
        }
        return new ModelResult(model.name(), model.params(), m, trainMs);
    }

    private static void printTopFeatures(String[] names, double[] imp, String label) {
        Integer[] order = new Integer[imp.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        java.util.Arrays.sort(order, (a, b) -> Double.compare(imp[b], imp[a]));
        StringBuilder sb = new StringBuilder("   Top 5 đặc trưng (" + label + "): ");
        for (int i = 0; i < Math.min(5, order.length); i++) {
            sb.append(String.format("%s%s=%.3f", i > 0 ? ", " : "", names[order[i]], imp[order[i]]));
        }
        System.out.println(sb);
    }

    public static void printSummary(String title, List<ModelResult> results) {
        System.out.println("\n" + "=".repeat(86));
        System.out.println(" " + title);
        System.out.println("=".repeat(86));
        System.out.printf(" %-20s %9s %9s %9s %9s %9s %12s%n", "Mô hình", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Train (ms)");
        System.out.println("-".repeat(86));
        for (ModelResult r : results) {
            Metrics m = r.metrics();
            System.out.printf(" %-20s %9.4f %9.4f %9.4f %9.4f %9.4f %,12d%n",
                    r.model(), m.accuracy, m.precision, m.recall, m.f1, m.rocAuc, r.trainMs());
        }
        System.out.println("=".repeat(86));
    }
}
