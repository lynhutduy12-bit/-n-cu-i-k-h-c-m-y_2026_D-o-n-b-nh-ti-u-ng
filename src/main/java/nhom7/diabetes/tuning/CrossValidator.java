package nhom7.diabetes.tuning;

import nhom7.diabetes.data.Dataset;
import nhom7.diabetes.evaluation.Metrics;
import nhom7.diabetes.model.Classifier;
import nhom7.diabetes.preprocessing.DataPreprocessor;
import nhom7.diabetes.preprocessing.StandardScaler;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

/**
 * Stratified K-Fold Cross-Validation: chia tập train thành K phần có cùng tỷ lệ nhãn,
 * lần lượt lấy 1 phần để kiểm định, K-1 phần còn lại để huấn luyện.
 */
public class CrossValidator {

    // Dữ liệu của 1 fold đã chuẩn hóa sẵn (tái sử dụng cho mọi bộ tham số)
    public static final class Fold {
        public final double[][] xTrain, xVal;
        public final int[] yTrain, yVal;

        Fold(double[][] xTrain, int[] yTrain, double[][] xVal, int[] yVal) {
            this.xTrain = xTrain;
            this.yTrain = yTrain;
            this.xVal = xVal;
            this.yVal = yVal;
        }
    }

    // Kết quả Cross-Validation của 1 bộ tham số
    public static final class CVResult {
        public final List<Metrics> folds;
        public final long timeMs;

        public CVResult(List<Metrics> folds, long timeMs) {
            this.folds = folds;
            this.timeMs = timeMs;
        }

        public double meanAcc()  { return folds.stream().mapToDouble(m -> m.accuracy).average().orElse(0); }
        public double meanPrec() { return folds.stream().mapToDouble(m -> m.precision).average().orElse(0); }
        public double meanRec()  { return folds.stream().mapToDouble(m -> m.recall).average().orElse(0); }
        public double meanF1()   { return folds.stream().mapToDouble(m -> m.f1).average().orElse(0); }
        public double meanAuc()  { return folds.stream().mapToDouble(m -> m.rocAuc).average().orElse(0); }

        public double stdF1() {
            double mean = meanF1();
            return Math.sqrt(folds.stream().mapToDouble(m -> (m.f1 - mean) * (m.f1 - mean)).average().orElse(0));
        }
    }

    // Gán mỗi mẫu vào 1 fold sao cho mỗi fold có tỷ lệ lớp như nhau
    public static int[] stratifiedFoldAssignment(int[] y, int k, long seed) {
        int[] foldOf = new int[y.length];
        Random rnd = new Random(seed);
        for (int cls = 0; cls <= 1; cls++) {
            List<Integer> idx = new ArrayList<>();
            for (int i = 0; i < y.length; i++) if (y[i] == cls) idx.add(i);
            Collections.shuffle(idx, rnd);
            for (int r = 0; r < idx.size(); r++) foldOf[idx.get(r)] = r % k;
        }
        return foldOf;
    }

    // Tạo K fold, mỗi fold đã chuẩn hóa bằng scaler fit trên phần train của fold
    public static Fold[] makeFolds(Dataset data, int k, long seed) {
        int[] foldOf = stratifiedFoldAssignment(data.y, k, seed);
        Fold[] folds = new Fold[k];
        for (int f = 0; f < k; f++) {
            List<Integer> tr = new ArrayList<>(), va = new ArrayList<>();
            for (int i = 0; i < data.size(); i++) (foldOf[i] == f ? va : tr).add(i);
            Dataset dTr = data.subset(tr.stream().mapToInt(Integer::intValue).toArray());
            Dataset dVa = data.subset(va.stream().mapToInt(Integer::intValue).toArray());
            StandardScaler sc = new StandardScaler(DataPreprocessor.CONTINUOUS_COLS).fit(dTr.X);
            folds[f] = new Fold(sc.transform(dTr.X), dTr.y, sc.transform(dVa.X), dVa.y);
        }
        return folds;
    }

    // Chạy Cross-Validation cho 1 mô hình (factory tạo mô hình mới cho mỗi fold)
    public static CVResult crossValidate(Supplier<Classifier> factory, Fold[] folds) {
        long t0 = System.currentTimeMillis();
        List<Metrics> res = new ArrayList<>();
        for (Fold fd : folds) {
            Classifier m = factory.get();
            m.fit(fd.xTrain, fd.yTrain);
            res.add(Metrics.evaluate(fd.yVal, m.predictProba(fd.xVal)));
        }
        return new CVResult(res, System.currentTimeMillis() - t0);
    }
}
