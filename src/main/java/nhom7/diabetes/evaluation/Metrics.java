package nhom7.diabetes.evaluation;

import nhom7.diabetes.config.AppConfig;

import java.util.Arrays;

/**
 * Các độ đo đánh giá mô hình phân loại nhị phân.
 *
 *   Accuracy  = (TP + TN) / (TP + TN + FP + FN)
 *   Precision = TP / (TP + FP)
 *   Recall    = TP / (TP + FN)          (độ nhạy - quan trọng trong y tế)
 *   F1-score  = 2·P·R / (P + R)
 *   ROC-AUC   = diện tích dưới đường cong ROC (tính bằng thống kê Mann–Whitney)
 */
public final class Metrics {

    public final int tp, tn, fp, fn;
    public final double accuracy, precision, recall, f1, rocAuc;

    private Metrics(int tp, int tn, int fp, int fn, double auc) {
        this.tp = tp; this.tn = tn; this.fp = fp; this.fn = fn;
        int n = tp + tn + fp + fn;
        this.accuracy = (double) (tp + tn) / n;
        this.precision = tp + fp == 0 ? 0 : (double) tp / (tp + fp);
        this.recall = tp + fn == 0 ? 0 : (double) tp / (tp + fn);
        this.f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
        this.rocAuc = auc;
    }

    public static Metrics evaluate(int[] yTrue, double[] proba) {
        return evaluate(yTrue, proba, AppConfig.THRESHOLD);
    }

    // Đánh giá với ngưỡng tùy chọn
    public static Metrics evaluate(int[] yTrue, double[] proba, double threshold) {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (int i = 0; i < yTrue.length; i++) {
            int pred = proba[i] >= threshold ? 1 : 0;
            if (pred == 1 && yTrue[i] == 1) tp++;
            else if (pred == 0 && yTrue[i] == 0) tn++;
            else if (pred == 1) fp++;
            else fn++;
        }
        return new Metrics(tp, tn, fp, fn, rocAuc(yTrue, proba));
    }

    // Tính ROC-AUC bằng thống kê Mann–Whitney
    public static double rocAuc(int[] y, double[] score) {
        int n = y.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Double.compare(score[a], score[b]));
        double rankSumPos = 0;
        long nPos = 0;
        int i = 0;
        while (i < n) {
            int j = i;
            while (j + 1 < n && score[order[j + 1]] == score[order[i]]) j++;
            double avgRank = (i + j) / 2.0 + 1; // hạng trung bình cho nhóm bằng nhau
            for (int k = i; k <= j; k++) if (y[order[k]] == 1) { rankSumPos += avgRank; nPos++; }
            i = j + 1;
        }
        long nNeg = n - nPos;
        if (nPos == 0 || nNeg == 0) return 0.5;
        return (rankSumPos - nPos * (nPos + 1) / 2.0) / ((double) nPos * nNeg);
    }

    // Tính các điểm của đường cong ROC
    public static double[][] rocCurve(int[] y, double[] proba) {
        int n = y.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Double.compare(proba[b], proba[a])); // giảm dần

        int totalPos = 0;
        for (int v : y) totalPos += v;
        int totalNeg = n - totalPos;

        java.util.List<double[]> points = new java.util.ArrayList<>();
        points.add(new double[]{0, 0});
        int tp = 0, fp = 0;
        for (int i = 0; i < n; i++) {
            if (y[order[i]] == 1) tp++; else fp++;
            // thêm 1 điểm khi sang giá trị xác suất mới
            boolean lastOfGroup = (i == n - 1) || proba[order[i + 1]] != proba[order[i]];
            if (lastOfGroup) points.add(new double[]{(double) fp / totalNeg, (double) tp / totalPos});
        }
        return points.toArray(new double[0][]);
    }

    public String confusionMatrix() {
        return String.format("""
                                 Dự đoán 0   Dự đoán 1
                    Thực tế 0   %9d   %9d
                    Thực tế 1   %9d   %9d""", tn, fp, fn, tp);
    }

    @Override
    public String toString() {
        return String.format("Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f  AUC=%.4f", accuracy, precision, recall, f1, rocAuc);
    }
}
