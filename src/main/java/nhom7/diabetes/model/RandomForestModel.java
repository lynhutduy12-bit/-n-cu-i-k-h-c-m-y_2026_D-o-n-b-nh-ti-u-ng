package nhom7.diabetes.model;

import java.util.Random;
import java.util.stream.IntStream;

/**
 * Rừng ngẫu nhiên (Random Forest): gồm nhiều cây quyết định.
 * Mỗi cây học trên mẫu bootstrap (Bagging) và chọn ngẫu nhiên một phần đặc trưng khi chia nút;
 * xác suất dự đoán = trung bình xác suất của các cây.
 */
public class RandomForestModel implements Classifier {

    private final int nEstimators;
    private final int maxDepth;
    private final int minSamplesLeaf;
    private final String maxFeatures;
    private final long seed;

    private DecisionTreeModel[] trees;
    private double[] importance;

    public RandomForestModel() {
        this(100, 0, 1, "sqrt", 42L);
    }

    public RandomForestModel(int nEstimators, int maxDepth, int minSamplesLeaf, String maxFeatures) {
        this(nEstimators, maxDepth, minSamplesLeaf, maxFeatures, 42L);
    }

    public RandomForestModel(int nEstimators, int maxDepth, int minSamplesLeaf, String maxFeatures, long seed) {
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.minSamplesLeaf = minSamplesLeaf;
        this.maxFeatures = maxFeatures;
        this.seed = seed;
    }

    private int resolveMaxFeatures(int d) {
        return switch (maxFeatures) {
            case "sqrt" -> Math.max(1, (int) Math.sqrt(d));
            case "log2" -> Math.max(1, (int) (Math.log(d) / Math.log(2)));
            default -> Math.max(1, (int) Math.round(Double.parseDouble(maxFeatures) * d));
        };
    }

    @Override
    public void fit(double[][] X, int[] y) {
        int n = X.length, d = X[0].length;
        int mf = resolveMaxFeatures(d);
        FeatureBinner binner = FeatureBinner.fit(X, 255);
        short[][] bins = binner.transform(X);
        trees = new DecisionTreeModel[nEstimators];

        IntStream.range(0, nEstimators).parallel().forEach(t -> {
            Random rnd = new Random(seed * 1_000_003L + t);
            int[] sample = new int[n];
            for (int i = 0; i < n; i++) sample[i] = rnd.nextInt(n); 
            DecisionTreeModel tree = new DecisionTreeModel(maxDepth, 2, minSamplesLeaf, "gini", mf, seed + 7919L * t);
            tree.fitBinned(binner, bins, y, sample);
            trees[t] = tree;
        });

        importance = new double[d];
        for (DecisionTreeModel tr : trees) {
            double[] imp = tr.featureImportance();
            for (int j = 0; j < d; j++) importance[j] += imp[j] / nEstimators;
        }
    }

    @Override
    public double[] predictProba(double[][] X) {
        double[] p = new double[X.length];
        IntStream.range(0, X.length).parallel().forEach(i -> {
            double s = 0;
            for (DecisionTreeModel tr : trees) s += tr.predictOne(X[i]);
            p[i] = s / trees.length;
        });
        return p;
    }

    @Override
    public double[] featureImportance() { return importance == null ? null : importance.clone(); }

    @Override
    public String name() { return "Random Forest"; }

    @Override
    public String params() {
        return "n_estimators=" + nEstimators + ", max_depth=" + (maxDepth > 0 ? maxDepth : "None")
                + ", min_samples_leaf=" + minSamplesLeaf + ", max_features=" + maxFeatures;
    }
}
