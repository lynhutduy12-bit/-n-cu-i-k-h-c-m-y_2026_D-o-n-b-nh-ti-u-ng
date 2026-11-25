package nhom7.diabetes.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


/**
 * Cây quyết định (Decision Tree), tiêu chí chia nút Gini hoặc Entropy.
 * Giá trị đặc trưng được chia thành tối đa 255 khoảng (bin) để tìm điểm chia nhanh hơn.
 */
public class DecisionTreeModel implements Classifier {

    private static final int MAX_BINS = 255;
    private static final int HARD_DEPTH_LIMIT = 64;

    private final int maxDepth;
    private final int minSamplesSplit;
    private final int minSamplesLeaf;
    private final String criterion;
    private final int maxFeatures;
    private final long seed;

    private final List<Node> nodes = new ArrayList<>();
    private double[] importance;

    private short[][] bins;
    private int[] yTrain;
    private FeatureBinner binner;
    private Random rnd;

    private static final class Node implements java.io.Serializable {
        int feature = -1;     
        double threshold;
        int left, right;
        double value;
    }

    public DecisionTreeModel() {
        this(0, 2, 1, "gini", 0, 42L);
    }

    public DecisionTreeModel(int maxDepth, int minSamplesSplit, int minSamplesLeaf, String criterion) {
        this(maxDepth, minSamplesSplit, minSamplesLeaf, criterion, 0, 42L);
    }

    public DecisionTreeModel(int maxDepth, int minSamplesSplit, int minSamplesLeaf, String criterion, int maxFeatures, long seed) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = Math.max(2, minSamplesSplit);
        this.minSamplesLeaf = Math.max(1, minSamplesLeaf);
        this.criterion = criterion;
        this.maxFeatures = maxFeatures;
        this.seed = seed;
    }

    @Override
    public void fit(double[][] X, int[] y) {
        FeatureBinner fb = FeatureBinner.fit(X, MAX_BINS);
        int[] idx = new int[y.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        fitBinned(fb, fb.transform(X), y, idx);
    }


    void fitBinned(FeatureBinner binner, short[][] bins, int[] y, int[] sampleIdx) {
        this.binner = binner;
        this.bins = bins;
        this.yTrain = y;
        this.rnd = new Random(seed);
        this.nodes.clear();
        this.importance = new double[bins.length];
        int[] idx = sampleIdx.clone();
        build(idx, 0, idx.length, 0);
        double s = 0;
        for (double v : importance) s += v;
        if (s > 0) for (int j = 0; j < importance.length; j++) importance[j] /= s;
        // giải phóng bộ nhớ tạm
        this.bins = null;
        this.yTrain = null;
        this.binner = null;
    }

    private int build(int[] idx, int start, int end, int depth) {
        int n = end - start;
        int pos = 0;
        for (int i = start; i < end; i++) pos += yTrain[idx[i]];

        Node node = new Node();
        node.value = (double) pos / n;
        int id = nodes.size();
        nodes.add(node);

        int depthLimit = maxDepth > 0 ? maxDepth : HARD_DEPTH_LIMIT;
        if (pos == 0 || pos == n || n < minSamplesSplit || n < 2 * minSamplesLeaf || depth >= depthLimit) {
            return id;
        }

        double parentImp = impurity(pos, n);
        int d = bins.length;
        int[] features = candidateFeatures(d);

        double bestScore = Double.MAX_VALUE;
        int bestF = -1, bestB = -1;
        for (int f : features) {
            int nb = binner.numBins(f);
            if (nb < 2) continue;
            int[] c0 = new int[nb], c1 = new int[nb];
            short[] col = bins[f];
            for (int i = start; i < end; i++) {
                int s = idx[i];
                if (yTrain[s] == 1) c1[col[s]]++; else c0[col[s]]++;
            }
            int l0 = 0, l1 = 0;
            for (int b = 0; b < nb - 1; b++) {
                l0 += c0[b];
                l1 += c1[b];
                if (c0[b] + c1[b] == 0) continue;
                int nl = l0 + l1, nr = n - nl;
                if (nr < minSamplesLeaf) break;
                if (nl < minSamplesLeaf) continue;
                double score = nl * impurity(l1, nl) + nr * impurity(pos - l1, nr);
                if (score < bestScore - 1e-12) {
                    bestScore = score;
                    bestF = f;
                    bestB = b;
                }
            }
        }
        if (bestF < 0 || bestScore >= n * parentImp - 1e-12) return id;

        short[] col = bins[bestF];
        int i = start, j = end - 1;
        while (i <= j) {
            if (col[idx[i]] <= bestB) i++;
            else { int t = idx[i]; idx[i] = idx[j]; idx[j] = t; j--; }
        }
        int mid = i;
        if (mid == start || mid == end) return id;

        importance[bestF] += n * parentImp - bestScore;
        node.feature = bestF;
        node.threshold = binner.threshold(bestF, bestB);
        node.left = build(idx, start, mid, depth + 1);
        node.right = build(idx, mid, end, depth + 1);
        return id;
    }

    private int[] candidateFeatures(int d) {
        int k = (maxFeatures <= 0 || maxFeatures >= d) ? d : maxFeatures;
        int[] all = new int[d];
        for (int i = 0; i < d; i++) all[i] = i;
        if (k == d) return all;
        for (int i = 0; i < k; i++) { 
            int j = i + rnd.nextInt(d - i);
            int t = all[i]; all[i] = all[j]; all[j] = t;
        }
        return java.util.Arrays.copyOf(all, k);
    }

    private double impurity(int pos, int n) {
        if (n == 0) return 0;
        double p1 = (double) pos / n, p0 = 1 - p1;
        if ("entropy".equals(criterion)) {
            double h = 0;
            if (p1 > 0) h -= p1 * Math.log(p1) / Math.log(2);
            if (p0 > 0) h -= p0 * Math.log(p0) / Math.log(2);
            return h;
        }
        return 1 - p1 * p1 - p0 * p0;
    }

    double predictOne(double[] x) {
        Node node = nodes.get(0);
        while (node.feature >= 0) {
            node = nodes.get(x[node.feature] <= node.threshold ? node.left : node.right);
        }
        return node.value;
    }

    @Override
    public double[] predictProba(double[][] X) {
        double[] p = new double[X.length];
        for (int i = 0; i < X.length; i++) p[i] = predictOne(X[i]);
        return p;
    }

    @Override
    public double[] featureImportance() { return importance == null ? null : importance.clone(); }

    public int nodeCount() { return nodes.size(); }

    public int depth() { return depthOf(0); }

    private int depthOf(int id) {
        Node n = nodes.get(id);
        if (n.feature < 0) return 0;
        return 1 + Math.max(depthOf(n.left), depthOf(n.right));
    }

    @Override
    public String name() { return "Decision Tree"; }

    @Override
    public String params() {
        return "max_depth=" + (maxDepth > 0 ? maxDepth : "None") + ", min_samples_split=" + minSamplesSplit
                + ", min_samples_leaf=" + minSamplesLeaf + ", criterion=" + criterion;
    }
}
