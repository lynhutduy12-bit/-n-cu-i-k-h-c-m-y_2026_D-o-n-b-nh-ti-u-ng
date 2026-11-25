package nhom7.diabetes.report;

import nhom7.diabetes.evaluation.Metrics;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;

/**
 * Vẽ biểu đồ ra file PNG bằng java.awt: đường cong ROC, ma trận nhầm lẫn,
 * so sánh chỉ số, phân tích ngưỡng, độ quan trọng đặc trưng.
 */
public final class ChartDrawer {

    private ChartDrawer() {}

    // Màu cho 4 mô hình (LR, DT, KNN, RF)
    private static final Color[] COLORS = {
            new Color(31, 119, 180),   // xanh dương
            new Color(255, 127, 14),   // cam
            new Color(44, 160, 44),    // xanh lá
            new Color(214, 39, 40),    // đỏ
            new Color(148, 103, 189)   // tím
    };

    private static final Font TITLE = new Font("SansSerif", Font.BOLD, 20);
    private static final Font LABEL = new Font("SansSerif", Font.PLAIN, 15);
    private static final Font SMALL = new Font("SansSerif", Font.PLAIN, 13);

    private static final Font NOTE = new Font("SansSerif", Font.ITALIC, 12);
    private static final Color NOTE_COLOR = new Color(90, 90, 90);

    // Bảng nghĩa tiếng Việt cho các từ tiếng Anh xuất hiện trên hình
    private static final java.util.Map<String, String> VI = java.util.Map.ofEntries(
            // 13 đặc trưng đầu vào
            java.util.Map.entry("hba1c_level", "Chỉ số HbA1c"),
            java.util.Map.entry("blood_glucose_level", "Mức đường huyết"),
            java.util.Map.entry("age", "Tuổi"),
            java.util.Map.entry("bmi", "Chỉ số khối cơ thể"),
            java.util.Map.entry("hypertension", "Tăng huyết áp"),
            java.util.Map.entry("heart_disease", "Bệnh tim"),
            java.util.Map.entry("gender_male", "Giới tính nam"),
            java.util.Map.entry("smoking_no_info", "Hút thuốc: không có thông tin"),
            java.util.Map.entry("smoking_never", "Chưa từng hút thuốc"),
            java.util.Map.entry("smoking_former", "Trước đây hút, nay đã bỏ"),
            java.util.Map.entry("smoking_current", "Đang hút thuốc"),
            java.util.Map.entry("smoking_not_current", "Hiện tại không hút"),
            java.util.Map.entry("smoking_ever", "Đã từng hút thuốc"),
            // chỉ số đánh giá
            java.util.Map.entry("Accuracy", "Độ chính xác"),
            java.util.Map.entry("Precision", "Độ chuẩn xác"),
            java.util.Map.entry("Recall", "Độ nhạy"),
            java.util.Map.entry("F1-score", "Cân bằng P và R"),
            java.util.Map.entry("ROC-AUC", "Diện tích dưới ROC"),
            // ô ma trận nhầm lẫn
            java.util.Map.entry("TN", "Âm tính thật"),
            java.util.Map.entry("FP", "Dương tính giả"),
            java.util.Map.entry("FN", "Âm tính giả"),
            java.util.Map.entry("TP", "Dương tính thật")
    );

    // Trả về "(nghĩa tiếng Việt)", hoặc chuỗi rỗng nếu không có trong bảng
    private static String vi(String english) {
        String v = VI.get(english);
        return v == null ? "" : "(" + v + ")";
    }

    // Vùng vẽ (plot area) dùng chung cho các biểu đồ dạng trục toạ độ
    private static final int LEFT = 90, TOP = 70, PLOT_W = 620, PLOT_H = 460;

    // ---------- 1. ĐƯỜNG CONG ROC ----------

    public static void rocChart(List<String> names, List<double[][]> curves, List<Double> aucs, String file) throws Exception {
        BufferedImage img = new BufferedImage(1000, 620, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = start(img);
        drawTitle(g, "Đường cong ROC trên tập test", img.getWidth());
        drawAxes(g, "False Positive Rate - FPR", "(Tỷ lệ dương tính giả)",
                "True Positive Rate - TPR = Recall", "(Tỷ lệ dương tính thật = Độ nhạy)");

        // đường chéo = mô hình đoán ngẫu nhiên (AUC = 0.5)
        g.setColor(Color.GRAY);
        g.setStroke(new BasicStroke(1.5f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10, new float[]{6, 6}, 0));
        g.drawLine(px(0), py(0), px(1), py(1));

        g.setStroke(new BasicStroke(2.5f));
        for (int m = 0; m < names.size(); m++) {
            g.setColor(COLORS[m]);
            double[][] pts = curves.get(m);
            for (int i = 1; i < pts.length; i++) {
                g.drawLine(px(pts[i - 1][0]), py(pts[i - 1][1]), px(pts[i][0]), py(pts[i][1]));
            }
        }

        // chú thích (legend)
        String[] labels = new String[names.size() + 1];
        for (int m = 0; m < names.size(); m++) labels[m] = String.format("%s (AUC = %.4f)", names.get(m), aucs.get(m));
        labels[names.size()] = "Đoán ngẫu nhiên (AUC = 0.5)";
        Color[] colors = new Color[labels.length];
        System.arraycopy(COLORS, 0, colors, 0, names.size());
        colors[names.size()] = Color.GRAY;
        drawLegend(g, labels, colors, LEFT + PLOT_W + 20, TOP + 20);

        save(img, file);
    }

    // ---------- 2. MA TRẬN NHẦM LẪN ----------

    public static void confusionMatrixChart(List<String> names, List<Metrics> metrics, String file) throws Exception {
        int cell = 130;                 // kích thước 1 ô
        int blockW = 420, blockH = 380; // kích thước 1 khối ma trận (kèm nhãn)
        BufferedImage img = new BufferedImage(blockW * 2 + 40, blockH * 2 + 80, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = start(img);
        drawTitle(g, "Ma trận nhầm lẫn trên tập test (ngưỡng 0.5)", img.getWidth());

        for (int m = 0; m < names.size(); m++) {
            int ox = 20 + (m % 2) * blockW + 110;   // góc trên-trái của ma trận
            int oy = 70 + (m / 2) * blockH + 50;
            Metrics mt = metrics.get(m);
            int[][] v = {{mt.tn, mt.fp}, {mt.fn, mt.tp}};
            int total = mt.tn + mt.fp + mt.fn + mt.tp;

            g.setFont(LABEL.deriveFont(Font.BOLD));
            g.setColor(Color.BLACK);
            centerText(g, names.get(m), ox + cell, oy - 25);

            for (int r = 0; r < 2; r++) {
                for (int c = 0; c < 2; c++) {
                    // ô càng nhiều mẫu thì màu càng đậm
                    double t = Math.log(1 + v[r][c]) / Math.log(1 + total);
                    int shade = (int) (255 - 200 * t);
                    g.setColor(new Color(shade, shade + (255 - shade) / 3, 255));
                    g.fillRect(ox + c * cell, oy + r * cell, cell, cell);
                    g.setColor(Color.DARK_GRAY);
                    g.drawRect(ox + c * cell, oy + r * cell, cell, cell);

                    g.setColor(t > 0.6 ? Color.WHITE : Color.BLACK);
                    g.setFont(TITLE);
                    int cx = ox + c * cell + cell / 2, cy = oy + r * cell + cell / 2;
                    centerText(g, String.format("%,d", v[r][c]), cx, cy - 12);
                    g.setFont(SMALL);
                    String tag = new String[][]{{"TN", "FP"}, {"FN", "TP"}}[r][c];
                    centerText(g, tag, cx, cy + 14);
                    g.setFont(NOTE);
                    centerText(g, vi(tag), cx, cy + 32);   // vd: (Âm tính thật)
                }
            }
            g.setColor(Color.BLACK);
            g.setFont(SMALL);
            centerText(g, "Dự đoán 0", ox + cell / 2, oy + 2 * cell + 20);
            centerText(g, "Dự đoán 1", ox + cell + cell / 2, oy + 2 * cell + 20);
            rightText(g, "Thực tế 0", ox - 8, oy + cell / 2);
            rightText(g, "Thực tế 1", ox - 8, oy + cell + cell / 2);
        }
        save(img, file);
    }

    // ---------- 3. SO SÁNH CÁC CHỈ SỐ ----------

    public static void metricBarChart(List<String> names, List<Metrics> metrics, String file) throws Exception {
        String[] metricNames = {"Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"};
        BufferedImage img = new BufferedImage(1000, 620, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = start(img);
        drawTitle(g, "So sánh các chỉ số đánh giá trên tập test", img.getWidth());
        drawAxes(g, null, null, "Giá trị", null);

        int groupW = PLOT_W / metricNames.length;   // độ rộng 1 nhóm cột
        int barW = (groupW - 20) / names.size();
        for (int k = 0; k < metricNames.length; k++) {
            int gx = LEFT + k * groupW + 10;
            for (int m = 0; m < names.size(); m++) {
                Metrics mt = metrics.get(m);
                double val = new double[]{mt.accuracy, mt.precision, mt.recall, mt.f1, mt.rocAuc}[k];
                int x = gx + m * barW;
                g.setColor(COLORS[m]);
                g.fillRect(x, py(val), barW - 2, py(0) - py(val));
                // ghi giá trị trong đầu cột
                Graphics2D g2 = (Graphics2D) g.create();
                g2.setColor(Color.WHITE);
                g2.setFont(new Font("SansSerif", Font.BOLD, 11));
                String txt = String.format("%.3f", val);
                int baseX = x + barW / 2 + 4;
                int baseY = py(val) + g2.getFontMetrics().stringWidth(txt) + 6;
                g2.rotate(-Math.PI / 2, baseX, baseY);
                g2.drawString(txt, baseX, baseY);
                g2.dispose();
            }
            g.setColor(Color.BLACK);
            g.setFont(LABEL);
            centerText(g, metricNames[k], gx + (groupW - 20) / 2, py(0) + 22);
            g.setFont(NOTE);
            g.setColor(NOTE_COLOR);
            centerText(g, vi(metricNames[k]), gx + (groupW - 20) / 2, py(0) + 42);   // vd: (Độ nhạy)
        }
        drawLegend(g, names.toArray(new String[0]), COLORS, LEFT + PLOT_W + 20, TOP + 20);
        save(img, file);
    }

    // ---------- 4. PHÂN TÍCH NGƯỠNG ----------

    public static void thresholdChart(String model, double[] thresholds, double[] prec, double[] rec, double[] f1,
                                      double chosen, String file) throws Exception {
        BufferedImage img = new BufferedImage(1000, 620, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = start(img);
        drawTitle(g, "Precision / Recall / F1 theo ngưỡng - " + model + " (5-Fold CV trên tập train)", img.getWidth());
        drawAxes(g, "Ngưỡng xác suất", null, "Giá trị", null);

        double[][] series = {prec, rec, f1};
        g.setStroke(new BasicStroke(2.5f));
        for (int s = 0; s < 3; s++) {
            g.setColor(COLORS[s]);
            for (int i = 1; i < thresholds.length; i++) {
                g.drawLine(px(thresholds[i - 1]), py(series[s][i - 1]), px(thresholds[i]), py(series[s][i]));
            }
        }
        // vạch đứng đánh dấu ngưỡng 0.5 (mặc định) và ngưỡng được chọn
        g.setStroke(new BasicStroke(1.5f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10, new float[]{6, 6}, 0));
        g.setColor(Color.GRAY);
        g.drawLine(px(0.5), py(0), px(0.5), py(1));
        g.setColor(COLORS[3]);
        g.drawLine(px(chosen), py(0), px(chosen), py(1));

        drawLegend(g, new String[]{"Precision", "Recall", "F1-score", "Ngưỡng mặc định 0.5",
                        String.format("Ngưỡng chọn = %.2f", chosen)},
                new Color[]{COLORS[0], COLORS[1], COLORS[2], Color.GRAY, COLORS[3]}, LEFT + PLOT_W + 20, TOP + 20);
        save(img, file);
    }

    // ---------- 5. ĐỘ QUAN TRỌNG ĐẶC TRƯNG ----------

    public static void featureImportanceChart(String model, String[] features, double[] importance, String file) throws Exception {
        // sắp xếp giảm dần
        Integer[] order = new Integer[features.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        java.util.Arrays.sort(order, (a, b) -> Double.compare(importance[b], importance[a]));
        double max = importance[order[0]];

        int rowH = 46, left = 250, width = 580;
        BufferedImage img = new BufferedImage(980, 100 + rowH * features.length, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = start(img);
        drawTitle(g, "Độ quan trọng đặc trưng - " + model, img.getWidth());

        for (int i = 0; i < order.length; i++) {
            int y = 70 + i * rowH;
            int w = (int) (width * importance[order[i]] / max);
            g.setColor(COLORS[0]);
            g.fillRect(left, y, Math.max(w, 1), rowH - 8);
            g.setColor(Color.BLACK);
            g.setFont(LABEL);
            int mid = y + (rowH - 8) / 2;
            rightText(g, features[order[i]], left - 10, mid - 8);
            g.setFont(NOTE);
            g.setColor(NOTE_COLOR);
            rightText(g, vi(features[order[i]]), left - 10, mid + 9);   // vd: (Chỉ số HbA1c)
            g.setFont(LABEL);
            g.setColor(Color.BLACK);
            g.drawString(String.format("%.4f", importance[order[i]]), left + w + 8, mid + 6);
        }
        save(img, file);
    }

    // ---------- Hàm hỗ trợ ----------

    // Đổi toạ độ dữ liệu x (0..1) sang pixel
    private static int px(double x) { return LEFT + (int) Math.round(x * PLOT_W); }

    // Đổi toạ độ dữ liệu y (0..1) sang pixel (trục y hướng lên)
    private static int py(double y) { return TOP + PLOT_H - (int) Math.round(y * PLOT_H); }

    private static Graphics2D start(BufferedImage img) {
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, img.getWidth(), img.getHeight());
        return g;
    }

    private static void drawTitle(Graphics2D g, String title, int width) {
        g.setColor(Color.BLACK);
        g.setFont(TITLE);
        centerText(g, title, width / 2, 32);
    }

    // Vẽ khung trục, lưới và vạch chia
    private static void drawAxes(Graphics2D g, String xLabel, String xNote, String yLabel, String yNote) {
        g.setFont(SMALL);
        for (int i = 0; i <= 10; i++) {
            double v = i / 10.0;
            g.setColor(new Color(230, 230, 230));
            g.drawLine(px(0), py(v), px(1), py(v));                  // lưới ngang
            if (xLabel != null) g.drawLine(px(v), py(0), px(v), py(1)); // lưới dọc
            g.setColor(Color.BLACK);
            rightText(g, String.format("%.1f", v), px(0) - 8, py(v));
            if (xLabel != null) centerText(g, String.format("%.1f", v), px(v), py(0) + 18);
        }
        g.setColor(Color.BLACK);
        g.setStroke(new BasicStroke(1.5f));
        g.drawRect(LEFT, TOP, PLOT_W, PLOT_H);

        g.setFont(LABEL);
        if (xLabel != null) centerText(g, xLabel, LEFT + PLOT_W / 2, py(0) + 42);
        if (xNote != null) {                        // nghĩa tiếng Việt ở dòng dưới
            g.setFont(NOTE);
            g.setColor(NOTE_COLOR);
            centerText(g, xNote, LEFT + PLOT_W / 2, py(0) + 62);
            g.setFont(LABEL);
            g.setColor(Color.BLACK);
        }
        Graphics2D g2 = (Graphics2D) g.create();   // nhãn trục y xoay dọc
        g2.rotate(-Math.PI / 2);
        FontMetrics fm = g2.getFontMetrics();
        int yLine = yNote == null ? 30 : 24;
        g2.drawString(yLabel, -(TOP + PLOT_H / 2) - fm.stringWidth(yLabel) / 2, yLine);
        if (yNote != null) {
            g2.setFont(NOTE);
            g2.setColor(NOTE_COLOR);
            fm = g2.getFontMetrics();
            g2.drawString(yNote, -(TOP + PLOT_H / 2) - fm.stringWidth(yNote) / 2, yLine + 18);
        }
        g2.dispose();
    }

    private static void drawLegend(Graphics2D g, String[] labels, Color[] colors, int x, int y) {
        for (int i = 0; i < labels.length; i++) {
            g.setColor(colors[i]);
            g.fillRect(x, y, 22, 12);
            g.setColor(Color.BLACK);
            g.setFont(SMALL);
            g.drawString(labels[i], x + 30, y + 12);
            String note = vi(labels[i]);
            if (!note.isEmpty()) {                 // nghĩa tiếng Việt ở dòng dưới
                g.setFont(NOTE);
                g.setColor(NOTE_COLOR);
                g.drawString(note, x + 30, y + 28);
                y += 16;
            }
            y += 26;
        }
    }

    private static void centerText(Graphics2D g, String s, int cx, int cy) {
        FontMetrics fm = g.getFontMetrics();
        g.drawString(s, cx - fm.stringWidth(s) / 2, cy + fm.getAscent() / 2 - 2);
    }

    private static void rightText(Graphics2D g, String s, int rx, int cy) {
        FontMetrics fm = g.getFontMetrics();
        g.drawString(s, rx - fm.stringWidth(s), cy + fm.getAscent() / 2 - 2);
    }

    private static void save(BufferedImage img, String file) throws Exception {
        File f = new File(file);
        f.getAbsoluteFile().getParentFile().mkdirs();
        ImageIO.write(img, "png", f);
        System.out.println("   Đã lưu biểu đồ: " + file);
    }
}
