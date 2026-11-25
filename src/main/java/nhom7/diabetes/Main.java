package nhom7.diabetes;

import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashSet;
import java.util.Scanner;
import java.util.Set;

import nhom7.diabetes.collection.DataCollector;
import nhom7.diabetes.db.DatabaseManager;
import nhom7.diabetes.evaluation.ModelEvaluator;
import nhom7.diabetes.model.ModelTrainer;
import nhom7.diabetes.preprocessing.DataPreprocessor;
import nhom7.diabetes.report.ReportExporter;
import nhom7.diabetes.tuning.HyperparameterTuner;
import nhom7.diabetes.web.WebServer;

/**
 * Chương trình chính - hiển thị menu:
 *   1. Chạy toàn bộ chương trình dự đoán (thu thập -> tiền xử lý -> huấn luyện -> tinh chỉnh -> đánh giá -> xuất kết quả)
 *   2. Chạy website dự đoán
 */
public class Main {

    public static void main(String[] args) throws Exception {
        System.setOut(new PrintStream(System.out, true, StandardCharsets.UTF_8));
        banner();
        DatabaseManager.initSchema();

        if (args.length > 0) {
            String models = args.length > 1 ? args[1] : "lr,dt,knn,rf";
            runCommand(args[0].toLowerCase(), models);
            return;
        }

        Scanner sc = new Scanner(System.in, StandardCharsets.UTF_8);
        while (true) {
            System.out.println("""

                    ============================ MENU ============================
                      1. Chạy chương trình dự đoán 
                      2. Chạy website dự đoán (http://localhost:8080)
                      0. Thoát
                    ==============================================================""");
            System.out.print("Chọn: ");
            if (!sc.hasNextLine()) return;
            String choice = sc.nextLine().trim();
            try {
                switch (choice) {
                    case "1" -> runCommand("all", "lr,dt,knn,rf");   
                    case "2" -> runCommand("web", null);    
                    case "0" -> { return; }
                    default -> System.out.println("Lựa chọn không hợp lệ. Hãy nhập 1, 2 hoặc 0.");
                }
            } catch (Exception e) {
                System.out.println("LỖI: " + e.getMessage());
            }
        }
    }

    private static void runCommand(String cmd, String models) throws Exception {
        long t0 = System.currentTimeMillis();
        Set<String> which = new LinkedHashSet<>();
        if (models != null) for (String m : models.split(",")) which.add(m.trim().toLowerCase());
        switch (cmd) {
            case "collect" -> new DataCollector().run();
            case "preprocess" -> new DataPreprocessor().run();
            case "train" -> new ModelTrainer().run();
            case "tune" -> new HyperparameterTuner().run(which);
            case "evaluate" -> new ModelEvaluator().run();
            case "export" -> new ReportExporter().run();
            case "web" -> new WebServer().run();
            case "all" -> {
                new DataCollector().run();
                new DataPreprocessor().run();
                new ModelTrainer().run();
                new HyperparameterTuner().run(which);
                new ModelEvaluator().run();
                new ReportExporter().run();
            }
            default -> {
                System.out.println("Lệnh không hợp lệ: " + cmd + " (collect | preprocess | train | tune | evaluate | export | all | web)");
                return;
            }
        }
        System.out.printf("%nHoàn tất '%s' trong %.1f giây.%n", cmd, (System.currentTimeMillis() - t0) / 1000.0);
    }

    private static void banner() {
        System.out.println("""
                ================================================================
                  XÂY DỰNG MÔ HÌNH DỰ ĐOÁN NGUY CƠ MẮC BỆNH TIỂU ĐƯỜNG - NHÓM 7
                  Môn: Các vấn đề hiện đại trong Trí tuệ nhân tạo
                  GVHD: TS. Phùng Thế Bảo
                ================================================================""");
        System.out.printf("  CPU: %d nhân | Java %s%n", Runtime.getRuntime().availableProcessors(), System.getProperty("java.version"));
    }
}