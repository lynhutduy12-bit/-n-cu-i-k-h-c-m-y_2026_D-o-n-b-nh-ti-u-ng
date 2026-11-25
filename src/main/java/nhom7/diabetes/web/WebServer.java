package nhom7.diabetes.web;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import nhom7.diabetes.config.AppConfig;
import nhom7.diabetes.db.DatabaseManager;
import nhom7.diabetes.model.SavedModel;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;

/**
 * Website dự đoán (dùng HttpServer có sẵn trong JDK):
 *   GET  /               -> giao diện (thư mục web/)
 *   POST /api/predict    -> dự đoán
 *   GET  /api/history    -> lịch sử dự đoán
 *   GET  /api/model-info -> thông tin mô hình
 */
public class WebServer {

    private static final int PORT = Integer.getInteger("port", 8080);
    private static final String WEB_DIR = System.getProperty("web.dir", "web");

    private PredictionService service;

    public void run() throws Exception {
        System.out.println("\n========== BƯỚC 6: WEBSITE DỰ ĐOÁN NGUY CƠ TIỂU ĐƯỜNG ==========");
        String modelFile = AppConfig.OUTPUT_DIR + "/best_model.ser";
        if (!new File(modelFile).exists()) {
            throw new IllegalStateException("Chưa có file " + modelFile + ". Hãy chạy Bước 5 (Đánh giá & chọn mô hình) trước.");
        }
        SavedModel saved = SavedModel.load(modelFile);
        service = new PredictionService(saved);
        DatabaseManager.initSchema();
        PredictionService.createHistoryTable();
        System.out.printf("Đã nạp mô hình: %s [%s], ngưỡng = %.2f%n", saved.model.name(), saved.model.params(), saved.threshold);

        HttpServer server = HttpServer.create(new InetSocketAddress(PORT), 0);
        server.createContext("/api/predict", this::handlePredict);
        server.createContext("/api/history", this::handleHistory);
        server.createContext("/api/model-info", this::handleModelInfo);
        server.createContext("/", this::handleStatic);
        server.setExecutor(Executors.newFixedThreadPool(4));
        server.start();

        System.out.println("Website đang chạy tại: http://localhost:" + PORT);
        System.out.println("Mở địa chỉ trên bằng trình duyệt. Nhấn Ctrl + C (hoặc đóng cửa sổ) để dừng.");
        Thread.currentThread().join();   // giữ chương trình chạy cho đến khi người dùng tắt
    }

    // ---------- API ----------

    // POST /api/predict
    private void handlePredict(HttpExchange ex) throws IOException {
        if (!ex.getRequestMethod().equals("POST")) {
            sendJson(ex, 405, "{\"error\":\"Chỉ hỗ trợ POST\"}");
            return;
        }
        try {
            String body = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            PredictionService.Result r = service.predict(parseForm(body));

            StringBuilder json = new StringBuilder("{");
            json.append("\"probability\":").append(String.format(java.util.Locale.US, "%.4f", r.probability)).append(',');
            json.append("\"threshold\":").append(String.format(java.util.Locale.US, "%.2f", service.getSavedModel().threshold)).append(',');
            json.append("\"atRisk\":").append(r.atRisk).append(',');
            json.append("\"level\":").append(q(r.level)).append(',');
            json.append("\"message\":").append(q(r.message)).append(',');
            json.append("\"indicators\":[");
            for (int i = 0; i < r.indicators.size(); i++) {
                String[] it = r.indicators.get(i);
                if (i > 0) json.append(',');
                json.append("{\"name\":").append(q(it[0])).append(",\"value\":").append(q(it[1]))
                        .append(",\"note\":").append(q(it[2])).append(",\"status\":").append(q(it[3])).append('}');
            }
            json.append("],\"warnings\":[");
            for (int i = 0; i < r.warnings.size(); i++) json.append(i > 0 ? "," : "").append(q(r.warnings.get(i)));
            json.append("]}");

            System.out.printf("[Dự đoán] xác suất = %.4f -> %s%n", r.probability, r.atRisk ? "CÓ nguy cơ" : "không");
            sendJson(ex, 200, json.toString());
        } catch (PredictionService.InputException e) {
            sendJson(ex, 400, "{\"error\":" + q(e.getMessage()) + "}");   // lỗi do người dùng nhập
        } catch (Exception e) {
            e.printStackTrace();
            sendJson(ex, 500, "{\"error\":" + q("Lỗi máy chủ: " + e.getMessage()) + "}");
        }
    }

    // GET /api/history
    private void handleHistory(HttpExchange ex) throws IOException {
        try {
            List<String[]> rows = service.recentHistory(10);
            StringBuilder json = new StringBuilder("[");
            for (int i = 0; i < rows.size(); i++) {
                if (i > 0) json.append(',');
                json.append('[');
                String[] row = rows.get(i);
                for (int j = 0; j < row.length; j++) json.append(j > 0 ? "," : "").append(q(row[j]));
                json.append(']');
            }
            sendJson(ex, 200, json.append(']').toString());
        } catch (Exception e) {
            sendJson(ex, 500, "{\"error\":" + q(e.getMessage()) + "}");
        }
    }

    // GET /api/model-info
    private void handleModelInfo(HttpExchange ex) throws IOException {
        SavedModel sm = service.getSavedModel();
        StringBuilder json = new StringBuilder("{");
        json.append("\"model\":").append(q(sm.model.name())).append(',');
        json.append("\"params\":").append(q(sm.model.params())).append(',');
        json.append("\"threshold\":").append(String.format(java.util.Locale.US, "%.2f", sm.threshold));
        String sql = "SELECT accuracy, precision_, recall, f1, roc_auc FROM model_results "
                + "WHERE stage='tuned' AND model=? ORDER BY id DESC LIMIT 1";
        try (Connection c = DatabaseManager.getConnection(); PreparedStatement ps = c.prepareStatement(sql)) {
            ps.setString(1, sm.model.name());
            try (ResultSet rs = ps.executeQuery()) {
                if (rs.next()) {
                    json.append(String.format(java.util.Locale.US,
                            ",\"accuracy\":%.4f,\"precision\":%.4f,\"recall\":%.4f,\"f1\":%.4f,\"auc\":%.4f",
                            rs.getDouble(1), rs.getDouble(2), rs.getDouble(3), rs.getDouble(4), rs.getDouble(5)));
                }
            }
        } catch (Exception ignored) {
            // không có số liệu thì chỉ trả về tên mô hình
        }
        sendJson(ex, 200, json.append('}').toString());
    }

    // ---------- File tĩnh (HTML, CSS, JS) ----------

    private void handleStatic(HttpExchange ex) throws IOException {
        String path = ex.getRequestURI().getPath();
        if (path.equals("/")) path = "/index.html";
        Path base = Path.of(WEB_DIR).toAbsolutePath().normalize();
        Path file = base.resolve(path.substring(1)).normalize();
        // chặn truy cập ra ngoài thư mục web/ (vd: /../data/diabetes.db)
        if (!file.startsWith(base) || !Files.isRegularFile(file)) {
            send(ex, 404, "text/plain; charset=utf-8", "404 - Không tìm thấy trang".getBytes(StandardCharsets.UTF_8));
            return;
        }
        String name = file.getFileName().toString();
        String type = name.endsWith(".html") ? "text/html; charset=utf-8"
                : name.endsWith(".css") ? "text/css; charset=utf-8"
                : name.endsWith(".js") ? "application/javascript; charset=utf-8"
                : name.endsWith(".png") ? "image/png"
                : "application/octet-stream";
        send(ex, 200, type, Files.readAllBytes(file));
    }

    // ---------- Hàm hỗ trợ ----------

    // Đọc body dạng "a=1&b=2" thành Map
    private static Map<String, String> parseForm(String body) {
        Map<String, String> map = new LinkedHashMap<>();
        for (String pair : body.split("&")) {
            if (pair.isEmpty()) continue;
            String[] kv = pair.split("=", 2);
            String k = URLDecoder.decode(kv[0], StandardCharsets.UTF_8);
            String v = kv.length > 1 ? URLDecoder.decode(kv[1], StandardCharsets.UTF_8) : "";
            map.put(k, v);
        }
        return map;
    }

    // Đặt chuỗi trong dấu nháy kép theo chuẩn JSON (thoát các ký tự đặc biệt)
    private static String q(String s) {
        StringBuilder sb = new StringBuilder("\"");
        for (char ch : s.toCharArray()) {
            switch (ch) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> {
                    if (ch < 0x20) sb.append(String.format("\\u%04x", (int) ch));
                    else sb.append(ch);
                }
            }
        }
        return sb.append('"').toString();
    }

    private static void sendJson(HttpExchange ex, int status, String json) throws IOException {
        send(ex, status, "application/json; charset=utf-8", json.getBytes(StandardCharsets.UTF_8));
    }

    private static void send(HttpExchange ex, int status, String contentType, byte[] data) throws IOException {
        ex.getResponseHeaders().set("Content-Type", contentType);
        ex.sendResponseHeaders(status, data.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(data);
        }
    }
}
