package nhom7.diabetes.report;

import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.Statement;
import java.util.Locale;

import nhom7.diabetes.config.AppConfig;
import nhom7.diabetes.db.DatabaseManager;

/**
 * Xuất kết quả từ CSDL ra thư mục output/ dạng CSV:
 * data_log, cv_results, model_results, feature_importance.
 */
public class ReportExporter {

    public void run() throws Exception {
        System.out.println("\n================= XUẤT KẾT QUẢ RA THƯ MỤC " + AppConfig.OUTPUT_DIR + "/ =================");
        Path dir = Path.of(AppConfig.OUTPUT_DIR);
        Files.createDirectories(dir);
        try (Connection c = DatabaseManager.getConnection()) {
            exportCsv(c, "SELECT step,item,value FROM data_log ORDER BY id", dir.resolve("data_log.csv"));
            exportCsv(c, "SELECT model,search_type,params,k_folds,acc_mean,prec_mean,rec_mean,f1_mean,f1_std,auc_mean,time_ms,is_best "
                    + "FROM cv_results ORDER BY model, f1_mean DESC", dir.resolve("cv_results.csv"));
            exportCsv(c, "SELECT stage,model,params,accuracy,precision_ AS precision,recall,f1,roc_auc,tn,fp,fn,tp,train_ms "
                    + "FROM model_results ORDER BY stage, id", dir.resolve("model_results.csv"));
            exportCsv(c, "SELECT stage,model,feature,importance FROM feature_importance ORDER BY stage, model, importance DESC",
                    dir.resolve("feature_importance.csv"));
        }
        System.out.println("Đã xuất: data_log.csv, cv_results.csv, model_results.csv, feature_importance.csv");
    }

    private void exportCsv(Connection c, String sql, Path file) throws Exception {
        try (Statement st = c.createStatement(); ResultSet rs = st.executeQuery(sql);
             PrintWriter pw = new PrintWriter(Files.newBufferedWriter(file, StandardCharsets.UTF_8))) {
            pw.print('\uFEFF'); // BOM để Excel đọc đúng tiếng Việt
            ResultSetMetaData md = rs.getMetaData();
            int n = md.getColumnCount();
            for (int i = 1; i <= n; i++) pw.print((i > 1 ? "," : "") + md.getColumnLabel(i));
            pw.println();
            while (rs.next()) {
                for (int i = 1; i <= n; i++) {
                    Object v = rs.getObject(i);
                    String s = v == null ? "" : (v instanceof Double d ? String.format(Locale.US, "%.6f", d) : v.toString());
                    if (s.contains(",") || s.contains("\"")) s = "\"" + s.replace("\"", "\"\"") + "\"";
                    pw.print((i > 1 ? "," : "") + s);
                }
                pw.println();
            }
        }
    }
}