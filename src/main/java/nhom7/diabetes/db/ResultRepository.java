package nhom7.diabetes.db;

import nhom7.diabetes.evaluation.Metrics;

import java.sql.Connection;
import java.sql.PreparedStatement;

/**
 * Lưu kết quả huấn luyện / đánh giá vào CSDL.
 */
public final class ResultRepository {

    private ResultRepository() {}

    public static void clearStage(String stage) throws Exception {
        try (Connection c = DatabaseManager.getConnection()) {
            try (PreparedStatement ps = c.prepareStatement("DELETE FROM model_results WHERE stage=?")) {
                ps.setString(1, stage);
                ps.executeUpdate();
            }
            try (PreparedStatement ps = c.prepareStatement("DELETE FROM feature_importance WHERE stage=?")) {
                ps.setString(1, stage);
                ps.executeUpdate();
            }
        }
    }

    public static void clearCv(String model) throws Exception {
        try (Connection c = DatabaseManager.getConnection();
             PreparedStatement ps = c.prepareStatement("DELETE FROM cv_results WHERE model=?")) {
            ps.setString(1, model);
            ps.executeUpdate();
        }
    }

    public static void saveModelResult(String stage, String model, String params, Metrics m, long trainMs) throws Exception {
        String sql = "INSERT INTO model_results(stage,model,params,accuracy,precision_,recall,f1,roc_auc,tn,fp,fn,tp,train_ms)"
                + " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)";
        try (Connection c = DatabaseManager.getConnection(); PreparedStatement ps = c.prepareStatement(sql)) {
            ps.setString(1, stage);
            ps.setString(2, model);
            ps.setString(3, params);
            ps.setDouble(4, m.accuracy);
            ps.setDouble(5, m.precision);
            ps.setDouble(6, m.recall);
            ps.setDouble(7, m.f1);
            ps.setDouble(8, m.rocAuc);
            ps.setInt(9, m.tn);
            ps.setInt(10, m.fp);
            ps.setInt(11, m.fn);
            ps.setInt(12, m.tp);
            ps.setLong(13, trainMs);
            ps.executeUpdate();
        }
    }

    public static void saveCvResult(String model, String searchType, String params, int k,
                                    double acc, double prec, double rec, double f1, double f1Std,
                                    double auc, long timeMs) throws Exception {
        String sql = "INSERT INTO cv_results(model,search_type,params,k_folds,acc_mean,prec_mean,rec_mean,f1_mean,f1_std,auc_mean,time_ms)"
                + " VALUES(?,?,?,?,?,?,?,?,?,?,?)";
        try (Connection c = DatabaseManager.getConnection(); PreparedStatement ps = c.prepareStatement(sql)) {
            ps.setString(1, model);
            ps.setString(2, searchType);
            ps.setString(3, params);
            ps.setInt(4, k);
            ps.setDouble(5, acc);
            ps.setDouble(6, prec);
            ps.setDouble(7, rec);
            ps.setDouble(8, f1);
            ps.setDouble(9, f1Std);
            ps.setDouble(10, auc);
            ps.setLong(11, timeMs);
            ps.executeUpdate();
        }
    }

    public static void markBestCv(String model, String params) throws Exception {
        try (Connection c = DatabaseManager.getConnection()) {
            try (PreparedStatement ps = c.prepareStatement("UPDATE cv_results SET is_best=0 WHERE model=?")) {
                ps.setString(1, model);
                ps.executeUpdate();
            }
            try (PreparedStatement ps = c.prepareStatement("UPDATE cv_results SET is_best=1 WHERE model=? AND params=?")) {
                ps.setString(1, model);
                ps.setString(2, params);
                ps.executeUpdate();
            }
        }
    }

    public static void saveImportance(String stage, String model, String[] features, double[] imp) throws Exception {
        if (imp == null) return;
        try (Connection c = DatabaseManager.getConnection();
             PreparedStatement ps = c.prepareStatement("INSERT INTO feature_importance(stage,model,feature,importance) VALUES(?,?,?,?)")) {
            for (int j = 0; j < features.length; j++) {
                ps.setString(1, stage);
                ps.setString(2, model);
                ps.setString(3, features[j]);
                ps.setDouble(4, imp[j]);
                ps.addBatch();
            }
            ps.executeBatch();
        }
    }
}
