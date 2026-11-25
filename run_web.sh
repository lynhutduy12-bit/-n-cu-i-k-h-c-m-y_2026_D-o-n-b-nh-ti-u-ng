#!/usr/bin/env bash
# Chạy WEBSITE dự đoán (Tuần 11-12) trên macOS / Linux. Cần chạy mục 5 trước để có output/best_model.ser
cd "$(dirname "$0")"
mkdir -p bin
javac --release 17 -encoding UTF-8 -d bin -cp "lib/*" $(find src -name "*.java") || exit 1
java -cp "bin:lib/*" nhom7.diabetes.Main web
