#!/usr/bin/env bash
# Chạy chương trình trên macOS / Linux (cần JDK 17+)
cd "$(dirname "$0")"
mkdir -p bin
javac --release 17 -encoding UTF-8 -d bin -cp "lib/*" $(find src -name "*.java") || exit 1
java -cp "bin:lib/*" nhom7.diabetes.Main "$@"
