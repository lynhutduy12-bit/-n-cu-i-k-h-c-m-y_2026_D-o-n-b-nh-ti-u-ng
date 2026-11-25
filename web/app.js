// ===== Xử lý phía trình duyệt: gửi dữ liệu form lên máy chủ và hiển thị kết quả =====

const form = document.getElementById("predictForm");
const submitBtn = document.getElementById("submitBtn");
const formError = document.getElementById("formError");

// Tên hiển thị tiếng Việt cho tiền sử hút thuốc (dùng trong bảng lịch sử)
const SMOKING_VI = {
    "never": "Chưa từng hút",
    "former": "Đã bỏ",
    "current": "Đang hút",
    "not current": "Hiện không hút",
    "ever": "Đã từng hút",
    "No Info": "Không rõ"
};

// ---------- 1. Thông tin mô hình (hiện ở góc trên) ----------
async function loadModelInfo() {
    try {
        const res = await fetch("/api/model-info");
        const info = await res.json();
        let text = "Mô hình: " + info.model + " | Ngưỡng: " + info.threshold;
        if (info.auc !== undefined) {
            text += " | ROC-AUC: " + info.auc.toFixed(3) + " | F1: " + info.f1.toFixed(3);
        }
        document.getElementById("modelBadge").textContent = text;
    } catch (e) {
        document.getElementById("modelBadge").textContent = "Không kết nối được máy chủ";
    }
}

// ---------- 2. Gửi form để dự đoán ----------
form.addEventListener("submit", async (event) => {
    event.preventDefault();                 // không tải lại trang
    formError.hidden = true;
    document.querySelectorAll("input.invalid").forEach(el => el.classList.remove("invalid"));

    // Kiểm tra nhanh phía trình duyệt: các ô số không được để trống
    const required = ["age", "bmi", "hba1c_level", "blood_glucose_level"];
    const empty = required.filter(id => document.getElementById(id).value.trim() === "");
    if (empty.length > 0) {
        empty.forEach(id => document.getElementById(id).classList.add("invalid"));
        showError("Vui lòng nhập đủ: Tuổi, BMI, HbA1c và Mức đường huyết.");
        return;
    }

    submitBtn.disabled = true;
    submitBtn.textContent = "Đang dự đoán...";
    try {
        // Gửi dữ liệu dạng form: gender=Male&age=45&...
        const body = new URLSearchParams(new FormData(form));
        const res = await fetch("/api/predict", { method: "POST", body: body });
        const data = await res.json();
        if (!res.ok) {
            showError(data.error || "Có lỗi xảy ra.");   // lỗi kiểm tra dữ liệu từ máy chủ
            return;
        }
        showResult(data);
        loadHistory();
    } catch (e) {
        showError("Không kết nối được máy chủ. Hãy kiểm tra chương trình Java đang chạy.");
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = "Dự đoán";
    }
});

function showError(msg) {
    formError.textContent = msg;
    formError.hidden = false;
}

// ---------- 3. Hiển thị kết quả ----------
function showResult(data) {
    document.getElementById("resultEmpty").hidden = true;
    document.getElementById("resultBox").hidden = false;

    const labels = { thap: "Nguy cơ thấp", theo_doi: "Cần theo dõi", cao: "Có nguy cơ" };
    const verdict = document.getElementById("verdict");
    verdict.className = "verdict level-" + data.level;
    document.getElementById("verdictLabel").textContent = labels[data.level];
    document.getElementById("probText").textContent = (data.probability * 100).toFixed(1) + "%";

    // Thanh xác suất: kim chỉ = xác suất, vạch đỏ = ngưỡng của mô hình
    document.getElementById("gaugeFill").style.left = "calc(" + (data.probability * 100) + "% - 2px)";
    document.getElementById("gaugeMark").style.left = (data.threshold * 100) + "%";
    document.getElementById("thresholdText").textContent =
        "Ngưỡng kết luận: " + (data.threshold * 100).toFixed(0) + "%";

    document.getElementById("messageText").textContent = data.message;

    const list = document.getElementById("indicatorList");
    list.innerHTML = "";
    data.indicators.forEach(it => {
        const li = document.createElement("li");
        li.className = it.status;
        li.innerHTML = "<span><b></b><span class='val'></span></span><span class='note'></span>";
        li.querySelector("b").textContent = it.name;
        li.querySelector(".val").textContent = it.value;
        li.querySelector(".note").textContent = it.note;
        list.appendChild(li);
    });

    const warn = document.getElementById("warningList");
    warn.innerHTML = "";
    data.warnings.forEach(w => {
        const li = document.createElement("li");
        li.textContent = w;
        warn.appendChild(li);
    });
}

// ---------- 4. Lịch sử dự đoán ----------
async function loadHistory() {
    try {
        const res = await fetch("/api/history");
        const rows = await res.json();
        const tbody = document.getElementById("historyBody");
        tbody.innerHTML = "";
        if (rows.length === 0) {
            tbody.innerHTML = "<tr><td colspan='11' class='center'>Chưa có dữ liệu</td></tr>";
            return;
        }
        rows.forEach(r => {
            const tr = document.createElement("tr");
            r[5] = SMOKING_VI[r[5]] || r[5];
            r.forEach((value, i) => {
                const td = document.createElement("td");
                if (i === r.length - 1) {             // cột kết luận: tô màu
                    const span = document.createElement("span");
                    span.className = "tag " + (value === "Có nguy cơ" ? "yes" : "no");
                    span.textContent = value;
                    td.appendChild(span);
                } else {
                    td.textContent = value;
                }
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
    } catch (e) {
        // bỏ qua: không làm gián đoạn trang
    }
}
document.getElementById("reloadHistory").addEventListener("click", loadHistory);

// ---------- 5. Tiện ích: tính BMI, dữ liệu mẫu ----------
document.getElementById("bmiToggle").addEventListener("click", () => {
    const box = document.getElementById("bmiCalc");
    box.hidden = !box.hidden;
});

document.getElementById("bmiApply").addEventListener("click", () => {
    const h = parseFloat(document.getElementById("heightCm").value) / 100;   // cm -> m
    const w = parseFloat(document.getElementById("weightKg").value);
    if (!h || !w) {
        showError("Nhập chiều cao (cm) và cân nặng (kg) để tính BMI.");
        return;
    }
    document.getElementById("bmi").value = (w / (h * h)).toFixed(1);       // BMI = kg / m²
    formError.hidden = true;
});

// Dữ liệu mẫu để demo nhanh khi thuyết trình (lấy luân phiên 2 trường hợp)
const SAMPLES = [
    { gender: "Male", age: 62, hypertension: "1", heart_disease: "0", smoking_history: "former",
      bmi: 31.2, hba1c_level: 7.0, blood_glucose_level: 220 },
    { gender: "Female", age: 28, hypertension: "0", heart_disease: "0", smoking_history: "never",
      bmi: 21.5, hba1c_level: 5.0, blood_glucose_level: 100 }
];
let sampleIndex = 0;
document.getElementById("sampleBtn").addEventListener("click", () => {
    const s = SAMPLES[sampleIndex];
    sampleIndex = (sampleIndex + 1) % SAMPLES.length;
    for (const [name, value] of Object.entries(s)) {
        const radio = form.querySelector(`input[type=radio][name="${name}"][value="${value}"]`);
        if (radio) radio.checked = true;
        else form.elements[name].value = value;
    }
});

loadModelInfo();
loadHistory();
