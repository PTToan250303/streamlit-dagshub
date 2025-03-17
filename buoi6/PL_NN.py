import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import joblib
import os
import mlflow
from mlflow.tracking import MlflowClient
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Hàm chuẩn hóa dữ liệu (giữ nguyên)
@st.cache_data
def standardize_data(X, fit=True, _scaler=None):
    if fit or _scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler
    else:
        return _scaler.transform(X), _scaler

# Hàm tải dữ liệu MNIST (giữ nguyên)
def load_mnist_data():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    return X, y

# Hàm chia dữ liệu (giữ nguyên)
def split_data():
    st.title("📌 Chia dữ liệu Train/Test")
    X, y = load_mnist_data()
    total_samples = X.shape[0]
    num_classes = len(np.unique(y))

    if "data_split_done" not in st.session_state:
        st.session_state.data_split_done = False  

    num_samples = st.slider("📌 Chọn số lượng ảnh để train:", 1000, total_samples - num_classes, 10000)
    test_size_percent = st.slider("📌 Chọn % dữ liệu Test", 10, 80, 10)
    test_size = test_size_percent / 100
    remaining_size = 100 - test_size_percent
    val_size_percent = st.slider("📌 Chọn % dữ liệu Validation (trong phần Train)", 0, min(80, remaining_size), 0)
    val_size = val_size_percent / 100

    test_samples = int(num_samples * test_size)
    train_val_samples = num_samples - test_samples
    val_samples = int(train_val_samples * (val_size_percent / remaining_size)) if val_size_percent > 0 else 0

    if test_samples < num_classes or (val_samples < num_classes and val_size_percent > 0):
        st.error(f"❌ Số lượng mẫu trong tập Test hoặc Validation phải lớn hơn hoặc bằng số lớp ({num_classes}).")
        return

    train_percent = remaining_size - val_size_percent
    if train_percent < 30:
        st.warning(f"⚠️ Tỷ lệ Train chỉ còn {train_percent}%! Điều này có thể ảnh hưởng đến hiệu suất mô hình.")

    st.write(f"📌 **Tỷ lệ phân chia:** Test={test_size_percent}%, Validation={val_size_percent}%, Train={train_percent}%")

    if st.session_state.data_split_done:
        if st.button("🔄 Reset & Chia lại"):
            st.session_state.data_split_done = False
            st.rerun()

    if st.button("✅ Xác nhận & Lưu"):
        st.session_state.data_split_done = True
        X_selected, _, y_selected, _ = train_test_split(X, y, train_size=num_samples, stratify=y, random_state=42)
        X_train_full, X_test, y_train_full, y_test = train_test_split(X_selected, y_selected, test_size=test_size, stratify=y_selected, random_state=42)

        if val_size_percent > 0:
            X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_size, stratify=y_train_full, random_state=42)
        else:
            X_train, X_val, y_train, y_val = X_train_full, np.array([]), y_train_full, np.array([])

        X_train = X_train / 255.0
        X_test = X_test / 255.0
        X_val = X_val / 255.0 if val_size_percent > 0 else X_val

        st.session_state.total_samples = num_samples
        st.session_state.X_train = X_train
        st.session_state.X_val = X_val
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_val = y_val
        st.session_state.y_test = y_test
        st.session_state.test_size = X_test.shape[0]
        st.session_state.val_size = X_val.shape[0] if val_size_percent > 0 else 0
        st.session_state.train_size = X_train.shape[0]

        summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Số lượng mẫu": [X_train.shape[0], X_val.shape[0] if val_size_percent > 0 else 0, X_test.shape[0]]
        })
        st.success("✅ Dữ liệu đã được chia và chuẩn hóa thành công!")
        st.table(summary_df)

    elif st.session_state.data_split_done:
        st.info("✅ Dữ liệu đã được chia. Nhấn 'Reset & Chia lại' để điều chỉnh.")
        summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Số lượng mẫu": [st.session_state.train_size, st.session_state.val_size, st.session_state.test_size]
        })
        st.table(summary_df)

# Hàm huấn luyện mô hình Neural Network (đã sửa với key duy nhất)
def train():
    if "X_train" not in st.session_state:
        st.error("⚠️ Chưa có dữ liệu! Vui lòng quay lại bước chia dữ liệu trước.")
        st.button("🔙 Quay lại bước chia dữ liệu", on_click=lambda: st.session_state.update({"page": "data_split"}))
        return

    X_train = st.session_state.X_train
    X_val = st.session_state.X_val
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_val = st.session_state.y_val
    y_test = st.session_state.y_test

    st.header("⚙️ Chọn mô hình Neural Network & Huấn luyện")
    hidden_layer_sizes = st.slider("Hidden Layer Sizes", 10, 200, (100,), step=10, key="train_hidden_layer_sizes")
    activation = st.selectbox("Activation Function", ["relu", "tanh"], key="train_activation")
    learning_rate_init = st.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001, key="train_learning_rate")
    max_iter = st.slider("Max Iterations", 100, 1000, 500, step=100, key="train_max_iter")

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=42
    )

    n_folds = st.slider("Chọn số folds (KFold Cross-Validation):", min_value=2, max_value=10, value=5, key="train_n_folds")

    if "experiment_name" not in st.session_state:
        st.session_state["experiment_name"] = "Neural_Network_Experiment"

    experiment_name = st.text_input("🔹 Nhập tên Experiment:", st.session_state["experiment_name"], key="train_experiment_name")

    if experiment_name:
        st.session_state["experiment_name"] = experiment_name

    mlflow.set_experiment(experiment_name)
    st.write(f"✅ Experiment Name: {experiment_name}")

    if st.button("Huấn luyện mô hình"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        st.session_state["run_name"] = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=f"Train_{st.session_state['run_name']}"):
            try:
                status_text.text("🔄 Ghi log tham số vào MLflow...")
                progress_bar.progress(10)
                mlflow.log_param("test_size", st.session_state.get("test_size", 0))
                mlflow.log_param("val_size", st.session_state.get("val_size", 0))
                mlflow.log_param("train_size", st.session_state.get("train_size", 0))
                mlflow.log_param("num_samples", st.session_state.get("total_samples", 0))

                status_text.text("⏳ Đang chạy Cross-Validation...")
                progress_bar.progress(40)
                cv_scores = cross_val_score(model, X_train, y_train, cv=n_folds, n_jobs=-1)
                mean_cv_score = cv_scores.mean()
                std_cv_score = cv_scores.std()

                st.success(f"📊 **Cross-Validation Accuracy**: {mean_cv_score:.4f} ± {std_cv_score:.4f}")

                status_text.text("⏳ Đang huấn luyện mô hình...")
                progress_bar.progress(70)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                st.success(f"✅ Độ chính xác trên test set: {acc:.4f}")

                status_text.text("🔄 Ghi log kết quả vào MLflow...")
                progress_bar.progress(90)
                mlflow.log_param("model", "Neural_Network")
                mlflow.log_param("hidden_layer_sizes", hidden_layer_sizes)
                mlflow.log_param("activation", activation)
                mlflow.log_param("learning_rate_init", learning_rate_init)
                mlflow.log_param("max_iter", max_iter)
                mlflow.log_metric("test_accuracy", acc)
                mlflow.log_metric("cv_accuracy_mean", mean_cv_score)
                mlflow.log_metric("cv_accuracy_std", std_cv_score)
                mlflow.sklearn.log_model(model, "model_neural_network")

                if "models" not in st.session_state:
                    st.session_state["models"] = {}
                model_name = f"neural_network_{activation}_{hidden_layer_sizes[0]}"
                count = 1
                base_model_name = model_name
                while model_name in st.session_state["models"]:
                    model_name = f"{base_model_name}_{count}"
                    count += 1
                st.session_state["models"][model_name] = model
                st.write(f"🔹 Mô hình đã được lưu với tên: {model_name}")
                st.write(f"Tổng số mô hình hiện tại: {len(st.session_state['models'])}")
                st.session_state["last_trained_model"] = model_name
                st.write("📋 Danh sách các mô hình đã lưu:")
                model_names = list(st.session_state["models"].keys())
                st.write(", ".join(model_names))

                status_text.text("✅ Hoàn tất huấn luyện!")
                progress_bar.progress(100)
                st.success(f"✅ Đã log dữ liệu cho **Train_{st.session_state['run_name']}**!")
                if "mlflow_url" in st.session_state:
                    st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")
                else:
                    st.warning("⚠️ URL MLflow chưa được thiết lập.")

            except Exception as e:
                st.error(f"❌ Lỗi khi huấn luyện: {str(e)}")
                mlflow.end_run()
                progress_bar.progress(0)
                status_text.text("❌ Huấn luyện thất bại!")

# Hàm dự đoán (giữ nguyên)
def du_doan():
    st.header("✍️ Vẽ số để dự đoán")
    model_names = list(st.session_state.get("models", {}).keys())
    if model_names:
        model_option = st.selectbox("🔍 Chọn mô hình:", model_names)
        model = st.session_state["models"][model_option]
        st.success(f"✅ Đã chọn mô hình từ session_state: {model_option}")
    else:
        st.warning("⚠️ Chưa có mô hình nào được huấn luyện.")
        return

    if "key_value" not in st.session_state:
        st.session_state.key_value = str(np.random.randint(0, 1000000))

    if st.button("🔄 Tải lại nếu không thấy canvas"):
        st.session_state.key_value = str(np.random.randint(0, 1000000))
        st.rerun()

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=150,
        width=150,
        drawing_mode="freedraw",
        key=st.session_state.key_value,
        update_streamlit=True
    )

    if st.button("Dự đoán số"):
        img = preprocess_canvas_image(canvas_result)
        if img is not None:
            st.image(Image.fromarray((img.reshape(28, 28) * 255).astype(np.uint8)), caption="Ảnh sau xử lý", width=100)
            prediction = model.predict(img)
            st.subheader(f"🔢 Dự đoán: {prediction[0]}")
        else:
            st.error("⚠️ Hãy vẽ một số trước khi bấm Dự đoán!")

# Hàm hiển thị experiment MLflow (giữ nguyên)
def show_experiment_selector():
    st.title("📊 MLflow Experiments - DAGsHub")
    experiments = mlflow.search_experiments()
    experiment_names = [exp.name for exp in experiments]
    selected_experiment_name = st.selectbox("🔍 Chọn một Experiment:", experiment_names)
    if not selected_experiment_name:
        st.error(f"❌ Experiment '{selected_experiment_name}' không tồn tại!")
        return
    selected_experiment = next((exp for exp in experiments if exp.name == selected_experiment_name), None)
    if not selected_experiment:
        st.error("❌ Không tìm thấy experiment trong danh sách.")
        return
    st.subheader(f"📌 Experiment: {selected_experiment_name}")
    st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
    st.write(f"**Trạng thái:** {'Active' if selected_experiment.lifecycle_stage == 'active' else 'Deleted'}")
    st.write(f"**Vị trí lưu trữ:** {selected_experiment.artifact_location}")
    runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])
    if runs.empty:
        st.warning("⚠ Không có runs nào trong experiment này.")
        return
    st.write("### 🏃‍♂️ Các Runs gần đây:")
    run_info = []
    for run_id in runs["run_id"]:
        run = mlflow.get_run(run_id)
        run_name = run.data.params.get("run_name", f"Run {run_id[:8]}")
        run_info.append((run_name, run_id))
    run_info.sort(key=lambda x: mlflow.get_run(x[1]).info.start_time, reverse=True)
    if run_info:
        latest_run_name, latest_run_id = run_info[0]
        selected_run_name = latest_run_name
        selected_run_id = latest_run_id
    else:
        st.warning("⚠ Không có runs nào trong experiment này.")
        return
    selected_run = mlflow.get_run(selected_run_id)
    if selected_run:
        st.subheader(f"📌 Thông tin Run: {selected_run_name}")
        st.write(f"**Run ID:** {selected_run_id}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")
        start_time_ms = selected_run.info.start_time
        start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S") if start_time_ms else "Không có thông tin"
        st.write(f"**Thời gian chạy:** {start_time}")
        params = selected_run.data.params
        metrics = selected_run.data.metrics
        if params:
            st.write("### ⚙️ Parameters:")
            st.json(params)
        if metrics:
            st.write("### 📊 Metrics:")
            st.json(metrics)

# Hàm lý thuyết Neural Network (giữ nguyên)
def ly_thuyet_neural_network():
    st.markdown('<h1 style="color:#FF4500; text-align:center;">Neural Network (Mạng Nơ-ron Nhân tạo)</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:24px; color:#4682B4;">📝 Tìm hiểu về Mạng Nơ-ron Nhân tạo (Neural Network) và cách nó hoạt động trong phân loại dữ liệu.</p>', unsafe_allow_html=True)
    st.markdown('<h2 style="font-size:32px; color:#32CD32;">📚 1. Neural Network là gì và cách hoạt động?</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">❓ Mạng Nơ-ron Nhân tạo (Artificial Neural Network - ANN) là mô hình học máy lấy cảm hứng từ cách hoạt động của não bộ người. Nó bao gồm nhiều nơ-ron nhân tạo được tổ chức thành các lớp để học và dự đoán từ dữ liệu phức tạp như chữ số viết tay trong MNIST.<br>🚀 <b>Các khái niệm chính:</b><br>- Nơ-ron (Neuron): Đơn vị cơ bản nhận đầu vào, áp dụng trọng số và bias, sau đó chuyển qua hàm kích hoạt.<br>- Lớp đầu vào (Input Layer): Nhận dữ liệu thô (ví dụ: 784 pixel từ ảnh 28x28).<br>- Lớp ẩn (Hidden Layer): Xử lý các đặc trưng trung gian.<br>- Lớp đầu ra (Output Layer): Đưa ra kết quả dự đoán (ví dụ: 10 lớp từ 0-9).<br>📐 <b>Quá trình học:</b><br></p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">- Forward Propagation: Tính toán đầu ra từ đầu vào qua các lớp.<br>- Backpropagation: Lan truyền ngược lỗi để cập nhật trọng số.<br></p>', unsafe_allow_html=True)
    st.latex(r'z = w^T x + b')
    st.markdown("Trong đó: - \( z \): Tổng có trọng số trước hàm kích hoạt. - \( w \): Trọng số. - \( x \): Đầu vào. - \( b \): Bias.")
    st.latex(r'a = f(z)')
    st.markdown("Trong đó: - \( a \): Đầu ra sau hàm kích hoạt \( f \).")
    st.markdown('<h2 style="font-size:32px; color:#FFD700;">📐 2. Hàm kích hoạt (Activation Functions)</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">- Hàm kích hoạt giúp giới thiệu tính phi tuyến, quan trọng để mô hình học các mẫu phức tạp.<br></p>', unsafe_allow_html=True)
    st.latex(r'f(z) = \frac{1}{1 + e^{-z}}')
    st.markdown("Trong đó: - \( f(z) \): Đầu ra trong khoảng (0, 1), phù hợp cho phân loại nhị phân.")
    st.latex(r'f(z) = \max(0, z)')
    st.markdown("Trong đó: - \( f(z) \): Đầu ra là \( z \) nếu \( z > 0 \), bằng 0 nếu \( z \leq 0 \), giảm vấn đề gradient vanishing.")
    st.latex(r'f(z) = \tanh(z)')
    st.markdown("Trong đó: - \( f(z) \): Đầu ra trong khoảng (-1, 1), cân bằng dữ liệu tốt hơn sigmoid.")
    st.markdown('<h2 style="font-size:32px; color:#00CED1;">📈 3. Quá trình học và Backpropagation</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">- **Hàm mất mát (Loss Function)**: Đo sai lệch giữa dự đoán và giá trị thực tế.<br></p>', unsafe_allow_html=True)
    st.latex(r'L = \frac{1}{2} \sum (y - \hat{y})^2')
    st.markdown("Trong đó: - \( L \): Giá trị mất mát. - \( y \): Giá trị thực tế. - \( \hat{y} \): Giá trị dự đoán.")
    st.latex(r'w_{new} = w_{old} - \eta \frac{\partial L}{\partial w}')
    st.markdown("Trong đó: - \( \eta \): Tốc độ học (learning rate). - \( \frac{\partial L}{\partial w} \): Gradient của hàm mất mát theo trọng số.")
    st.markdown('<h2 style="font-size:32px; color:#FF69B4;">📊 4. Ví dụ minh họa trên MNIST</h2>', unsafe_allow_html=True)
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(10), np.bincount([int(i) for i in y[:100]]))
    ax.set_title("Phân bố lớp trong 100 mẫu MNIST")
    ax.set_xlabel("Số")
    ax.set_ylabel("Số lượng")
    st.pyplot(fig)
    st.markdown('<p style="font-size:18px;">📊 Biểu đồ trên thể hiện phân bố các chữ số trong 100 mẫu đầu tiên của MNIST.</p>', unsafe_allow_html=True)
    st.markdown('<h2 style="font-size:32px; color:#FFA500;">⚠️ 5. Ưu điểm, hạn chế và ứng dụng</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">👍 **Ưu điểm**: - Linh hoạt với dữ liệu phức tạp như hình ảnh, âm thanh.<br>- Có khả năng học các đặc trưng không tuyến tính.<br>👎 **Hạn chế**: - Tốn tài nguyên tính toán lớn.<br>- Dễ bị overfitting nếu không điều chỉnh kỹ.<br>🌍 **Ứng dụng**: Nhận diện chữ số (MNIST), xử lý ngôn ngữ tự nhiên, xe tự hành.<br></p>', unsafe_allow_html=True)
    st.markdown('<h2 style="font-size:32px; color:#1E90FF;">🔗 6. Tài liệu tham khảo</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">📖 Xem chi tiết tại <a href="https://kdientu.duytan.edu.vn/media/50176/ly-thuyet-mang-neural.pdf?form=MG0AV3">Tài liệu Neural Network - Đại học Duy Tân</a>.</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:20px; color:#6A5ACD;">🙏 Cảm ơn bạn đã khám phá Neural Network!</p>', unsafe_allow_html=True)

# Hàm xử lý ảnh từ canvas (giữ nguyên)
def preprocess_canvas_image(canvas_result):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data[:, :, 0].astype(np.uint8))
        img = img.resize((28, 28)).convert("L")
        img = np.array(img, dtype=np.float32) / 255.0
        return img.reshape(1, -1)
    return None

# Hàm Pseudo Labeling (đã sửa với key duy nhất)
def pseudo_labeling():
    st.header("🎯 Pseudo Labeling với Neural Network trên MNIST")

    if "data_split_done" not in st.session_state or "X_train" not in st.session_state:
        st.error("⚠️ Vui lòng chia dữ liệu trước ở tab 'Chia dữ liệu Train/Test'!")
        st.button("🔙 Quay lại bước chia dữ liệu", on_click=lambda: st.session_state.update({"page": "data_split"}))
        return

    X_train_full = st.session_state.X_train
    y_train_full = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    st.markdown("""
    <p style="font-size:20px;">
    📋 **Quy trình Pseudo Labeling**:<br>
    (0) Chia tập train/test ban đầu.<br>
    (1) Lấy 1% số lượng ảnh cho mỗi class (0-9) làm tập train ban đầu.<br>
    (2) Huấn luyện mô hình NN trên tập 1% này.<br>
    (3) Dự đoán nhãn cho 99% dữ liệu còn lại.<br>
    (4) Gán Pseudo Label với ngưỡng (threshold) cho các mẫu có độ tin cậy cao.<br>
    (5) Lặp lại từ bước (2) với tập dữ liệu mới cho đến khi gán hết nhãn hoặc đạt số bước lặp tối đa.<br>
    </p>
    """, unsafe_allow_html=True)

    # Chọn tham số với key duy nhất
    threshold = st.slider("📌 Ngưỡng quyết định (Threshold) cho Pseudo Label", 0.5, 1.0, 0.95, step=0.05, key="pseudo_threshold")
    max_iterations = st.slider("📌 Số bước lặp tối đa", 1, 10, 5, key="pseudo_max_iterations")
    hidden_layer_sizes = st.slider("Hidden Layer Sizes", 10, 200, (100,), step=10, key="pseudo_hidden_layer_sizes")
    activation = st.selectbox("Activation Function", ["relu", "tanh"], key="pseudo_activation")
    learning_rate_init = st.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001, key="pseudo_learning_rate")
    max_iter = st.slider("Max Iterations", 100, 1000, 500, step=100, key="pseudo_max_iter")

    if st.button("🎬 Bắt đầu Pseudo Labeling"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Bước (1): Lấy 1% dữ liệu cho mỗi class
        status_text.text("🔄 Đang lấy 1% dữ liệu ban đầu cho mỗi class...")
        progress_bar.progress(10)
        num_classes = 10
        samples_per_class = int(len(X_train_full) * 0.01 / num_classes)
        X_labeled = np.array([])
        y_labeled = np.array([])
        for cls in range(num_classes):
            indices = np.where(y_train_full == str(cls))[0]
            if len(indices) > samples_per_class:
                selected_indices = np.random.choice(indices, samples_per_class, replace=False)
            else:
                selected_indices = indices
            X_labeled = np.concatenate((X_labeled, X_train_full[selected_indices])) if X_labeled.size else X_train_full[selected_indices]
            y_labeled = np.concatenate((y_labeled, y_train_full[selected_indices].astype(float))) if y_labeled.size else y_train_full[selected_indices].astype(float)
        X_unlabeled = np.delete(X_train_full, np.unique(np.concatenate([np.where(y_train_full == str(cls))[0] for cls in range(num_classes)])), axis=0)
        y_unlabeled = np.array(["unlabeled"] * len(X_unlabeled))

        # Biến để theo dõi số mẫu đã gán nhãn
        labeled_count = len(y_labeled)
        total_samples = len(X_train_full)
        iteration = 0

        while iteration < max_iterations and labeled_count < total_samples:
            status_text.text(f"⏳ Đang thực hiện bước lặp {iteration + 1}/{max_iterations}...")
            progress_bar.progress(int((iteration + 1) * 100 / max_iterations))

            # Bước (2): Huấn luyện mô hình trên tập dữ liệu đã gán nhãn
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                random_state=42
            )
            model.fit(X_labeled, y_labeled)
            mlflow.start_run(run_name=f"Pseudo_Labeling_Iter_{iteration}")
            mlflow.log_param("threshold", threshold)
            mlflow.log_param("iteration", iteration)
            mlflow.log_param("labeled_samples", len(y_labeled))
            mlflow.log_metric("accuracy_on_test", accuracy_score(y_test, model.predict(X_test)))
            mlflow.sklearn.log_model(model, f"model_iter_{iteration}")
            mlflow.end_run()

            # Bước (3): Dự đoán nhãn cho dữ liệu chưa gán nhãn
            probabilities = model.predict_proba(X_unlabeled)
            y_pred_unlabeled = model.predict(X_unlabeled)
            confidence = np.max(probabilities, axis=1)

            # Bước (4): Gán Pseudo Label với ngưỡng
            mask = confidence >= threshold
            new_labeled_indices = np.where(mask)[0]
            if len(new_labeled_indices) == 0:
                st.warning(f"⚠️ Không có mẫu nào đạt ngưỡng {threshold} ở bước lặp {iteration + 1}. Kết thúc!")
                break

            X_new_labeled = X_unlabeled[new_labeled_indices]
            y_new_labeled = y_pred_unlabeled[new_labeled_indices]
            X_labeled = np.concatenate((X_labeled, X_new_labeled))
            y_labeled = np.concatenate((y_labeled, y_new_labeled))
            X_unlabeled = np.delete(X_unlabeled, new_labeled_indices, axis=0)
            y_unlabeled = np.delete(y_unlabeled, new_labeled_indices)

            labeled_count = len(y_labeled)
            st.success(f"✅ Bước {iteration + 1}: Đã gán nhãn cho {len(new_labeled_indices)} mẫu. Tổng số mẫu đã gán: {labeled_count}/{total_samples}")

            iteration += 1

        # Kết thúc
        status_text.text("✅ Hoàn tất Pseudo Labeling!")
        progress_bar.progress(100)
        st.write(f"📊 Tổng số mẫu đã gán nhãn: {labeled_count}/{total_samples}")
        if labeled_count == total_samples:
            st.success("🎉 Đã gán nhãn cho toàn bộ tập train ban đầu!")
        else:
            st.warning(f"⚠️ Chỉ gán nhãn được {labeled_count} mẫu sau {iteration} bước lặp.")

        # Đánh giá mô hình cuối cùng
        final_model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=42
        )
        final_model.fit(X_labeled, y_labeled)
        final_accuracy = accuracy_score(y_test, final_model.predict(X_test))
        st.success(f"✅ Độ chính xác trên test set với mô hình cuối cùng: {final_accuracy:.4f}")

def data():
    st.markdown("""
        <h1 style="text-align: center; color: #1E90FF; font-size: 48px; text-shadow: 2px 2px 4px #000000;">
             Khám Phá Bộ Dữ Liệu MNIST
        </h1>
        <style>
        @keyframes fadeIn { from {opacity: 0;} to {opacity: 1;} }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="background-color: #F0F8FF; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h2 style="color: #32CD32; font-size: 32px;">📊 Tổng Quan Về MNIST</h2>
            <p style="font-size: 20px; color: #333; text-align: justify;">
                MNIST (Modified National Institute of Standards and Technology) là bộ dữ liệu <b>huyền thoại</b> 
                trong nhận diện chữ số viết tay, với <b>70.000 ảnh</b> (60.000 train, 10.000 test), mỗi ảnh 
                có kích thước <b>28x28 pixel</b> grayscale.
            </p>
        </div>
    """, unsafe_allow_html=True)

    X, y = load_mnist_data()
    fig, ax = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        ax[i].imshow(X[i].reshape(28, 28), cmap='gray')
        ax[i].axis('off')
        ax[i].set_title(f"Nhãn: {int(y[i])}")
    # Sử dụng st.pyplot() thay vì st.image()
    st.pyplot(fig)

    st.markdown("""
        <h2 style="color: #FF4500; font-size: 32px;">🌍 Ứng Dụng Thực Tế</h2>
        <div style="display: flex; gap: 20px;">
            <div style="background-color: #ECF0F1; padding: 15px; border-radius: 10px; flex: 1;">
                <p style="font-size: 20px; color: #2E86C1;">Nhận diện số trên hóa đơn.</p>
            </div>
            <div style="background-color: #ECF0F1; padding: 15px; border-radius: 10px; flex: 1;">
                <p style="font-size: 20px; color: #2E86C1;">Xử lý mã bưu kiện.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='color: #8A2BE2; font-size: 32px;'>🏆 Hiệu Suất Mô Hình</h2>", unsafe_allow_html=True)
    data = {"Mô hình": ["Neural Network", "SVM", "CNN"], "Độ chính xác": ["0.98", "0.97", "0.99"]}
    df = pd.DataFrame(data)
    st.table(df.style.set_properties(**{'background-color': '#F5F5F5', 'border': '1px solid #DDD', 'text-align': 'center', 'font-size': '18px'}).set_table_styles([{'selector': 'th', 'props': [('background-color', '#3498DB'), ('color', 'white')]}]))

# Hàm chính để tích hợp tất cả các tab
def Classification():
    if "mlflow_initialized" not in st.session_state:
        DAGSHUB_MLFLOW_URI = "https://dagshub.com/PTToan250303/Linear_replication.mlflow"
        mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
        st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
        os.environ["MLFLOW_TRACKING_USERNAME"] = "PTToan250303"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "5ca8caf353d564c358852da97c7487e64fc30a73"
        mlflow.set_experiment("Neural_Network_Classification")

    st.markdown("""
        <style>
        .title { font-size: 48px; font-weight: bold; text-align: center; color: #4682B4; margin-top: 50px; }
        .subtitle { font-size: 24px; text-align: center; color: #4A4A4A; }
        hr { border: 1px solid #ddd; }
        </style>
        <div class="title">MNIST Neural Network App</div>
        <hr>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📘 Lý thuyết Neural Network", "📘 Data", "⚙️ Huấn luyện", "🔢 Dự đoán", "🔥 Mlflow", "🎯 Pseudo Labeling"])

    with tab1:
        ly_thuyet_neural_network()
    with tab2:
        data()
    with tab3:
        split_data()
        train()
    with tab4:
        du_doan()
    with tab5:
        show_experiment_selector()
    with tab6:
        pseudo_labeling()

if __name__ == "__main__":
    Classification()