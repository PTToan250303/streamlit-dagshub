import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
import random
from tensorflow.keras import layers

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

# Hàm chuẩn hóa dữ liệu
@st.cache_data
def standardize_data(X, fit=True, _scaler=None):
    if fit or _scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler
    else:
        return _scaler.transform(X), _scaler

# Hàm tải dữ liệu MNIST từ OpenML
def load_mnist_data():
    if "mnist_data" not in st.session_state:
        Xmt = np.load("buoi2/X.npy")
        ymt = np.load("buoi2/y.npy")
        X = Xmt.reshape(Xmt.shape[0], -1)  # Giữ nguyên định dạng dữ liệu
        y = ymt.reshape(-1)
        st.session_state["mnist_data"] = (X, y)
    return st.session_state["mnist_data"]

# Hàm chia dữ liệu
def split_data():
    st.title("📌 Chia dữ liệu Train/Test")
    
    # Đọc dữ liệu
    X, y = load_mnist_data()
    total_samples = X.shape[0] 
    
    # Thanh kéo chọn số lượng ảnh để train
    num_samples = st.slider("📌 Chọn số lượng ảnh để huấn luyện:", 1000, total_samples, 10000)
    num_samples =num_samples -10
    # Thanh kéo chọn tỷ lệ Train/Test
    test_size = st.slider("📌 Chọn % dữ liệu Test", 10, 50, 20)
    train_size = 100 - test_size
    val_size = st.slider("📌 Chọn % dữ liệu Validation (trong Train)", 0, 50, 15)
    
    st.write(f"📌 **Tỷ lệ phân chia:** Test={test_size}%, Validation={val_size}%, Train={train_size - val_size}%")
    
    if st.button("✅ Xác nhận & Lưu"):
        X_selected, _, y_selected, _ = train_test_split(X, y, train_size=num_samples, stratify=y, random_state=42)
        X_train_full, X_test, y_train_full, y_test = train_test_split(X_selected, y_selected, test_size=test_size/100, stratify=y_selected, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_size / (100 - test_size), stratify=y_train_full, random_state=42)
        
        # Lưu vào session_state
        st.session_state.update({
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test
        })
        
        summary_df = pd.DataFrame({"Tập dữ liệu": ["Train", "Validation", "Test"], "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]})
        st.success("✅ Dữ liệu đã được chia thành công!")
        st.table(summary_df)


# Hàm huấn luyện mô hình Neural Network
import streamlit as st
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import tensorflow
from tensorflow import keras

def train():
    # Khởi tạo các biến trong session_state nếu chưa có
    if "training_results" not in st.session_state:
        st.session_state["training_results"] = []  # Lưu kết quả huấn luyện của từng vòng lặp
    if "prediction_images" not in st.session_state:
        st.session_state["prediction_images"] = []  # Lưu hình ảnh dự đoán và thông tin đúng/sai của từng vòng lặp
    if "final_metrics" not in st.session_state:
        st.session_state["final_metrics"] = {}  # Lưu độ chính xác cuối cùng

    num = 0
    if "X_train" not in st.session_state:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return
    
    X_train, X_val, X_test = [st.session_state[k].reshape(-1, 28 * 28) / 255.0 for k in ["X_train", "X_val", "X_test"]]
    y_train, y_val, y_test = [st.session_state[k] for k in ["y_train", "y_val", "y_test"]]
    
    k_folds = st.slider("Số fold cho Cross-Validation:", 3, 10, 5)
    num_layers = st.slider("Số lớp ẩn:", 1, 20, 2)
    num_neurons = st.slider("Số neuron mỗi lớp:", 32, 512, 128, 32)
    activation = st.selectbox("Hàm kích hoạt:", ["relu", "sigmoid", "tanh"])
    optimizer = st.selectbox("Optimizer:", ["adam", "sgd", "rmsprop"])
    epochs = st.slider("🕰 Số epochs:", min_value=1, max_value=50, value=20, step=1)
    learning_rate = st.slider("⚡ Tốc độ học (Learning Rate):", min_value=1e-5, max_value=1e-1, value=1e-3, step=1e-5, format="%.5f")

    st.title(f"Chọn tham số cho Pseudo Labelling ")
    labeled_ratio = st.slider("📊 Tỉ lệ dữ liệu có nhãn ban đầu (%):", min_value=1, max_value=20, value=1, step=1)
    max_iterations = st.slider("🔄 Số lần lặp tối đa của Pseudo-Labeling:", min_value=1, max_value=10, value=3, step=1)
    confidence_threshold = st.slider("✅ Ngưỡng tin cậy Pseudo Labeling (%):", min_value=50, max_value=99, value=95, step=1) / 100.0

    loss_fn = "sparse_categorical_crossentropy"
    if "experiment_name" not in st.session_state:
        st.session_state["experiment_name"] = "My_Experiment"

    experiment_name = st.text_input("🔹 Nhập tên Experiment:", st.session_state["experiment_name"], key="experiment_name_input")    

    if experiment_name:
        st.session_state["experiment_name"] = experiment_name

    mlflow.set_experiment(experiment_name)
    st.write(f"✅ Experiment Name: {experiment_name}")
    
    # Hiển thị kết quả huấn luyện đã lưu (nếu có) khi chuyển tab
    if st.session_state["training_results"]:
        st.subheader("Kết quả huấn luyện trước đó:")
        for result in st.session_state["training_results"]:
            st.write(f"**Vòng lặp {result['iteration']}:**")
            st.write(f"- **Gán nhãn giả cho {result['num_pseudo_added']} mẫu với độ tin cậy ≥ {confidence_threshold}:**")
            st.write(f"  - Số nhãn giả đúng: {result['correct_pseudo_labels']}")
            st.write(f"  - Số nhãn giả sai: {result['incorrect_pseudo_labels']}")
            st.write(f"- **Số ảnh đã gán nhãn:** {result['total_labeled']}")
            st.write(f"- **Số ảnh chưa gán nhãn:** {result['remaining_unlabeled']}")
            st.write(f"- **Độ chính xác trên tập test:** {result['test_accuracy']:.4f}")
            # Tìm thông tin số lượng nhãn đúng/sai tương ứng với vòng lặp
            for img_data in st.session_state["prediction_images"]:
                if img_data["iteration"] == result["iteration"]:
                    st.write(f"- **Số lượng nhãn dự đoán đúng (trong 10 ảnh):** {img_data['correct_predictions']}")
                    st.write(f"- **Số lượng nhãn dự đoán sai (trong 10 ảnh):** {img_data['incorrect_predictions']}")
            st.write("---")

    # Hiển thị hình ảnh dự đoán và thông tin đúng/sai đã lưu (nếu có) khi chuyển tab
    if st.session_state["prediction_images"]:
        for img_data in st.session_state["prediction_images"]:
            st.subheader(f"Dự đoán 10 ảnh từ tập test sau vòng lặp {img_data['iteration']}")
            st.pyplot(img_data["figure"])
            st.write(f"- **Số lượng nhãn dự đoán đúng:** {img_data['correct_predictions']}")
            st.write(f"- **Số lượng nhãn dự đoán sai:** {img_data['incorrect_predictions']}")

    # Hiển thị độ chính xác cuối cùng đã lưu (nếu có) khi chuyển tab
    if st.session_state["final_metrics"]:
        st.success(f"✅ Huấn luyện hoàn tất!")
        st.write(f"📊 **Độ chính xác trung bình trên tập validation:** {st.session_state['final_metrics']['avg_val_accuracy']:.4f}")
        st.write(f"📊 **Độ chính xác trên tập test:** {st.session_state['final_metrics']['test_accuracy']:.4f}")
        st.success(f"✅ Đã log dữ liệu cho Experiments Neural_Network với Name: **Train_{st.session_state['run_name']}**!")
        st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")

    if st.button("🚀 Huấn luyện mô hình"):
        # Reset kết quả trước đó khi bắt đầu huấn luyện mới
        st.session_state["training_results"] = []
        st.session_state["prediction_images"] = []
        st.session_state["final_metrics"] = {}

        if "run_name" not in st.session_state:
            st.session_state["run_name"] = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}" 
        with st.spinner("Đang huấn luyện..."):
            mlflow.start_run(run_name=f"Train_{st.session_state['run_name']}")
            mlflow.log_params({
                "num_layers": num_layers,
                "num_neurons": num_neurons,
                "activation": activation,
                "optimizer": optimizer,
                "learning_rate": learning_rate,
                "k_folds": k_folds,
                "epochs": epochs,
                "labeled_ratio": labeled_ratio,
                "max_iterations": max_iterations,
                "confidence_threshold": confidence_threshold
            })

            num_labeled = int(len(X_train) * labeled_ratio / 100)
            labeled_idx = np.random.choice(len(X_train), num_labeled, replace=False)
            unlabeled_idx = np.setdiff1d(np.arange(len(X_train)), labeled_idx)

            X_labeled, y_labeled = X_train[labeled_idx], y_train[labeled_idx]
            X_unlabeled = X_train[unlabeled_idx]
            y_unlabeled_true = y_train[unlabeled_idx]  # Lấy nhãn thực tế của dữ liệu chưa có nhãn để so sánh

            total_pseudo_labels = 0  # Tổng số nhãn giả được thêm vào
            for iteration in range(max_iterations):
                kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                accuracies, losses = [], []
                training_progress = st.progress(0)
                training_status = st.empty()

                num = 0
                total_steps = k_folds * max_iterations
                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_labeled, y_labeled)):
                    X_k_train, X_k_val = X_labeled[train_idx], X_labeled[val_idx]
                    y_k_train, y_k_val = y_labeled[train_idx], y_labeled[val_idx]

                    model = keras.Sequential([
                        layers.Input(shape=(X_k_train.shape[1],))
                    ] + [
                        layers.Dense(num_neurons, activation=activation) for _ in range(num_layers)
                    ] + [
                        layers.Dense(10, activation="softmax")
                    ])

                    if optimizer == "adam":
                        opt = keras.optimizers.Adam(learning_rate=learning_rate)
                    elif optimizer == "sgd":
                        opt = keras.optimizers.SGD(learning_rate=learning_rate)
                    else:
                        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)

                    model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])

                    start_time = time.time()
                    history = model.fit(X_k_train, y_k_train, epochs=epochs, validation_data=(X_k_val, y_k_val), verbose=0)
                    elapsed_time = time.time() - start_time

                    accuracies.append(history.history["val_accuracy"][-1])
                    losses.append(history.history["val_loss"][-1])
                    num += 1
                    progress_percent = int((num / k_folds) * 100)

                    training_progress.progress(progress_percent)
                    training_status.text(f"⏳ Đang huấn luyện... {progress_percent}%")

                avg_val_accuracy = np.mean(accuracies)
                avg_val_loss = np.mean(losses)

                mlflow.log_metrics({
                    "avg_val_accuracy": avg_val_accuracy,
                    "avg_val_loss": avg_val_loss,
                    "elapsed_time": elapsed_time
                })
                pseudo_preds = model.predict(X_unlabeled)
                pseudo_labels = np.argmax(pseudo_preds, axis=1)
                confidence_scores = np.max(pseudo_preds, axis=1)
                confident_mask = confidence_scores > confidence_threshold

                num_pseudo_added = np.sum(confident_mask)
                total_pseudo_labels += num_pseudo_added

                # Tính số nhãn giả đúng và sai
                pseudo_labels_confident = pseudo_labels[confident_mask]
                y_unlabeled_true_confident = y_unlabeled_true[confident_mask]
                correct_pseudo_labels = np.sum(pseudo_labels_confident == y_unlabeled_true_confident)
                incorrect_pseudo_labels = num_pseudo_added - correct_pseudo_labels

                X_labeled = np.concatenate([X_labeled, X_unlabeled[confident_mask]])
                y_labeled = np.concatenate([y_labeled, pseudo_labels[confident_mask]])
                X_unlabeled = X_unlabeled[~confident_mask]
                y_unlabeled_true = y_unlabeled_true[~confident_mask]  # Cập nhật nhãn thực tế của dữ liệu chưa có nhãn

                # Đánh giá mô hình trên tập test sau khi gán nhãn giả
                test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

                # Lưu kết quả huấn luyện vào session_state
                st.session_state["training_results"].append({
                    "iteration": iteration + 1,
                    "num_pseudo_added": num_pseudo_added,
                    "correct_pseudo_labels": correct_pseudo_labels,
                    "incorrect_pseudo_labels": incorrect_pseudo_labels,
                    "total_labeled": len(X_labeled),
                    "total_pseudo_labels": total_pseudo_labels,
                    "remaining_unlabeled": len(X_unlabeled),
                    "test_accuracy": test_accuracy
                })

                st.write(f"**Vòng lặp {iteration+1}:**")
                st.write(f"- **Gán nhãn giả cho {num_pseudo_added} mẫu với độ tin cậy ≥ {confidence_threshold}:**")
                st.write(f"  - Số nhãn giả đúng: {correct_pseudo_labels}")
                st.write(f"  - Số nhãn giả sai: {incorrect_pseudo_labels}")
                st.write(f"- **Số ảnh đã gán nhãn:** {len(X_labeled)}")
                st.write(f"- **Số ảnh chưa gán nhãn:** {len(X_unlabeled)}")
                st.write(f"- **Độ chính xác trên tập test:** {test_accuracy:.4f}")

                # Dự đoán và hiển thị 10 ảnh từ tập test
                st.subheader(f"Dự đoán 10 ảnh từ tập test sau vòng lặp {iteration+1}")
                indices = np.random.choice(len(X_test), 10, replace=False)
                X_test_samples = X_test[indices]
                y_test_samples = y_test[indices]

                predictions = model.predict(X_test_samples)
                predicted_labels = np.argmax(predictions, axis=1)

                # Tính số lượng nhãn dự đoán đúng và sai
                correct_predictions = np.sum(predicted_labels == y_test_samples)
                incorrect_predictions = len(y_test_samples) - correct_predictions

                # Hiển thị số lượng nhãn dự đoán đúng và sai trong kết quả vòng lặp
                st.write(f"- **Số lượng nhãn dự đoán đúng (trong 10 ảnh):** {correct_predictions}")
                st.write(f"- **Số lượng nhãn dự đoán sai (trong 10 ảnh):** {incorrect_predictions}")
                st.write("---")

                fig, axes = plt.subplots(2, 5, figsize=(15, 6))
                axes = axes.ravel()
                for i in range(10):
                    axes[i].imshow(X_test_samples[i].reshape(28, 28), cmap='gray')
                    axes[i].set_title(f"Thực tế: {y_test_samples[i]}\nDự đoán: {predicted_labels[i]}")
                    axes[i].axis('off')
                plt.tight_layout()

                # Lưu hình ảnh dự đoán và thông tin đúng/sai vào session_state
                st.session_state["prediction_images"].append({
                    "iteration": iteration + 1,
                    "figure": fig,
                    "correct_predictions": correct_predictions,
                    "incorrect_predictions": incorrect_predictions
                })
                st.pyplot(fig)

                # Lưu độ chính xác vào MLflow để theo dõi
                mlflow.log_metrics({
                    f"test_accuracy_iter_{iteration+1}": test_accuracy,
                    f"correct_predictions_iter_{iteration+1}": correct_predictions,
                    f"incorrect_predictions_iter_{iteration+1}": incorrect_predictions,
                    f"correct_pseudo_labels_iter_{iteration+1}": correct_pseudo_labels,
                    f"incorrect_pseudo_labels_iter_{iteration+1}": incorrect_pseudo_labels
                })
                if len(X_unlabeled) == 0:
                    break

            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            mlflow.log_metrics({"test_accuracy": test_accuracy, "test_loss": test_loss})

            # Lưu độ chính xác cuối cùng vào session_state
            st.session_state["final_metrics"] = {
                "avg_val_accuracy": avg_val_accuracy,
                "test_accuracy": test_accuracy
            }

            mlflow.end_run()
            st.session_state["trained_model"] = model

            # Hoàn thành tiến trình
            training_progress.progress(100)
            training_status.text("✅ Huấn luyện hoàn tất!")

            st.success(f"✅ Huấn luyện hoàn tất!")
    
            st.write(f"📊 **Độ chính xác trên tập test:** {test_accuracy:.4f}")
       
            st.success(f"✅ Đã log dữ liệu cho Experiments Neural_Network với Name: **Train_{st.session_state['run_name']}**!")
            st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")
# Xử lý ảnh từ canvas
def preprocess_canvas_image(canvas_result):
    """Chuyển đổi ảnh từ canvas sang định dạng phù hợp để dự đoán."""
    if canvas_result.image_data is None:
        return None
    img = canvas_result.image_data[:, :, :3]  # Chỉ lấy 3 kênh RGB
    img = Image.fromarray(img).convert("L").resize((28, 28))  # Chuyển sang grayscale, resize về 28x28
    img = np.array(img) / 255.0  # Chuẩn hóa về [0,1]
    img = img.reshape(1, -1)  # Đưa về dạng vector giống như trong `thi_nghiem()`
    return img

# Hàm dự đoán
def du_doan():
    st.header("✍️ Vẽ số để dự đoán")

    # 📥 Load mô hình đã huấn luyện
    if "trained_model" in st.session_state:
        model = st.session_state["trained_model"]
        st.success("✅ Đã sử dụng mô hình vừa huấn luyện!")
    else:
        st.error("⚠️ Chưa có mô hình! Hãy huấn luyện trước.")


    # 🆕 Cập nhật key cho canvas khi nhấn "Tải lại"
    if "key_value" not in st.session_state:
        st.session_state.key_value = str(random.randint(0, 1000000))  

    if st.button("🔄 Tải lại nếu không thấy canvas"):
        st.session_state.key_value = str(random.randint(0, 1000000))  

    # ✍️ Vẽ số
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

            # Dự đoán số
            prediction = model.predict(img)
            predicted_number = np.argmax(prediction, axis=1)[0]
            max_confidence = np.max(prediction)

            st.subheader(f"🔢 Dự đoán: {predicted_number}")
            st.write(f"📊 Mức độ tin cậy: {max_confidence:.2%}")

            # Hiển thị bảng confidence score
            prob_df = pd.DataFrame(prediction.reshape(1, -1), columns=[str(i) for i in range(10)]).T
            prob_df.columns = ["Mức độ tin cậy"]
            st.bar_chart(prob_df)

        else:
            st.error("⚠️ Hãy vẽ một số trước khi bấm Dự đoán!")

from datetime import datetime

def show_experiment_selector():
    st.title("📊 MLflow Experiments - DAGsHub")

    # Lấy danh sách tất cả experiments
    experiments = mlflow.search_experiments()
    experiment_names = [exp.name for exp in experiments]    
    # Tìm experiment theo tên
    
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

    # Lấy danh sách runs trong experiment
    runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])

    if runs.empty:
        st.warning("⚠ Không có runs nào trong experiment này.")
        return

    st.write("### 🏃‍♂️ Các Runs gần đây:")

    # Lấy danh sách run_name từ params
    run_info = []
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_params = mlflow.get_run(run_id).data.params
        run_name = run_params.get("run_name", f"Run {run_id[:8]}")  # Nếu không có tên, lấy 8 ký tự đầu của ID
        run_info.append((run_name, run_id))
    # Đảm bảo danh sách run_info được sắp xếp theo thời gian chạy gần nhất
    run_info.sort(key=lambda x: mlflow.get_run(x[1]).info.start_time, reverse=True)
    
    # Tạo dictionary để map run_name -> run_id
    # Lấy run gần nhất
    if run_info:
        latest_run_name, latest_run_id = run_info[0]  # Chọn run mới nhất
        selected_run_name = latest_run_name
        selected_run_id = latest_run_id
    else:
        st.warning("⚠ Không có runs nào trong experiment này.")
        return

    # Hiển thị thông tin chi tiết của run được chọn
    selected_run = mlflow.get_run(selected_run_id)

    if selected_run:
        st.subheader(f"📌 Thông tin Run: {selected_run_name}")
        st.write(f"**Run ID:** {selected_run_id}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")
        start_time_ms = selected_run.info.start_time  # Thời gian lưu dưới dạng milliseconds

        # Chuyển sang định dạng ngày giờ dễ đọc
        if start_time_ms:
            start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
        else:
            start_time = "Không có thông tin"

        st.write(f"**Thời gian chạy:** {start_time}")

        # Hiển thị thông số đã log
        params = selected_run.data.params
        metrics = selected_run.data.metrics

        if params:
            st.write("### ⚙️ Parameters:")
            st.json(params)

        if metrics:
            st.write("### 📊 Metrics:")
            st.json(metrics)

        # Kiểm tra và hiển thị dataset artifact
        dataset_uri = f"{selected_experiment.artifact_location}/{selected_run_id}/artifacts/dataset.csv" 
        try:
            mlflow.artifacts.download_artifacts(dataset_uri)
            st.write("### 📂 Dataset:")
            st.write(f"📥 [Tải dataset]({dataset_uri})")
        except Exception as e:
            st.warning("⚠ Không tìm thấy dataset.csv trong artifacts.")
# Hàm lý thuyết Neural Network
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Hàm lý Lý thuyết về Pseudo Labelling
import streamlit as st

def explain_Pseudo_Labelling():
    st.markdown("## 📚 Lý thuyết về Pseudo Labelling")

    # Giới thiệu tổng quan
    st.markdown("""
    **Pseudo Labelling** (Gán nhãn giả) là một kỹ thuật học bán giám sát (semi-supervised learning) được sử dụng để tận dụng dữ liệu chưa có nhãn (unlabeled data) trong quá trình huấn luyện mô hình học máy. Ý tưởng chính là sử dụng mô hình đã được huấn luyện trên một tập dữ liệu có nhãn nhỏ để dự đoán nhãn cho dữ liệu chưa có nhãn, sau đó sử dụng các nhãn giả này để mở rộng tập huấn luyện và tiếp tục huấn luyện mô hình.
    """)

    # Khi nào sử dụng Pseudo Labelling
    st.subheader("🔍 Khi nào sử dụng Pseudo Labelling?")
    st.markdown("""
    Pseudo Labelling thường được áp dụng trong các tình huống sau:
    - **Dữ liệu có nhãn hạn chế:** Khi bạn chỉ có một lượng nhỏ dữ liệu có nhãn (labeled data) nhưng có rất nhiều dữ liệu chưa có nhãn (unlabeled data).
    - **Dữ liệu chưa nhãn có giá trị:** Dữ liệu chưa nhãn có thể cung cấp thông tin bổ sung để cải thiện hiệu suất của mô hình.
    - **Mô hình có độ chính xác ban đầu tốt:** Mô hình ban đầu (huấn luyện trên tập dữ liệu có nhãn nhỏ) cần có khả năng dự đoán nhãn giả đủ đáng tin cậy.
    """)

    # Cách hoạt động của Pseudo Labelling
    st.subheader("⚙️ Cách hoạt động của Pseudo Labelling")
    st.markdown("""
    Quy trình của Pseudo Labelling thường bao gồm các bước sau:
    1. **Huấn luyện ban đầu:**
       - Sử dụng một tập dữ liệu nhỏ có nhãn (labeled data) để huấn luyện mô hình ban đầu.
       - Ví dụ: Với tập dữ liệu MNIST, bạn có thể lấy 1% dữ liệu có nhãn (khoảng 600 ảnh từ 60,000 ảnh train).
    2. **Dự đoán nhãn giả:**
       - Sử dụng mô hình đã huấn luyện để dự đoán nhãn cho dữ liệu chưa có nhãn.
       - Kết quả dự đoán thường là xác suất cho từng lớp (ví dụ: [0.1, 0.85, 0.05] cho 3 lớp).
    3. **Lọc dữ liệu tin cậy:**
       - Chọn các mẫu có độ tin cậy cao (dựa trên ngưỡng xác suất, ví dụ: xác suất lớn nhất ≥ 0.95).
       - Ví dụ: Nếu xác suất dự đoán cho lớp "5" là 0.98 (> 0.95), gán nhãn giả "5" cho mẫu đó.
    4. **Mở rộng tập huấn luyện:**
       - Thêm các mẫu vừa được gán nhãn giả vào tập dữ liệu có nhãn ban đầu.
    5. **Lặp lại:**
       - Huấn luyện lại mô hình trên tập dữ liệu mới (gồm dữ liệu có nhãn ban đầu + dữ liệu gán nhãn giả).
       - Lặp lại quá trình cho đến khi:
         - Hết dữ liệu chưa nhãn.
         - Đạt số lần lặp tối đa.
         - Hiệu suất mô hình không cải thiện thêm.
    """)

    # Ưu điểm và nhược điểm
    st.subheader("✅ Ưu điểm và ⚠️ Nhược điểm")
    st.markdown("""
    ### Ưu điểm:
    - **Tận dụng dữ liệu chưa nhãn:** Giúp cải thiện hiệu suất mô hình khi dữ liệu có nhãn hạn chế.
    - **Đơn giản và hiệu quả:** Dễ triển khai, không yêu cầu các kỹ thuật phức tạp.
    - **Tăng độ chính xác:** Nếu nhãn giả được dự đoán chính xác, mô hình sẽ học được từ dữ liệu mới và cải thiện hiệu suất.

    ### Nhược điểm:
    - **Phụ thuộc vào mô hình ban đầu:** Nếu mô hình ban đầu dự đoán sai nhiều, nhãn giả sẽ không chính xác, dẫn đến hiệu ứng "lỗi tích lũy" (error propagation).
    - **Ngưỡng lựa chọn:** Việc chọn ngưỡng xác suất (threshold) là một thách thức. Ngưỡng quá cao có thể bỏ sót nhiều mẫu, ngưỡng quá thấp có thể gán nhãn sai.
    - **Tốn tài nguyên:** Quá trình lặp lại và huấn luyện nhiều lần có thể tốn thời gian và tài nguyên tính toán.
    """)

    # Ứng dụng thực tế
    st.subheader("🌟 Ứng dụng thực tế")
    st.markdown("""
    Pseudo Labelling thường được sử dụng trong các bài toán sau:
    - **Phân loại ảnh:** Ví dụ, trên tập dữ liệu MNIST (chữ số viết tay), nơi dữ liệu có nhãn ít nhưng dữ liệu chưa nhãn dồi dào.
    - **Xử lý ngôn ngữ tự nhiên (NLP):** Gán nhãn cho văn bản chưa có nhãn (ví dụ: phân loại cảm xúc, nhận diện thực thể).
    - **Y học:** Sử dụng dữ liệu y tế chưa có nhãn để cải thiện mô hình chẩn đoán bệnh.
    """)

    # Ví dụ minh họa
    st.subheader("📊 Ví dụ minh họa trên MNIST")
    st.markdown("""
    Giả sử bạn có tập dữ liệu MNIST với 60,000 ảnh train (có nhãn) và 10,000 ảnh test:
    1. Lấy 1% dữ liệu có nhãn (600 ảnh, 60 ảnh mỗi lớp từ 0-9).
    2. Huấn luyện một Neural Network trên 600 ảnh này.
    3. Dự đoán nhãn cho 59,400 ảnh còn lại (99% dữ liệu train).
    4. Chọn ngưỡng xác suất là 0.95:
       - Nếu một ảnh có xác suất cao nhất ≥ 0.95, gán nhãn giả cho ảnh đó.
       - Ví dụ: Dự đoán [0.01, 0.02, 0.95, ...] → Gán nhãn "2".
    5. Thêm các ảnh được gán nhãn giả vào tập huấn luyện (600 ảnh ban đầu + ảnh mới).
    6. Lặp lại quá trình cho đến khi gán nhãn hết 60,000 ảnh hoặc đạt số lần lặp tối đa.
    """)

    # Kết luận
    st.subheader("🎯 Kết luận")
    st.markdown("""
    Pseudo Labelling là một kỹ thuật mạnh mẽ trong học bán giám sát, giúp tận dụng dữ liệu chưa có nhãn để cải thiện hiệu suất mô hình. Tuy nhiên, cần cẩn thận khi chọn ngưỡng xác suất và đảm bảo mô hình ban đầu có độ chính xác đủ tốt để tránh lỗi tích lũy. Kỹ thuật này đặc biệt hữu ích trong các bài toán thực tế như phân loại ảnh (MNIST) hoặc xử lý ngôn ngữ tự nhiên.
    """)

def show_prediction_table():
    st.table({
        "Ảnh": ["Ảnh 1", "Ảnh 2", "Ảnh 3", "Ảnh 4", "Ảnh 5"],
        "Dự đoán": [7, 2, 3, 5, 8],
        "Xác suất": [0.98, 0.85, 0.96, 0.88, 0.97],
        "Gán nhãn?": ["✅", "❌", "✅", "❌", "✅"]
    })
# Hàm hiển thị thông tin về MNIST
def data():
    st.title("Khám Phá Bộ Dữ Liệu MNIST")
   

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
def pseudo_labelling():
    st.title("Phụ Giúp Phân L��p")
    st.subheader("Phụ giúp Neural Network phân l��p dữ liệu MNIST")
def Classification():
    if "mlflow_initialized" not in st.session_state:
        DAGSHUB_MLFLOW_URI = "https://dagshub.com/PTToan250303/Linear_replication.mlflow"
        mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
        st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
        os.environ["MLFLOW_TRACKING_USERNAME"] = "PTToan250303"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "5ca8caf353d564c358852da97c7487e64fc30a73"
        mlflow.set_experiment("Semi_supervised_Classification")
    st.markdown("""
        <style>
        .title { font-size: 48px; font-weight: bold; text-align: center; color: #4682B4; margin-top: 50px; }
        .subtitle { font-size: 24px; text-align: center; color: #4A4A4A; }
        hr { border: 1px solid #ddd; }
        </style>
        <div class="title">MNIST Semi supervised App</div>
        <hr>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5,tab6 = st.tabs(["📘 Lý thuyết Neural Network", "📘 Data", "⚙️ Huấn luyện", "🔢 Dự đoán", "🔥 Mlflow","🎯 Pseudo Labelling"])

    with tab1:
        explain_Pseudo_Labelling()
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
        pseudo_labelling()

if __name__ == "__main__":
   Classification()