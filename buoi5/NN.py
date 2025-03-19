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
   
    num=0
    if "X_train" not in st.session_state:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return
    
    X_train, X_val, X_test = [st.session_state[k].reshape(-1, 28 * 28) / 255.0 for k in ["X_train", "X_val", "X_test"]]
    y_train, y_val, y_test = [st.session_state[k] for k in ["y_train", "y_val", "y_test"]]
    
    k_folds = st.slider("Số fold cho Cross-Validation:", 3, 10, 5)
    num_layers = st.slider("Số lớp ẩn:", 1, 5, 2)
    num_neurons = st.slider("Số neuron mỗi lớp:", 32, 512, 128, 32)
    activation = st.selectbox("Hàm kích hoạt:", ["relu", "sigmoid", "tanh"])
    optimizer = st.selectbox("Optimizer:", ["adam", "sgd", "rmsprop"])
    epochs = st.slider("🕰 Số epochs:", min_value=1, max_value=50, value=20, step=1)
    learning_rate = st.slider("⚡ Tốc độ học (Learning Rate):", min_value=1e-5, max_value=1e-1, value=1e-3, step=1e-5, format="%.5f")

    loss_fn = "sparse_categorical_crossentropy"
    # Chỉ nhập tên Experiment (Không có phần nhập tên Run)
    if "experiment_name" not in st.session_state:
        st.session_state["experiment_name"] = "My_Experiment"

    experiment_name = st.text_input("🔹 Nhập tên Experiment:", st.session_state["experiment_name"], key="experiment_name_input")    

    if experiment_name:
        st.session_state["experiment_name"] = experiment_name

    mlflow.set_experiment(experiment_name)
    st.write(f"✅ Experiment Name: {experiment_name}")
    
    if st.button("🚀 Huấn luyện mô hình"):
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
                "epochs": epochs
            })

            kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            accuracies, losses = [], []

            # Thanh tiến trình tổng quát cho toàn bộ quá trình huấn luyện
            training_progress = st.progress(0)
            training_status = st.empty()

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
                X_k_train, X_k_val = X_train[train_idx], X_train[val_idx]
                y_k_train, y_k_val = y_train[train_idx], y_train[val_idx]

                model = keras.Sequential([
                    layers.Input(shape=(X_k_train.shape[1],))
                ] + [
                    layers.Dense(num_neurons, activation=activation) for _ in range(num_layers)
                ] + [
                    layers.Dense(10, activation="softmax")
                ])

                # Chọn optimizer với learning rate
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

                # Cập nhật thanh tiến trình chính (theo fold)
                
                
                progress_percent = int((num / k_folds)*100)
                
                num = num +1
                training_progress.progress(progress_percent)
                
                            

                
                training_status.text(f"⏳ Đang huấn luyện... {progress_percent}%")

            avg_val_accuracy = np.mean(accuracies)
            avg_val_loss = np.mean(losses)

            mlflow.log_metrics({
                "avg_val_accuracy": avg_val_accuracy,
                "avg_val_loss": avg_val_loss,
                "elapsed_time": elapsed_time
            })

            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            mlflow.log_metrics({"test_accuracy": test_accuracy, "test_loss": test_loss})

            mlflow.end_run()
            st.session_state["trained_model"] = model

            # Hoàn thành tiến trình
            training_progress.progress(1.0)
            training_status.text("✅ Huấn luyện hoàn tất!")

            st.success(f"✅ Huấn luyện hoàn tất!")
            st.write(f"📊 **Độ chính xác trung bình trên tập validation:** {avg_val_accuracy:.4f}")
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


# Hàm lý thuyết Neural Network dựa trên tài liệu
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# Hàm lý thuyết Neural Network
def explain_neural_network():
    # Tiêu đề chính
    st.title("🧠 Hiểu Biết Cơ Bản Về Mạng Nơ-ron Nhân Tạo")
    gif_path = "buoi5/g1.gif"  
    st.image(gif_path, caption="Hình ảnh minh họa dữ liệu MNIST", use_container_width="auto")
    # Giới thiệu
    st.markdown("""
    **Mạng nơ-ron nhân tạo (Artificial Neural Network - ANN)** là một mô hình tính toán được lấy cảm hứng từ cách hoạt động của não bộ con người. Nó bao gồm nhiều đơn vị xử lý gọi là nơ-ron, được liên kết với nhau qua các lớp (layers), cho phép mô hình học hỏi và nhận diện các đặc điểm hoặc quy luật từ dữ liệu.
    """)

    # Cấu trúc mạng nơ-ron
    st.subheader("🔍 Cấu trúc chính của mạng nơ-ron")
    st.markdown("""
    Mạng nơ-ron thường được chia thành ba phần cơ bản:
    1. **Lớp đầu vào (Input Layer):** Nơi dữ liệu được đưa vào hệ thống.
    2. **Lớp ẩn (Hidden Layers):** Các lớp trung gian chịu trách nhiệm xử lý thông tin bằng cách sử dụng các trọng số (weights) và hàm kích hoạt (activation function).
    3. **Lớp đầu ra (Output Layer):** Đưa ra kết quả cuối cùng, chẳng hạn như dự đoán hoặc phân loại.
    
    *Ví dụ:* Nếu bạn tưởng tượng mạng nơ-ron như một nhà máy, lớp đầu vào là nguyên liệu thô, các lớp ẩn là dây chuyền sản xuất, và lớp đầu ra là sản phẩm hoàn thiện.
    """)
    st.image("buoi5/oXvOtJt.png", caption="Cấu trúc mạng nơ-ron(mmlab.uit.edu.vn)", use_container_width="auto")
    # Ghi chú: Nếu có hình ảnh, bạn có thể thêm bằng st.image("đường_dẫn_hình_ảnh")

    # Cách hoạt động của nơ-ron
    st.subheader("⚙️ Cách hoạt động của một nơ-ron")
    st.markdown("""
    Mỗi nơ-ron trong mạng nhận tín hiệu từ các nơ-ron ở lớp trước, nhân chúng với các trọng số, cộng thêm một giá trị gọi là **bias**, rồi áp dụng một hàm kích hoạt để quyết định tín hiệu nào sẽ được truyền tiếp.
    """)
    st.markdown("### Công thức cơ bản của một nơ-ron:")
    st.latex(r"z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b")
    st.markdown("""
    Trong đó:
    - $$ x_1, x_2, \dots, x_n $$: Các giá trị đầu vào.
    - $$ w_1, w_2, \dots, w_n $$: Trọng số tương ứng.
    - $$ b $$: Giá trị bias.
    - $$ z $$: Tổng có trọng số.
    
    Sau khi tính $$ z $$, giá trị này sẽ được đưa qua một **hàm kích hoạt** để tạo ra đầu ra cuối cùng của nơ-ron.
    """)

    # Hàm kích hoạt
    st.subheader("🎯 Các hàm kích hoạt phổ biến")
    st.markdown("""
    Hàm kích hoạt đóng vai trò quan trọng trong việc giúp mạng nơ-ron xử lý các vấn đề phức tạp, đặc biệt là những mối quan hệ phi tuyến tính trong dữ liệu. Dưới đây là một số hàm phổ biến:
    """)
    st.image("buoi5/tmkfP14.png", caption="hàm kích hoạt của Sigmoid và Tanh ", use_container_width="auto")
    st.markdown("""
    1. **Sigmoid:** Biến đổi đầu vào thành giá trị từ 0 đến 1, thường dùng cho bài toán phân loại hai lớp.
    """)
    st.latex(r"f(z) = \frac{1}{1 + e^{-z}}")
    
    st.markdown("""
    2. **Tanh:** Đưa đầu ra vào khoảng từ -1 đến 1, phù hợp với dữ liệu có giá trị âm và dương.
    """)
    st.latex(r"f(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}")
    

    st.markdown("""
    3. **ReLU:** Đơn giản nhưng hiệu quả, trả về 0 nếu đầu vào âm và giữ nguyên giá trị nếu dương.
    """)
    st.latex(r"f(z) = \max(0, z)")
    st.image("buoi5/UmoHHfH.png", caption="hàm kích hoạt của ReLU", use_container_width="auto")
    # Quá trình học
    st.subheader("🔄 Quá trình học của mạng nơ-ron")
    st.markdown("""
    Mạng nơ-ron học thông qua việc điều chỉnh trọng số dựa trên hai bước chính: **lan truyền thuận** và **lan truyền ngược**.
    """)

    # Lan truyền thuận
    st.markdown("#### 1. Lan truyền thuận (Forward Propagation)")
    st.markdown("""
    Dữ liệu được đưa từ lớp đầu vào qua các lớp ẩn, rồi đến lớp đầu ra. Mỗi lớp thực hiện phép tính:
    """)
    st.latex(r"f^{(l)} = \sigma(W^{(l)} f^{(l-1)} + b^{(l)})")
    st.markdown("""
    - $$ f^{(l)} $$: Đầu ra của lớp thứ $$ l $$.
    - $$ W^{(l)} $$: Ma trận trọng số của lớp $$ l $$.
    - $$ b^{(l)} $$: Bias của lớp $$ l $$.
    - $$ \sigma $$: Hàm kích hoạt.
    """)

    # Tính toán sai số
    st.markdown("#### 2. Tính toán sai số (Loss Function)")
    st.markdown("""
    Sai số giữa kết quả dự đoán và giá trị thực tế được đo bằng hàm mất mát, ví dụ:
    - **Mean Squared Error (MSE):** Dùng cho bài toán hồi quy:
    """)
    st.latex(r"L = \frac{1}{N} \sum (y_{thực} - y_{dự đoán})^2")
    st.markdown("""
    - **Cross-Entropy Loss:** Dùng cho bài toán phân loại:
    """)
    st.latex(r"L = - \sum y_{thực} \log(y_{dự đoán})")

    # Lan truyền ngược
    st.markdown("#### 3. Lan truyền ngược (Backpropagation)")
    st.markdown("""
    Mạng sử dụng đạo hàm của hàm mất mát để điều chỉnh trọng số:
    """)
    st.latex(r"\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial a^{(l)}} \cdot \frac{\partial a^{(l)}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}")
    st.markdown("""
    Quá trình này giúp mạng “học” bằng cách giảm dần sai số.
    """)

    # Tối ưu hóa
    st.markdown("#### 4. Tối ưu hóa trọng số")
    st.markdown("""
    Để cập nhật trọng số, các thuật toán tối ưu được sử dụng:
    - **Gradient Descent:** Di chuyển trọng số theo hướng giảm gradient:
    """)
    st.latex(r"W^{(l)} = W^{(l)} - \alpha \frac{\partial L}{\partial W^{(l)}}")
    st.markdown("""
    - **Adam:** Kết hợp động lượng và điều chỉnh tốc độ học:
    """)
    st.latex(r"W^{(l)} = W^{(l)} - \alpha \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon}")

    # Kết luận
    st.subheader("🌟 Tổng kết")
    st.markdown("""
    Mạng nơ-ron nhân tạo là một công cụ mạnh mẽ trong học máy, có khả năng học hỏi từ dữ liệu phức tạp. Việc nắm rõ cách nó hoạt động – từ cấu trúc, hàm kích hoạt, đến quá trình huấn luyện – là chìa khóa để áp dụng và cải thiện hiệu suất của mô hình trong thực tế.
    """)
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

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📘 Lý thuyết Neural Network", "📘 Data", "⚙️ Huấn luyện", "🔢 Dự đoán", "🔥 Mlflow"])

    with tab1:
        explain_neural_network()
    with tab2:
        data()
    with tab3:
        split_data()
        train()
    with tab4:
        du_doan()
    with tab5:
        show_experiment_selector()

if __name__ == "__main__":
   Classification()