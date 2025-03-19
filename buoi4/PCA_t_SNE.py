import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml

def explain_pca():
    st.markdown("## 🧠 Hiểu PCA một cách đơn giản")

    st.markdown("""
    **PCA (Phân tích thành phần chính)** là một phương pháp giúp giảm số chiều của dữ liệu mà vẫn giữ được thông tin quan trọng nhất.  
    Hãy tưởng tượng bạn có một tập dữ liệu nhiều chiều (nhiều cột), nhưng bạn muốn biểu diễn nó trong không gian 2D hoặc 3D để dễ hiểu hơn. PCA giúp bạn làm điều đó!  

    ### 🔹 **Ví dụ trực quan**:
    Hãy tưởng tượng bạn có một tập dữ liệu gồm nhiều điểm phân bố theo một đường chéo trong không gian 2D:
    """)

   
    np.random.seed(42)
    x = np.random.rand(100) * 10  
    y = x * 0.8 + np.random.randn(100) * 2  
    X = np.column_stack((x, y))

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], color="blue", alpha=0.5, label="Dữ liệu ban đầu")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.legend()
    st.pyplot(fig)

    st.markdown(r"""
    ## 📌 PCA - Giải thích Trực Quan  
    Dữ liệu này có sự phân tán rõ ràng theo một hướng chính. PCA sẽ tìm ra hướng đó để biểu diễn dữ liệu một cách tối ưu.

    ---

    ### 🔹 **Các bước thực hiện PCA dễ hiểu**

    1️⃣ **Tìm điểm trung tâm (mean vector)**  
    - Trước tiên, tính giá trị trung bình của từng đặc trưng (feature) trong tập dữ liệu.  
    - Vector trung bình này giúp xác định "trung tâm" của dữ liệu.  
    $$ 
    \mu = \frac{1}{n} \sum_{i=1}^{n} x_i 
    $$  
    - Trong đó:
        - $$ n $$ là số lượng mẫu dữ liệu.
        - $$ x_i $$ là từng điểm dữ liệu.

    2️⃣ **Dịch chuyển dữ liệu về gốc tọa độ**  
    - Để đảm bảo phân tích chính xác hơn, ta dịch chuyển dữ liệu sao cho trung tâm của nó nằm tại gốc tọa độ bằng cách trừ đi vector trung bình:  
    $$ 
    X_{\text{norm}} = X - \mu
    $$  
    - Khi đó, dữ liệu sẽ có giá trị trung bình bằng 0.

    3️⃣ **Tính ma trận hiệp phương sai**  
    - Ma trận hiệp phương sai giúp đo lường mức độ biến thiên giữa các đặc trưng:  
    $$ 
    C = \frac{1}{n} X_{\text{norm}}^T X_{\text{norm}}
    $$  
    - Ý nghĩa:
        - Nếu phần tử $$ C_{ij} $$ có giá trị lớn → Hai đặc trưng $$ i $$ và $$ j $$ có mối tương quan mạnh.
        - Nếu $$ C_{ij} $$ gần 0 → Hai đặc trưng không liên quan nhiều.

    4️⃣ **Tìm các hướng quan trọng nhất**  
    - Tính trị riêng (eigenvalues) và vector riêng (eigenvectors) từ ma trận hiệp phương sai:  
    $$ 
    C v = \lambda v
    $$  
    - Trong đó:
        - $$ v $$ là vector riêng (eigenvector) - đại diện cho các hướng chính của dữ liệu.
        - $$ \lambda $$ là trị riêng (eigenvalue) - thể hiện độ quan trọng của từng hướng.
    - Vector riêng có trị riêng lớn hơn sẽ mang nhiều thông tin quan trọng hơn.

    5️⃣ **Chọn số chiều mới và tạo không gian con**  
    - Chọn $$ K $$ vector riêng tương ứng với $$ K $$ trị riêng lớn nhất để tạo ma trận $$ U_K $$:  
    $$ 
    U_K = [v_1, v_2, ..., v_K]
    $$  
    - Các vector này tạo thành hệ trực giao và giúp ta biểu diễn dữ liệu tối ưu trong không gian mới.

    6️⃣ **Chiếu dữ liệu vào không gian mới**  
    - Biểu diễn dữ liệu trong hệ trục mới bằng cách nhân dữ liệu chuẩn hóa với ma trận $$ U_K $$:  
    $$ 
    X_{\text{new}} = X_{\text{norm}} U_K
    $$  
    - Dữ liệu mới $$ X_{\text{new}} $$ có số chiều ít hơn nhưng vẫn giữ được nhiều thông tin quan trọng.

    7️⃣ **Dữ liệu mới chính là tọa độ của các điểm trong không gian mới.**  
    - Mỗi điểm dữ liệu giờ đây được biểu diễn bằng các thành phần chính thay vì các đặc trưng ban đầu.

    ---

    ### 🔹 **Trực quan hóa quá trình PCA**  
    Dưới đây là minh họa cách PCA tìm ra trục quan trọng nhất của dữ liệu:
    """)



    # PCA thủ công
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], color="blue", alpha=0.5, label="Dữ liệu ban đầu")
    origin = np.mean(X, axis=0)

    for i in range(2):
        ax.arrow(origin[0], origin[1], 
                 eigenvectors[0, i] * 3, eigenvectors[1, i] * 3, 
                 head_width=0.3, head_length=0.3, color="red", label=f"Trục {i+1}")

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.legend()
    st.pyplot(fig)

    st.markdown("""
    **🔹 Kết quả:**  
    
    
    
    
    
    - Trục đỏ là hướng mà PCA tìm ra.  
    - Nếu chọn 1 trục chính, ta có thể chiếu dữ liệu lên nó để giảm chiều.  
      
    Nhờ đó, chúng ta có thể biểu diễn dữ liệu một cách gọn gàng hơn mà không mất quá nhiều thông tin!  
    """)
    # st.image("img1.png")  # Đường dẫn cần đúng
    st.markdown("""
        ### ✅ **Ưu điểm của PCA**  
        - **Giảm chiều dữ liệu hiệu quả**: PCA giúp giảm số chiều của dữ liệu mà vẫn giữ lại phần lớn thông tin quan trọng.  
        - **Tăng tốc độ xử lý**: Khi số chiều giảm, các mô hình học máy sẽ chạy nhanh hơn và yêu cầu ít tài nguyên hơn.  
        - **Giảm nhiễu**: PCA có thể loại bỏ các thành phần nhiễu bằng cách giữ lại các thành phần chính có phương sai cao.  
        - **Trực quan hóa dữ liệu**: PCA giúp hiển thị dữ liệu nhiều chiều dưới dạng 2D hoặc 3D để con người dễ quan sát hơn.  

        ---

        ### ❌ **Nhược điểm của PCA**  
        - **Mất thông tin**: PCA chọn những thành phần có phương sai cao nhất, có thể làm mất thông tin quan trọng.  
        - **Không phải lúc nào cũng hiệu quả**: PCA chỉ hoạt động tốt khi dữ liệu có cấu trúc tuyến tính. Với dữ liệu phi tuyến tính, t-SNE có thể tốt hơn.  
        - **Khó diễn giải**: Sau khi giảm chiều, các thành phần chính không còn giữ nguyên ý nghĩa gốc, khiến việc hiểu dữ liệu trở nên khó khăn hơn.  
        - **Ảnh hưởng bởi dữ liệu đầu vào**: PCA nhạy cảm với thang đo dữ liệu. Nếu dữ liệu chưa được chuẩn hóa, kết quả có thể bị méo mó.  
        """)

    
    
    
    
    

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
def explain_tsne():
    
    st.markdown(r"""
    ## 🌌 t-Distributed Stochastic Neighbor Embedding (t-SNE)
    t-SNE là một phương pháp giảm chiều mạnh mẽ, giúp hiển thị dữ liệu đa chiều trên mặt phẳng 2D hoặc không gian 3D bằng cách bảo toàn mối quan hệ giữa các điểm gần nhau.

    ---
    
    ### 🔹 **Nguyên lý hoạt động của t-SNE**
    
    1️⃣ **Tính xác suất điểm gần nhau trong không gian gốc**  
       - Với mỗi điểm $$ x_i $$, xác suất có điều kiện giữa $$ x_i $$ và $$ x_j $$ được tính dựa trên khoảng cách Gaussian:  
       $$ 
       p_{j|i} = \frac{\exp(-\| x_i - x_j \|^2 / 2\sigma^2)}{\sum_{k \neq i} \exp(-\| x_i - x_k \|^2 / 2\sigma^2)} 
       $$  
       - Trong đó:
         - $$ \sigma $$ là độ lệch chuẩn (bandwidth) của Gaussian Kernel.
         - Xác suất này phản ánh mức độ gần gũi của các điểm dữ liệu trong không gian ban đầu.
      
    2️⃣ **Tính xác suất trong không gian giảm chiều (2D/3D)**  
       - Trong không gian giảm chiều, t-SNE sử dụng phân phối t-Student với một mức độ tự do để giữ khoảng cách giữa các điểm:  
       $$ 
       q_{j|i} = \frac{(1 + \| y_i - y_j \|^2)^{-1}}{\sum_{k \neq i} (1 + \| y_i - y_k \|^2)^{-1}}
       $$  
       - Ý nghĩa:
         - Phân phối t-Student giúp giảm tác động của các điểm xa nhau, tạo ra cụm dữ liệu rõ hơn.
      
    3️⃣ **Tối ưu hóa khoảng cách giữa $$ p_{j|i} $$ và $$ q_{j|i} $$**  
       - t-SNE cố gắng làm cho phân phối xác suất trong không gian gốc gần bằng trong không gian mới bằng cách tối thiểu hóa **hàm mất mát Kullback-Leibler (KL divergence)**:  
       $$ 
       KL(P||Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
       $$  
       - Ý nghĩa:
         - Nếu $$ P $$ và $$ Q $$ giống nhau, KL divergence sẽ nhỏ.
         - t-SNE cập nhật tọa độ $$ y_i $$ để giảm KL divergence, giúp bảo toàn cấu trúc dữ liệu.

    ---
    
    ### 📊 **Trực quan hóa quá trình t-SNE**  
    Dưới đây là minh họa cách t-SNE biến đổi dữ liệu từ không gian gốc sang không gian giảm chiều:  
    """)

    # Trực quan hóa bằng biểu đồ matplotlib
    

    
    # st.image("buoi6/img2.png")  # Đường dẫn cần đúng

    st.markdown(r"""
    ---
    
    ### ✅ **Ưu điểm của t-SNE**
    - Tạo cụm dữ liệu rõ ràng, dễ quan sát.
    - Giữ được mối quan hệ phi tuyến tính trong dữ liệu.

    ### ❌ **Nhược điểm của t-SNE**
    - Chạy chậm hơn PCA, đặc biệt với dữ liệu lớn.
    - Nhạy cảm với tham số **perplexity** (nếu chọn sai có thể gây méo mó dữ liệu).

    ---
    
    📌 **Ghi nhớ:**  
    - t-SNE phù hợp để **trực quan hóa dữ liệu**, nhưng **không phù hợp cho giảm chiều phục vụ mô hình học máy** (do không bảo toàn cấu trúc tổng thể của dữ liệu).  
    """)

import mlflow
import os
import time
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def input_mlflow():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/PTToan250303/Linear_replication.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)

    st.session_state['mlflow_url']=DAGSHUB_MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "PTToan250303"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "5ca8caf353d564c358852da97c7487e64fc30a73"
    mlflow.set_experiment("Classification")  

def thi_nghiem():
    st.title("📉 Giảm chiều dữ liệu MNIST với PCA & t-SNE")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Go up one directory from buoi4
    x_path = os.path.join(base_dir, "buoi2", "X.npy")

    Xmt = np.load(x_path)
    ymt = np.load(os.path.join(base_dir, "buoi2", "y.npy"))
    # Load dữ liệu
    X = Xmt.reshape(Xmt.shape[0], -1) 
    y = ymt.reshape(-1) 

    # Tùy chọn thuật toán
    method = st.radio("Chọn phương pháp giảm chiều", ["PCA", "t-SNE"])
    n_components = st.slider("Số chiều giảm xuống", 2, 3, 2)
    
    # Giới hạn số mẫu để tăng tốc
    # Thanh trượt chọn số lượng mẫu sử dụng từ MNIST
    num_samples = st.slider("Chọn số lượng mẫu MNIST sử dụng:", min_value=1000, max_value=60000, value=5000, step=1000)

    # Giới hạn số mẫu để tăng tốc
    X_subset, y_subset = X[:num_samples], y[:num_samples]

    input_mlflow()
     # Chỉ nhập tên Experiment (Không có phần nhập tên Run)
    if "experiment_name" not in st.session_state:
        st.session_state["experiment_name"] = "My_Experiment"

    experiment_name = st.text_input("🔹 Nhập tên Experiment:", st.session_state["experiment_name"], key="experiment_name_input")    

    if experiment_name:
        st.session_state["experiment_name"] = experiment_name

    mlflow.set_experiment(experiment_name)
    st.write(f"✅ Experiment Name: {experiment_name}")
    
    if st.button("🚀 Chạy giảm chiều"):
        if "run_name" not in st.session_state:
            st.session_state["run_name"] = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}" 
        with st.spinner("Đang xử lý..."):
            mlflow.start_run(run_name=st.session_state["run_name"])
            mlflow.log_param("method", method)
            mlflow.log_param("n_components", n_components)
            mlflow.log_param("num_samples", num_samples)
            mlflow.log_param("original_dim", X.shape[1])
            
            if method == "t-SNE":
                perplexity = min(30, num_samples - 1)
                mlflow.log_param("perplexity", perplexity)
                reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
            else:
                reducer = PCA(n_components=n_components)
            
            start_time = time.time()
            X_reduced = reducer.fit_transform(X_subset)
            elapsed_time = time.time() - start_time
            mlflow.log_metric("elapsed_time", elapsed_time)
            
            if method == "PCA":
                explained_variance = np.sum(reducer.explained_variance_ratio_)
                mlflow.log_metric("explained_variance_ratio", explained_variance)
            elif method == "t-SNE" and hasattr(reducer, "kl_divergence_"):
                mlflow.log_metric("KL_divergence", reducer.kl_divergence_)
            
            # Hiển thị kết quả
            if n_components == 2:
                fig = px.scatter(x=X_reduced[:, 0], y=X_reduced[:, 1], color=y_subset.astype(str),
                                 title=f"{method} giảm chiều xuống {n_components}D",
                                 labels={'x': "Thành phần 1", 'y': "Thành phần 2"})
            else:
                fig = px.scatter_3d(x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2],
                                     color=y_subset.astype(str),
                                     title=f"{method} giảm chiều xuống {n_components}D",
                                     labels={'x': "Thành phần 1", 'y': "Thành phần 2", 'z': "Thành phần 3"})

            st.plotly_chart(fig)
            
            # Lưu kết quả vào MLflow
            os.makedirs("logs", exist_ok=True)
            fig_path = f"logs/{method}_{n_components}D.png"
            fig.write_image(fig_path)
            mlflow.log_artifact(fig_path)
            
            np.save(f"logs/{method}_X_reduced.npy", X_reduced)
            mlflow.log_artifact(f"logs/{method}_X_reduced.npy")
           
            mlflow.end_run()
            st.success(f"✅ Đã log dữ liệu cho **Train_{st.session_state['run_name']}**!")
            st.markdown(f"### 🔗 [Truy cập MLflow DAGsHub]({st.session_state['mlflow_url']})")
            st.success("Hoàn thành!")
    
    
from datetime import datetime    
import streamlit as st
import mlflow
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


        
        
        
          
import mlflow
import os
from mlflow.tracking import MlflowClient
def pca_tsne():
    #st.title("🚀 MLflow DAGsHub Tracking với Streamlit")
    
    st.markdown("""
        <style>
        .title { font-size: 48px; font-weight: bold; text-align: center; color: #4682B4; margin-top: 50px; }
        .subtitle { font-size: 24px; text-align: center; color: #4A4A4A; }
        hr { border: 1px solid #ddd; }
        </style>
        <div class="title">MNIST PCA_t_SNE App</div>
        <hr>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3,tab4 = st.tabs(["📘 Lý thuyết PCA", "📘 Lý thuyết t-NSE", "📘 Giảm chiều","🔥 Mlflow"] )

    with tab1:
        explain_pca()

    with tab2:
        explain_tsne()
    
    with tab3:
        thi_nghiem()
    with tab4:
        show_experiment_selector()


if __name__ == "__main__":
    pca_tsne()