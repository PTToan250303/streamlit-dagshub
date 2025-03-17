import streamlit as st
import numpy as np
import openml
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import logging
from datetime import datetime

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hàm cấu hình MLflow của bạn
def input_mlflow():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/PTToan250303/Linear_replication.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)

    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "PTToan250303"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "5ca8caf353d564c358852da97c7487e64fc30a73"
    mlflow.set_experiment("Classification")
    logger.info(f"Đã cấu hình MLflow với URI: {DAGSHUB_MLFLOW_URI}")

# Hàm tải dữ liệu MNIST từ OpenML
def load_mnist_data():
    try:
        logger.info("Đang tải dữ liệu MNIST từ OpenML...")
        dataset = openml.datasets.get_dataset(554)  # ID của tập MNIST
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute
        )
        logger.info("Đã tải dữ liệu MNIST thành công!")
        return X, y
    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu: {e}")
        st.error(f"Lỗi khi tải dữ liệu: {e}")
        return None, None

# Hàm áp dụng PCA
def apply_pca(X, n_components=2):
    logger.info(f"Áp dụng PCA với {n_components} thành phần...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    logger.info("PCA đã hoàn thành!")
    return X_pca, pca

# Hàm áp dụng t-SNE
def apply_tsne(X, n_components=2, perplexity=30, n_iter=1000):
    logger.info(f"Áp dụng t-SNE với {n_components} thành phần, perplexity={perplexity}, n_iter={n_iter}...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)
    X_tsne = tsne.fit_transform(X)
    logger.info("t-SNE đã hoàn thành!")
    return X_tsne, tsne

# Hàm vẽ và hiển thị kết quả
def plot_and_display(X_reduced, y, title, method):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y, palette="deep", s=50)
    plt.title(f"{title} - {method}")
    plt.xlabel(f"Thành phần 1")
    plt.ylabel(f"Thành phần 2")
    st.pyplot(plt)

# Hàm chính của ứng dụng Streamlit
def main():
    st.title("Thu gọn chiều trên tập dữ liệu MNIST với PCA và t-SNE")
    st.write("Ứng dụng sử dụng Streamlit và MLflow để phân tích dữ liệu MNIST từ OpenML.")

    # Khởi tạo MLflow
    if 'mlflow_url' not in st.session_state:
        input_mlflow()

    # Tải dữ liệu
    if 'X' not in st.session_state or 'y' not in st.session_state:
        X, y = load_mnist_data()
        if X is not None and y is not None:
            st.session_state['X'] = X
            st.session_state['y'] = y
        else:
            st.stop()

    X = st.session_state['X']
    y = st.session_state['y']

    # Lựa chọn phương pháp thu gọn chiều
    method = st.selectbox("Chọn phương pháp thu gọn chiều:", ["PCA", "t-SNE"])

    if method == "PCA":
        n_components = st.slider("Số thành phần chính:", 2, 10, 2)
        if st.button("Thực hiện PCA"):
            with mlflow.start_run(run_name=f"PCA_n={n_components}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                X_pca, pca = apply_pca(X, n_components)
                mlflow.log_param("n_components", n_components)
                mlflow.log_metric("explained_variance_ratio", sum(pca.explained_variance_ratio_))
                mlflow.sklearn.log_model(pca, "pca_model")
                logger.info(f"Đã log PCA model với n_components={n_components}")
                plot_and_display(X_pca, y, "Kết quả PCA", "PCA")
                st.write(f"Tỷ lệ phương sai được giải thích: {sum(pca.explained_variance_ratio_):.4f}")

    elif method == "t-SNE":
        n_components = 2  # t-SNE thường dùng 2 hoặc 3 thành phần
        perplexity = st.slider("Perplexity:", 5, 50, 30)
        n_iter = st.slider("Số lần lặp (n_iter):", 250, 2000, 1000)
        if st.button("Thực hiện t-SNE"):
            with mlflow.start_run(run_name=f"tSNE_p={perplexity}_i={n_iter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                X_tsne, tsne = apply_tsne(X, n_components, perplexity, n_iter)
                mlflow.log_param("perplexity", perplexity)
                mlflow.log_param("n_iter", n_iter)
                mlflow.sklearn.log_model(tsne, "tsne_model")
                logger.info(f"Đã log t-SNE model với perplexity={perplexity}, n_iter={n_iter}")
                plot_and_display(X_tsne, y, "Kết quả t-SNE", "t-SNE")

if __name__ == "__main__":
    main()