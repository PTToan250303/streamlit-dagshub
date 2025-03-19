import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import os
# Tải dữ liệu MNIST từ OpenM
from sklearn.metrics import pairwise_distances_argmin
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import base64
from matplotlib.lines import Line2D
from sklearn.datasets import make_moons
import io

if "mlflow_url" not in st.session_state:
    st.session_state["mlflow_url"] = "http://127.0.0.1:5000"  

# Hàm chuẩn hóa dữ liệu
@st.cache_data
def standardize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Hàm K-means
def kmeans_init_centers(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]

def kmeans_assign_labels(X, centers):
    D = pairwise_distances_argmin(X, centers)
    return D

def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k, :]
        if len(Xk) > 0:
            centers[k, :] = np.mean(Xk, axis=0)
    return centers

def has_converged(centers, new_centers):
    return set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers])

@st.cache_data
def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers) or it >= 100:
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)

# Hàm hiển thị dữ liệu K-means
def kmeans_display(X, label, centers=None, title="Kết quả phân cụm K-means"):
    K = np.max(label) + 1 if len(np.unique(label)) > 0 else 2
    colors = ['red' if l == 0 else 'blue' for l in label]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=80, edgecolors='k')
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='green', marker='x', s=200, label='Centers')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Cluster 0'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Cluster 1')
    ]
    if centers is not None:
        legend_elements.append(Line2D([0], [0], marker='x', color='w', markerfacecolor='green', markersize=10, label='Centers'))
    plt.legend(handles=legend_elements, title="Nhãn", loc='upper right')
    plt.axis('equal')

# Hàm hiển thị dữ liệu DBSCAN (hiển thị 2 cụm)
def dbscan_display(X, labels, title="Kết quả phân cụm DBSCAN"):
    cluster_labels = labels[labels != -1]  # Chỉ lấy các nhãn cụm
    cluster_points = X[labels != -1]  # Chỉ lấy các điểm thuộc cụm
    noise_points = X[labels == -1]  # Điểm nhiễu

    # Gán màu cho các cụm
    colors = ['red' if l == 0 else 'blue' for l in cluster_labels]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors, s=80, edgecolors='k', label='Clusters')
    plt.scatter(noise_points[:, 0], noise_points[:, 1], c='black', marker='x', s=100, label='Noise')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Cluster 0'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Cluster 1'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=10, label='Noise')
    ]
    plt.legend(handles=legend_elements, title="Phân loại", loc='upper right')
    plt.axis('equal')

# Hàm dự đoán cụm cho điểm mới (DBSCAN)
def predict_new_point_dbscan(X_train, labels, new_point, eps):
    distances = np.sqrt(np.sum((X_train - new_point) ** 2, axis=1))
    nearest_core = np.where(distances <= eps)[0]
    if len(nearest_core) > 0:
        nearest_label = labels[nearest_core[0]]
        return nearest_label if nearest_label != -1 else "Nhiễu"
    return "Nhiễu"

# Hàm tạo animation DBSCAN (tối ưu)
@st.cache_data
def create_dbscan_animation(X, labels_dbscan, core_sample_indices, eps, min_samples):
    fig_dbscan, ax_dbscan = plt.subplots()
    scat_dbscan = ax_dbscan.scatter(X[:, 0], X[:, 1], c='black', s=80, edgecolors='k')  # Bắt đầu với màu đen
    ax_dbscan.set_xlabel('X1')
    ax_dbscan.set_ylabel('X2')
    ax_dbscan.set_title('Quá trình phân cụm của DBSCAN')
    legend_elements_dbscan = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Cluster 0'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Cluster 1'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=10, label='Noise')
    ]
    ax_dbscan.legend(handles=legend_elements_dbscan, title="Phân loại", loc='upper right')

    # Tăng bước nhảy để giảm số khung hình
    step = max(1, len(core_sample_indices) // 20)  # Tối đa 20 khung hình
    frames = list(range(0, len(core_sample_indices), step)) + [len(core_sample_indices)]  # Thêm khung cuối

    def init_dbscan():
        ax_dbscan.clear()
        ax_dbscan.scatter(X[:, 0], X[:, 1], c='black', s=80, edgecolors='k')
        ax_dbscan.set_xlabel('X1')
        ax_dbscan.set_ylabel('X2')
        ax_dbscan.set_title('Quá trình phân cụm của DBSCAN')
        ax_dbscan.legend(handles=legend_elements_dbscan, title="Phân loại", loc='upper right')
        return ax_dbscan,

    def update_dbscan(frame):
        ax_dbscan.clear()
        if frame < len(core_sample_indices):
            core_mask = np.zeros(len(X), dtype=bool)
            core_mask[core_sample_indices[:frame + 1]] = True
            cluster_labels = labels_dbscan[core_mask]
            cluster_points = X[core_mask]
            noise_points = X[labels_dbscan == -1]
            remaining_points = X[~core_mask]

            ax_dbscan.scatter(remaining_points[:, 0], remaining_points[:, 1], c='black', s=80, edgecolors='k')
            colors = ['red' if l == 0 else 'blue' for l in cluster_labels]
            ax_dbscan.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors, s=80, edgecolors='k', label='Clusters')
            ax_dbscan.scatter(noise_points[:, 0], noise_points[:, 1], c='black', marker='x', s=100, label='Noise')
        else:
            dbscan_display(X, labels_dbscan, 'Kết quả cuối cùng (DBSCAN)')
        ax_dbscan.set_xlabel('X1')
        ax_dbscan.set_ylabel('X2')
        ax_dbscan.legend(handles=legend_elements_dbscan, title="Phân loại", loc='upper right')
        return ax_dbscan,

    ani_dbscan = animation.FuncAnimation(fig_dbscan, update_dbscan, init_func=init_dbscan, frames=frames, interval=500, repeat=False, blit=False)

    # Lưu animation với độ phân giải thấp hơn
    try:
        writer = PillowWriter(fps=2)
        with writer.saving(fig_dbscan, "dbscan_animation.gif", dpi=80):  # Giảm dpi
            for i in frames:
                update_dbscan(i)
                writer.grab_frame()
        with open("dbscan_animation.gif", "rb") as file:
            gif_data = file.read()
        return base64.b64encode(gif_data).decode('utf-8')
    except Exception as e:
        st.error(f"Lỗi khi lưu GIF DBSCAN: {e}")
        return None

# Hàm tạo animation K-means (tối ưu)
@st.cache_data
def create_kmeans_animation(X, centers_kmeans, labels_kmeans, iterations_kmeans):
    fig_kmeans, ax_kmeans = plt.subplots()
    colors_kmeans = ['red' if l == 0 else 'blue' for l in labels_kmeans[0]]
    scat_kmeans = ax_kmeans.scatter(X[:, 0], X[:, 1], c=colors_kmeans, s=80, edgecolors='k')
    ax_kmeans.set_xlabel('X1')
    ax_kmeans.set_ylabel('X2')
    ax_kmeans.set_title('Quá trình phân cụm của K-means')
    legend_elements_kmeans = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Cluster 0'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Cluster 1')
    ]
    ax_kmeans.legend(handles=legend_elements_kmeans, title="Nhãn", loc='upper right')

    def init_kmeans():
        ax_kmeans.clear()
        ax_kmeans.scatter(X[:, 0], X[:, 1], c=colors_kmeans, s=80, edgecolors='k')
        ax_kmeans.set_xlabel('X1')
        ax_kmeans.set_ylabel('X2')
        ax_kmeans.set_title('Quá trình phân cụm của K-means')
        ax_kmeans.legend(handles=legend_elements_kmeans, title="Nhãn", loc='upper right')
        return ax_kmeans,

    def update_kmeans(frame):
        ax_kmeans.clear()
        colors = ['red' if l == 0 else 'blue' for l in labels_kmeans[frame]]
        ax_kmeans.scatter(X[:, 0], X[:, 1], c=colors, s=80, edgecolors='k')
        ax_kmeans.scatter(centers_kmeans[frame][:, 0], centers_kmeans[frame][:, 1], c='green', marker='x', s=200, label='Centers')
        ax_kmeans.set_xlabel('X1')
        ax_kmeans.set_ylabel('X2')
        ax_kmeans.set_title(f'Bước {frame + 1} (K-means)')
        ax_kmeans.legend(handles=legend_elements_kmeans + [Line2D([0], [0], marker='x', color='w', markerfacecolor='green', markersize=10, label='Centers')], title="Nhãn", loc='upper right')
        return ax_kmeans,

    frames_kmeans = min(iterations_kmeans + 1, 10)
    ani_kmeans = animation.FuncAnimation(fig_kmeans, update_kmeans, init_func=init_kmeans, frames=frames_kmeans, interval=2000, repeat=False, blit=False)

    try:
        writer = PillowWriter(fps=0.5)
        with writer.saving(fig_kmeans, "kmeans_animation.gif", dpi=80):  # Giảm dpi
            for i in range(frames_kmeans):
                update_kmeans(i)
                writer.grab_frame()
        with open("kmeans_animation.gif", "rb") as file:
            gif_data = file.read()
        return base64.b64encode(gif_data).decode('utf-8')
    except Exception as e:
        st.error(f"Lỗi khi lưu GIF K-means: {e}")
        return None

# Hàm tổng với nội dung
def ly_thuyet_dbscan():
    # Tiêu đề chính
    st.markdown('<h1 style="color:#FF4500; text-align:center;">🌟 DBSCAN Clustering 🌟</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:24px; color:#4682B4;">📝 Tìm hiểu thuật toán DBSCAN để phân cụm dữ liệu dựa trên mật độ.</p>', unsafe_allow_html=True)

    # Chọn nguồn dữ liệu
    st.markdown('<h2 style="font-size:28px; color:#32CD32;">📊 Chọn nguồn dữ liệu</h2>', unsafe_allow_html=True)
    data_option = st.radio("Chọn loại dữ liệu:", ("Dữ liệu giả lập", "Dữ liệu tùy chỉnh"), key="dbscan_data_option")

    if data_option == "Dữ liệu giả lập":
        X, _ = make_moons(n_samples=300, noise=0.1)
        X = standardize_data(X)
        original_label = np.zeros(len(X))
    else:
        st.markdown('<p style="font-size:20px;">Thêm các cặp giá trị X1, X2 (nhãn sẽ được phân cụm bởi DBSCAN):</p>', unsafe_allow_html=True)
        if 'custom_data' not in st.session_state:
            st.session_state.custom_data = {'X1': [], 'X2': []}

        col1, col2 = st.columns(2)
        with col1:
            x1_input = st.number_input("Giá trị X1:", value=0.0, step=0.1, key="x1_input")
        with col2:
            x2_input = st.number_input("Giá trị X2:", value=0.0, step=0.1, key="x2_input")

        if st.button("➕ Thêm điểm"):
            st.session_state.custom_data['X1'].append(x1_input)
            st.session_state.custom_data['X2'].append(x2_input)
            st.rerun()

        if st.session_state.custom_data['X1']:
            df = pd.DataFrame(st.session_state.custom_data)
            st.markdown('<p style="font-size:18px;">Dữ liệu đã nhập:</p>', unsafe_allow_html=True)
            st.dataframe(df)

            delete_index = st.selectbox("Chọn điểm để xóa (nếu cần):", options=range(len(st.session_state.custom_data['X1'])), format_func=lambda i: f"Điểm {i}: X1={st.session_state.custom_data['X1'][i]}, X2={st.session_state.custom_data['X2'][i]}")
            if st.button("🗑️ Xóa điểm"):
                st.session_state.custom_data['X1'].pop(delete_index)
                st.session_state.custom_data['X2'].pop(delete_index)
                st.rerun()

            X = np.array([st.session_state.custom_data['X1'], st.session_state.custom_data['X2']]).T
            X = standardize_data(X)
            original_label = np.zeros(len(X))
            if len(X) < 2:
                st.error("Vui lòng nhập ít nhất 2 cặp dữ liệu để phân cụm!")
                return
        else:
            st.warning("Chưa có dữ liệu nào được thêm. Hãy nhập ít nhất 2 cặp X1, X2 để tiếp tục!")
            return

    # Phần 1: Lý thuyết với ví dụ đơn giản
    st.markdown('<h2 style="font-size:32px; color:#32CD32;">📚 1. DBSCAN là gì và cách hoạt động?</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:22px;">
    ❓ DBSCAN là thuật toán phân cụm dựa trên mật độ, nhóm các điểm gần nhau và xác định nhiễu.<br>
    🚀 <b>Các bước chính:</b><br>
    - Xác định điểm lõi (core points) dựa trên `eps` và `min_samples`.<br>
    - Mở rộng cụm từ điểm lõi đến điểm biên (border points).<br>
    - Đánh dấu các điểm không thuộc cụm nào là nhiễu (noise).<br>
    📐 <b>Tham số:</b> `eps` (khoảng cách tối đa), `min_samples` (số điểm tối thiểu).<br>
    </p>
    """, unsafe_allow_html=True)

    # Ví dụ đơn giản với 2 vòng lặp
    st.markdown('<h3 style="font-size:26px; color:#4682B4;">📋 Ví dụ: Phân cụm 5 điểm với DBSCAN</h3>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:20px;">
    Giả sử ta có 5 điểm 2D: <br>
    - A = (1, 1), B = (1.5, 1.2), C = (1.8, 1.5), D = (4, 4), E = (4.5, 4.2)<br>
    Tham số: `eps = 0.5`, `min_samples = 2`.<br>
    </p>
    """, unsafe_allow_html=True)

    # Vòng lặp 1
    st.markdown('<h4 style="font-size:24px; color:#FF4500;">🔄 Vòng lặp 1:</h4>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:20px;">
    - <b>Khởi tạo:</b> Chọn điểm A=(1, 1).<br>
    - <b>Kiểm tra lân cận:</b> Tính khoảng cách Euclidean:<br>
      - A đến B: √((1.5-1)² + (1.2-1)²) ≈ 0.32 < 0.5 → B là lân cận.<br>
      - A đến C: √((1.8-1)² + (1.5-1)²) ≈ 0.9 > 0.5 → C không lân cận.<br>
      - A đến D: √((4-1)² + (4-1)²) ≈ 4.24 > 0.5 → D không lân cận.<br>
      - A đến E: √((4.5-1)² + (4.2-1)²) ≈ 4.8 > 0.5 → E không lân cận.<br>
    - <b>Điểm lõi:</b> A có 1 điểm lân cận (B), nhỏ hơn `min_samples=2`, nên A chưa phải điểm lõi.<br>
    - <b>Tiếp tục:</b> Chọn B=(1.5, 1.2), lân cận: A (0.32), C (0.33), D (3.3), E (3.6).<br>
      B có 2 điểm lân cận (A, C) ≥ `min_samples=2`, nên B là điểm lõi, tạo cụm 1.<br>
    - <b>Mở rộng:</b> Thêm A và C vào cụm 1 (do gần B).<br>
    - <b>Kết quả vòng 1:</b> Cụm 1: {A, B, C}, D và E là nhiễu tạm thời.<br>
    </p>
    """, unsafe_allow_html=True)

    # Vòng lặp 2
    st.markdown('<h4 style="font-size:24px; color:#FF4500;">🔄 Vòng lặp 2:</h4>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:20px;">
    - <b>Khởi tạo:</b> Chọn D=(4, 4), lân cận: E (0.5) < 0.5.<br>
      D có 1 điểm lân cận (E) ≥ `min_samples=2`, nên D là điểm lõi, tạo cụm 2.<br>
    - <b>Mở rộng:</b> Thêm E vào cụm 2.<br>
    - <b>Kết quả cuối:</b> Cụm 1: {A, B, C}, Cụm 2: {D, E}, không còn điểm chưa thăm.<br>
    </p>
    """, unsafe_allow_html=True)

    # Hiển thị biểu đồ ví dụ
    example_data = np.array([[1, 1], [1.5, 1.2], [1.8, 1.5], [4, 4], [4.5, 4.2]])
    example_data = standardize_data(example_data)
    dbscan_example = DBSCAN(eps=0.5, min_samples=2).fit(example_data)
    fig, ax = plt.subplots()
    dbscan_display(example_data, dbscan_example.labels_, "Ví dụ phân cụm 5 điểm (eps=0.5, min_samples=2)")
    st.pyplot(fig)

    st.markdown("""
    <p style="font-size:20px;">
    💡 <b>Lưu ý:</b> Bạn có thể tự tính khoảng cách Euclidean để kiểm tra!<br>
    </p>
    """, unsafe_allow_html=True)

    # Phần 1.5: Hình động minh họa DBSCAN
    st.markdown('<h2 style="font-size:32px; color:#FFD700;">🎥 1.5. Quá trình phân cụm với hình động (DBSCAN)</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">📈 Xem quá trình DBSCAN phân cụm dữ liệu:</p>', unsafe_allow_html=True)

    # Chạy DBSCAN để lấy nhãn
    eps = st.slider("Chọn eps:", 0.1, 1.0, 0.3, 0.1)
    min_samples = st.slider("Chọn min_samples:", 2, 10, 5, 1)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels_dbscan = dbscan.labels_

    # Tạo animation DBSCAN (đã tối ưu)
    gif_base64_dbscan = create_dbscan_animation(X, labels_dbscan, dbscan.core_sample_indices_, eps, min_samples)
    if gif_base64_dbscan:
        st.markdown(f'<img src="data:image/gif;base64,{gif_base64_dbscan}" alt="animation_dbscan">', unsafe_allow_html=True)

    # Phần 2: Trực quan hóa kết quả cuối cùng (DBSCAN)
    st.markdown('<h2 style="font-size:32px; color:#FFD700;">📈 2. Kết quả phân cụm (DBSCAN)</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">👀 Xem kết quả phân cụm sau khi DBSCAN hoàn tất:</p>', unsafe_allow_html=True)

    fig_dbscan_result, ax_dbscan_result = plt.subplots()
    dbscan_display(X, labels_dbscan)
    n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    st.markdown(f'<p style="font-size:20px; color:#4682B4;">📊 Số cụm: {n_clusters_dbscan}, Số nhiễu: {np.sum(labels_dbscan == -1)}</p>', unsafe_allow_html=True)
    st.pyplot(fig_dbscan_result)

    # Nút bấm để so sánh với K-means (chỉ hình động)
    st.markdown('<h2 style="font-size:32px; color:#FFD700;">🔄 So sánh với K-means (Chỉ Hình Động)</h2>', unsafe_allow_html=True)
    if st.button("So sánh với K-means"):
        st.markdown('<p style="font-size:22px;">📊 So sánh quá trình phân cụm của DBSCAN và K-means qua hình động:</p>', unsafe_allow_html=True)

        # Chạy K-means để lấy dữ liệu cho hình động
        K = 2
        with st.spinner("🔄 Đang tính toán phân cụm K-means..."):
            centers_kmeans, labels_kmeans, iterations_kmeans = kmeans(X, K)

        # Tạo animation K-means (đã tối ưu)
        gif_base64_kmeans = create_kmeans_animation(X, centers_kmeans, labels_kmeans, iterations_kmeans)
        if gif_base64_kmeans:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<p style="font-size:20px; text-align:center;">📊 DBSCAN Animation</p>', unsafe_allow_html=True)
                st.markdown(f'<img src="data:image/gif;base64,{gif_base64_dbscan}" alt="animation_dbscan">', unsafe_allow_html=True)
            with col2:
                st.markdown('<p style="font-size:20px; text-align:center;">📊 K-means Animation (K=2)</p>', unsafe_allow_html=True)
                st.markdown(f'<img src="data:image/gif;base64,{gif_base64_kmeans}" alt="animation_kmeans">', unsafe_allow_html=True)

    # Phần 3: Tùy chỉnh và dự đoán (chỉ DBSCAN)
    st.markdown('<h2 style="font-size:32px; color:#00CED1;">🎮 3. Thử nghiệm với điểm mới</h2>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:20px;">Nhập giá trị X1 ({min(X[:, 0]):.2f} đến {max(X[:, 0]):.2f}) và X2 ({min(X[:, 1]):.2f} đến {max(X[:, 1]):.2f}):</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        input_x1 = st.number_input("X1:", min_value=float(min(X[:, 0])), max_value=float(max(X[:, 0])), value=float(np.mean(X[:, 0])), step=0.1)
    with col2:
        input_x2 = st.number_input("X2:", min_value=float(min(X[:, 1])), max_value=float(max(X[:, 1])), value=float(np.mean(X[:, 1])), step=0.1)

    X_new = np.array([[input_x1, input_x2]])
    X_new = standardize_data(X_new)
    predicted_label_dbscan = predict_new_point_dbscan(X, labels_dbscan, X_new[0], eps)

    fig_dbscan_predict, ax_dbscan_predict = plt.subplots()
    dbscan_display(X, labels_dbscan)
    ax_dbscan_predict.scatter([input_x1], [input_x2], c='green', marker='*', s=150, label='Điểm mới')
    ax_dbscan_predict.set_title('Dự đoán cụm cho điểm mới (DBSCAN)')
    ax_dbscan_predict.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Cluster 0'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Cluster 1'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=10, label='Noise'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=10, label='Điểm mới')
    ], title="Phân loại", loc='upper right')
    st.pyplot(fig_dbscan_predict)

    st.markdown(f'<p style="font-size:20px;">📈 <b>Dự đoán (DBSCAN):</b> Điểm (X1={input_x1:.2f}, X2={input_x2:.2f}) thuộc {"Cụm" if predicted_label_dbscan != "Nhiễu" else ""} {predicted_label_dbscan if predicted_label_dbscan != "Nhiễu" else ""}</p>', unsafe_allow_html=True)

    # Phần 4: Ưu điểm và hạn chế (chỉ DBSCAN)
    st.markdown('<h2 style="font-size:32px; color:#FFA500;">⚠️ 4. Ưu điểm và hạn chế của DBSCAN</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:22px;">
    👍 <b>Ưu điểm:</b><br>
    - 🌟 Phát hiện cụm bất kỳ hình dạng.<br>
    - 🚫 Không cần chỉ định số cụm trước.<br>
    - 🔍 Xử lý nhiễu và ngoại lai tốt.<br>
    👎 <b>Hạn chế:</b><br>
    - 🚨 Nhạy cảm với `eps` và `min_samples`.<br>
    - ⚙️ Khó với cụm mật độ khác nhau.<br>
    💡 <b>Gợi ý:</b> Sử dụng đồ thị khoảng cách k-th gần nhất để chọn `eps`.<br>
    </p>
    """, unsafe_allow_html=True)

    # Phần 5: Tài liệu tham khảo
    st.markdown('<h2 style="font-size:32px; color:#1E90FF;">🔗 5. Tài liệu tham khảo</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">📖 Xem chi tiết về DBSCAN tại <a href="https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/?form=MG0AV3">Analytics Vidhya</a>.</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">🙏 Cảm ơn bạn đã khám phá DBSCAN!</p>', unsafe_allow_html=True)

# Hàm tính toán cho K-means (dựa trên bài viết)
def kmeans_init_centers(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]

def kmeans_assign_labels(X, centers):
    D = pairwise_distances_argmin(X, centers)
    return D

def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k, :]
        if len(Xk) > 0:  # Kiểm tra để tránh lỗi khi tập con rỗng
            centers[k, :] = np.mean(Xk, axis=0)
    return centers

def has_converged(centers, new_centers):
    return set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers])

def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers) or it >= 100:  # Giới hạn vòng lặp
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)

# Hàm hiển thị dữ liệu
def kmeans_display(X, label, centers=None):
    K = np.max(label) + 1 if len(np.unique(label)) > 0 else 2
    colors = ['red' if l == 0 else 'blue' for l in label]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=80, edgecolors='k')
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='green', marker='x', s=200, label='Centers')
    plt.xlabel('X1')
    plt.ylabel('X2')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Cluster 0'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Cluster 1')
    ]
    if centers is not None:
        legend_elements.append(Line2D([0], [0], marker='x', color='w', markerfacecolor='green', markersize=10, label='Centers'))
    plt.legend(handles=legend_elements, title="Nhãn", loc='upper right')
    plt.axis('equal')

# Hàm tổng với nội dung
def ly_thuyet_kmeans():
    # Tiêu đề chính
    st.markdown('<h1 style="color:#FF4500; text-align:center;">🌟 K-means Clustering 🌟</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:24px; color:#4682B4;">📝 Tìm hiểu cách sử dụng K-means để phân cụm dữ liệu một cách trực quan và hiệu quả.</p>', unsafe_allow_html=True)

    # Chọn nguồn dữ liệu
    st.markdown('<h2 style="font-size:28px; color:#32CD32;">📊 Chọn nguồn dữ liệu</h2>', unsafe_allow_html=True)
    data_option = st.radio("Chọn loại dữ liệu:", ("Dữ liệu giả lập", "Dữ liệu tùy chỉnh"), key="dbscan_data_option_1")

    if data_option == "Dữ liệu giả lập":
        # Dữ liệu mẫu (phân bố tương tự ví dụ trước)
        np.random.seed(42)
        v1_class0 = np.random.normal(1.5, 0.2, 20)  # Cluster 0: X1 quanh 1.5
        v2_class0 = np.random.normal(2.4, 0.2, 20)  # Cluster 0: X2 quanh 2.4
        v1_class1 = np.random.normal(1.6, 0.2, 20)  # Cluster 1: X1 quanh 1.6
        v2_class1 = np.random.normal(2.6, 0.2, 20)  # Cluster 1: X2 quanh 2.6
        v1 = np.concatenate([v1_class0, v1_class1])
        v2 = np.concatenate([v2_class0, v2_class1])
        X = np.array([[x, y] for x, y in zip(v1, v2)])
        original_label = np.array([0] * 20 + [1] * 20)
    else:
        # Dữ liệu tùy chỉnh
        st.markdown('<p style="font-size:20px;">Thêm các cặp giá trị X1, X2 (nhãn sẽ được phân cụm bởi K-means):</p>', unsafe_allow_html=True)
        if 'custom_data' not in st.session_state:
            st.session_state.custom_data = {'X1': [], 'X2': []}

        col1, col2 = st.columns(2)
        with col1:
            x1_input = st.number_input("Giá trị X1:", value=1.2, step=0.1, key="x1_input")
        with col2:
            x2_input = st.number_input("Giá trị X2:", value=2.3, step=0.1, key="x2_input")

        if st.button("➕ Thêm điểm"):
            st.session_state.custom_data['X1'].append(x1_input)
            st.session_state.custom_data['X2'].append(x2_input)
            st.rerun()

        if st.session_state.custom_data['X1']:
            df = pd.DataFrame(st.session_state.custom_data)
            st.markdown('<p style="font-size:18px;">Dữ liệu đã nhập:</p>', unsafe_allow_html=True)
            st.dataframe(df)

            delete_index = st.selectbox("Chọn điểm để xóa (nếu cần):", options=range(len(st.session_state.custom_data['X1'])), format_func=lambda i: f"Điểm {i}: X1={st.session_state.custom_data['X1'][i]}, X2={st.session_state.custom_data['X2'][i]}")
            if st.button("🗑️ Xóa điểm"):
                st.session_state.custom_data['X1'].pop(delete_index)
                st.session_state.custom_data['X2'].pop(delete_index)
                st.rerun()

            X = np.array([st.session_state.custom_data['X1'], st.session_state.custom_data['X2']]).T
            original_label = np.zeros(len(X))  # Nhãn ban đầu không quan trọng
            if len(X) < 2:
                st.error("Vui lòng nhập ít nhất 2 cặp dữ liệu để phân cụm!")
                return
        else:
            st.warning("Chưa có dữ liệu nào được thêm. Hãy nhập ít nhất 2 cặp X1, X2 để tiếp tục!")
            return

    # Phần 1: Lý thuyết với ví dụ đơn giản
    st.markdown('<h2 style="font-size:32px; color:#32CD32;">📚 1. K-means là gì và cách hoạt động?</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:22px;">
    ❓ K-means là một thuật toán unsupervised learning dùng để phân cụm dữ liệu thành K nhóm dựa trên sự tương đồng.<br>
    🚀 <b>Các bước chính:</b><br>
    - Chọn ngẫu nhiên K tâm cụm (centers) ban đầu.<br>
    - Gán mỗi điểm dữ liệu vào cụm gần nhất dựa trên khoảng cách Euclidean.<br>
    - Cập nhật tâm cụm bằng trung bình của các điểm trong cụm.<br>
    - Lặp lại cho đến khi tâm cụm không đổi.<br>
    📐 <b>Thuật toán:</b> Tối ưu hóa bằng cách giảm tổng khoảng cách từ các điểm đến tâm cụm của chúng.<br>
    </p>
    """, unsafe_allow_html=True)

    # Ví dụ đơn giản với 2 vòng lặp
    st.markdown('<h3 style="font-size:26px; color:#4682B4;">📋 Ví dụ: Phân cụm 4 điểm với K=2</h3>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:22px;">
    Giả sử ta có 4 điểm 2D: <br>
    - A = (1, 1)<br>
    - B = (2, 1)<br>
    - C = (4, 3)<br>
    - D = (5, 4)<br>
    Mục tiêu: Phân thành 2 cụm (K=2) qua 2 vòng lặp.<br>
    </p>
    """, unsafe_allow_html=True)

    # Vòng lặp 1
    st.markdown('<h4 style="font-size:24px; color:#FF4500;">🔄 Vòng lặp 1:</h4>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:20px;">
    - <b>Khởi tạo tâm cụm:</b> Chọn ngẫu nhiên A=(1, 1) cho Cluster 0, C=(4, 3) cho Cluster 1.<br>
    - <b>Gán nhãn:</b> Tính khoảng cách Euclidean từ mỗi điểm đến tâm cụm:<br>
      - A đến (1, 1): 0 → Cluster 0<br>
      - B đến (1, 1): √((2-1)² + (1-1)²) = 1 → Cluster 0<br>
      - C đến (4, 3): 0 → Cluster 1<br>
      - D đến (4, 3): √((5-4)² + (4-3)²) = √2 ≈ 1.41 → Cluster 1<br>
    - <b>Cập nhật tâm cụm:</b> Trung bình của các điểm trong cụm:<br>
      - Cluster 0: (A, B) → ((1+2)/2, (1+1)/2) = (1.5, 1)<br>
      - Cluster 1: (C, D) → ((4+5)/2, (3+4)/2) = (4.5, 3.5)<br>
    </p>
    """, unsafe_allow_html=True)

    # Vòng lặp 2
    st.markdown('<h4 style="font-size:24px; color:#FF4500;">🔄 Vòng lặp 2:</h4>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:20px;">
    - <b>Gán nhãn:</b> Tính lại khoảng cách:<br>
      - A đến (1.5, 1): √((1-1.5)² + (1-1)²) = 0.5 → Cluster 0<br>
      - B đến (1.5, 1): √((2-1.5)² + (1-1)²) = 0.5 → Cluster 0<br>
      - C đến (4.5, 3.5): √((4-4.5)² + (3-3.5)²) = 0.707 → Cluster 1<br>
      - D đến (4.5, 3.5): √((5-4.5)² + (4-3.5)²) = 0.707 → Cluster 1<br>
    - <b>Cập nhật tâm cụm:</b> Trung bình:<br>
      - Cluster 0: (A, B) → ((1+2)/2, (1+1)/2) = (1.5, 1)<br>
      - Cluster 1: (C, D) → ((4+5)/2, (3+4)/2) = (4.5, 3.5)<br>
    - <b>Kết luận:</b> Tâm cụm không đổi, thuật toán hội tụ sau 2 vòng lặp.<br>
    </p>
    """, unsafe_allow_html=True)

    # Hiển thị biểu đồ ví dụ
    example_data = np.array([[1, 1], [2, 1], [4, 3], [5, 4]])
    example_labels = np.array([0, 0, 1, 1])  # Kết quả sau 2 vòng lặp
    example_centers = np.array([[1.5, 1], [4.5, 3.5]])
    fig, ax = plt.subplots()
    kmeans_display(example_data, example_labels, example_centers)
    ax.set_title('Ví dụ phân cụm 4 điểm (K=2)')
    st.pyplot(fig)

    st.markdown("""
    <p style="font-size:20px;">
    💡 <b>Lưu ý:</b> Bạn có thể tự tính toán các khoảng cách và trung bình để kiểm tra!<br>
    </p>
    """, unsafe_allow_html=True)

    # Phần 1.5: Hình động minh họa
    st.markdown('<h2 style="font-size:32px; color:#FFD700;">🎥 1.5. Quá trình phân cụm với hình động</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">📈 Xem quá trình K-means phân cụm dữ liệu qua các bước:</p>', unsafe_allow_html=True)

    # Tạo hình động
    K = 2  # Số cụm cố định là 2
    fig, ax = plt.subplots()
    colors = ['red' if l == 0 else 'blue' for l in original_label]
    scat = ax.scatter(X[:, 0], X[:, 1], c=colors, s=80, edgecolors='k')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Quá trình phân cụm của K-means')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Cluster 0'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Cluster 1')
    ]
    ax.legend(handles=legend_elements, title="Nhãn", loc='upper right')

    def init():
        ax.clear()
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=80, edgecolors='k')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title('Quá trình phân cụm của K-means')
        ax.legend(handles=legend_elements, title="Nhãn", loc='upper right')
        return ax,

    def update(frame, centers_list, labels_list):
        ax.clear()
        colors = ['red' if l == 0 else 'blue' for l in labels_list[frame]]
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=80, edgecolors='k')
        ax.scatter(centers_list[frame][:, 0], centers_list[frame][:, 1], c='green', marker='x', s=200, label='Centers')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title(f'Bước {frame + 1}')
        ax.legend(handles=legend_elements + [Line2D([0], [0], marker='x', color='w', markerfacecolor='green', markersize=10, label='Centers')], title="Nhãn", loc='upper right')
        return ax,

    # Chạy K-means để lấy dữ liệu cho hình động
    with st.spinner("🔄 Đang tính toán phân cụm..."):
        centers, labels, iterations = kmeans(X, K)
        frames = min(iterations + 1, 10)  # Giới hạn số khung hình

    # Tạo animation
    ani = animation.FuncAnimation(fig, update, init_func=init, fargs=(centers, labels), frames=frames, interval=2000, repeat=False, blit=False)

    # Lưu animation thành GIF
    try:
        writer = PillowWriter(fps=0.5)
        with writer.saving(fig, "kmeans_animation.gif", dpi=100):
            for i in range(frames):
                update(i, centers, labels)
                writer.grab_frame()
    except Exception as e:
        st.error(f"Lỗi khi lưu GIF: {e}")
        return

    # Hiển thị GIF trong Streamlit
    with open("kmeans_animation.gif", "rb") as file:
        gif_data = file.read()
    gif_base64 = base64.b64encode(gif_data).decode('utf-8')
    st.markdown(f'<img src="data:image/gif;base64,{gif_base64}" alt="animation">', unsafe_allow_html=True)

    # Phần 2: Trực quan hóa kết quả cuối cùng
    st.markdown('<h2 style="font-size:32px; color:#FFD700;">📈 2. Kết quả phân cụm</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">👀 Xem kết quả phân cụm sau khi K-means hoàn tất:</p>', unsafe_allow_html=True)

    fig, ax = plt.subplots()
    final_labels = labels[-1]
    kmeans_display(X, final_labels, centers[-1])
    ax.set_title('Kết quả phân cụm cuối cùng (K=2)')
    st.pyplot(fig)

    st.markdown(f'<p style="font-size:20px; color:#4682B4;">📊 Số bước lặp: {iterations}</p>', unsafe_allow_html=True)

    # Phần 3: Tùy chỉnh và dự đoán
    st.markdown('<h2 style="font-size:32px; color:#00CED1;">🎮 3. Thử nghiệm với điểm mới</h2>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:20px;">Nhập giá trị X1 ({min(X[:, 0]):.2f} đến {max(X[:, 0]):.2f}) và X2 ({min(X[:, 1]):.2f} đến {max(X[:, 1]):.2f}):</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        input_x1 = st.number_input("X1:", min_value=float(min(X[:, 0])), max_value=float(max(X[:, 0])), value=float(np.mean(X[:, 0])), step=0.1)
    with col2:
        input_x2 = st.number_input("X2:", min_value=float(min(X[:, 1])), max_value=float(max(X[:, 1])), value=float(np.mean(X[:, 1])), step=0.1)

    X_new = np.array([[input_x1, input_x2]])
    closest_center = pairwise_distances_argmin(X_new, centers[-1])[0]
    predicted_cluster = closest_center

    fig, ax = plt.subplots()
    kmeans_display(X, final_labels, centers[-1])
    ax.scatter([input_x1], [input_x2], c='green', marker='*', s=150, label='Điểm mới')
    ax.set_title('Dự đoán cụm cho điểm mới')
    ax.legend(handles=legend_elements + [Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=10, label='Điểm mới')], title="Nhãn", loc='upper right')
    st.pyplot(fig)

    st.markdown(f'<p style="font-size:20px;">📈 <b>Dự đoán:</b> Điểm (X1={input_x1:.2f}, X2={input_x2:.2f}) thuộc Cluster {predicted_cluster} ({"Đỏ" if predicted_cluster == 0 else "Xanh"})</p>', unsafe_allow_html=True)

    # Phần 4: Hạn chế
    st.markdown('<h2 style="font-size:32px; color:#FFA500;">⚠️ 4. Hạn chế của K-means</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:22px;">
    👍 <b>Ưu điểm:</b><br>
    - 🌟 Đơn giản, nhanh chóng với dữ liệu lớn.<br>
    - 📈 Hiệu quả khi cụm có hình dạng cầu.<br>
    👎 <b>Nhược điểm:</b><br>
    - 🚨 Nhạy cảm với giá trị khởi tạo ban đầu.<br>
    - ⚙️ Không hoạt động tốt với cụm không đều hoặc không phải hình cầu.<br>
    💡 <b>Gợi ý:</b> Thử nhiều lần khởi tạo hoặc sử dụng K-means++ để cải thiện.
    </p>
    """, unsafe_allow_html=True)

    # Phần 5: Tài liệu tham khảo
    st.markdown('<h2 style="font-size:32px; color:#1E90FF;">🔗 5. Tài liệu tham khảo</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">📖 Xem chi tiết về K-means tại <a href="https://machinelearningcoban.com/2017/01/01/kmeans/?form=MG0AV3">Machine Learning Cơ Bản</a>.</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">🙏 Cảm ơn bạn đã tham gia khám phá K-means!</p>', unsafe_allow_html=True)


def data():
    # Tiêu đề chính với hiệu ứng và màu sắc bắt mắt
    st.markdown("""
        <h1 style="text-align: center; color: #1E90FF; font-size: 48px; text-shadow: 2px 2px 4px #000000;">
             Khám Phá Bộ Dữ Liệu MNIST 
        </h1>
    """, unsafe_allow_html=True)

    # Thêm CSS animation cho hiệu ứng
    st.markdown(
        """
        <style>
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Mô tả chi tiết về MNIST với bố cục đẹp
    st.markdown("""
        <div style="background-color: #F0F8FF; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h2 style="color: #32CD32; font-size: 32px;">📊 Tổng Quan Về MNIST</h2>
            <p style="font-size: 20px; color: #333; text-align: justify;">
                MNIST (Modified National Institute of Standards and Technology) là một trong những bộ dữ liệu <b>huyền thoại</b> 
                trong lĩnh vực học máy, đặc biệt nổi bật trong nhận diện mẫu và phân loại hình ảnh. Đây là "bàn đạp" đầu tiên 
                cho hàng ngàn nhà nghiên cứu và lập trình viên trên toàn cầu!<br><br>
                - 🌟 Chứa <b>70.000 ảnh chữ số viết tay</b> từ 0 đến 9, mỗi ảnh có độ phân giải <b>28x28 pixel</b>.<br>
                - 🔄 Được chia thành:<br>
                  + <b>60.000 ảnh</b> cho tập huấn luyện (training set).<br>
                  + <b>10.000 ảnh</b> cho tập kiểm tra (test set).<br>
                - 🎨 Mỗi hình ảnh là chữ số viết tay, được chuẩn hóa thành dạng <b>grayscale</b> (đen trắng), sẵn sàng cho các cuộc 
                  "thử thách" AI!
            </p>
        </div>
    """, unsafe_allow_html=True)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Lên một cấp từ buoi3

    # Trỏ đến ảnh trong `buoi2/`
    image_path = os.path.join(base_dir, "buoi2", "img3.png")
    # Hiển thị hình ảnh với hiệu ứng tương tác
    st.markdown("<h2 style='color: #FF4500; font-size: 32px;'>📸 Khám Phá Hình Ảnh Từ MNIST</h2>", unsafe_allow_html=True)
    st.image(image_path, caption="✨ Một số mẫu chữ số viết tay từ MNIST - Bạn có nhận ra chúng không?", use_container_width=True, output_format="auto")
    st.markdown("<p style='font-size: 18px; color: #6A5ACD;'>👉 Hãy thử đếm xem có bao nhiêu chữ số 7 trong ảnh trên nhé!</p>", unsafe_allow_html=True)

    # Ứng dụng thực tế với hiệu ứng thẻ
    st.markdown("""
        <h2 style="color: #9B59B6; font-size: 32px;">🌍 Ứng Dụng Thực Tế Của MNIST</h2>
        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
            <div style="background-color: #ECF0F1; padding: 15px; border-radius: 10px; flex: 1; min-width: 200px;">
                <p style="font-size: 20px; color: #2E86C1;">📋 Nhận diện số trên hóa đơn, biên lai mua sắm.</p>
            </div>
            <div style="background-color: #ECF0F1; padding: 15px; border-radius: 10px; flex: 1; min-width: 200px;">
                <p style="font-size: 20px; color: #2E86C1;">📦 Xử lý mã số trên bưu kiện tại bưu điện.</p>
            </div>
            <div style="background-color: #ECF0F1; padding: 15px; border-radius: 10px; flex: 1; min-width: 200px;">
                <p style="font-size: 20px; color: #2E86C1;">📚 Tự động hóa nhận diện tài liệu cổ.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Ví dụ mô hình với bảng tương tác
    st.markdown("<h2 style='color: #FF6347; font-size: 32px;'>🤖 Các Mô Hình Học Máy Với MNIST</h2>", unsafe_allow_html=True)
    st.write("""
        MNIST là "sân chơi" lý tưởng để thử sức với các mô hình học máy đỉnh cao. Dưới đây là những "ngôi sao" đã được thử nghiệm:
        - 🌱 **Logistic Regression**
        - 🌳 **Decision Trees**
        - 🔍 **K-Nearest Neighbors (KNN)**
        - ⚙️ **Support Vector Machines (SVM)**
        - 🧠 **Convolutional Neural Networks (CNNs)** (vua của nhận diện hình ảnh!)
    """)

def ly_thuyet_K_means():
    st.title("📌 K-Means Clustering")

    # 🔹 Giới thiệu về K-Means
    st.markdown(r"""
        ## 📌 **K-Means Clustering**
        **K-Means** là một thuật toán **phân cụm không giám sát** phổ biến, giúp chia tập dữ liệu thành **K cụm** sao cho các điểm trong cùng một cụm có đặc trưng tương đồng nhất.  

        ---

        ### 🔹 **Ý tưởng chính của K-Means**
        1️⃣ **Khởi tạo \( K \) tâm cụm (centroids)** ngẫu nhiên từ tập dữ liệu.  
        2️⃣ **Gán mỗi điểm dữ liệu vào cụm có tâm gần nhất**, sử dụng khoảng cách Euclidean:  
        """)

    st.latex(r"""
        d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}
        """)

    st.markdown(r"""
        3️⃣ **Cập nhật lại tâm cụm** bằng cách tính trung bình của các điểm trong cụm:  
        """)

    st.latex(r"""
        \mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i
        """)

    st.markdown(r"""
        4️⃣ **Lặp lại quá trình trên** cho đến khi không có sự thay đổi hoặc đạt đến số vòng lặp tối đa.  

        ---

        ### 🔢 **Công thức tối ưu hóa trong K-Means**
        K-Means tìm cách tối thiểu hóa tổng bình phương khoảng cách từ mỗi điểm đến tâm cụm của nó:  
        """)

    st.latex(r"""
        J = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2
        """)

    st.markdown(r"""
        Trong đó:  
        - **\( J \)**: Hàm mất mát (tổng bình phương khoảng cách).  
        - **\( x_i \)**: Điểm dữ liệu thứ \( i \).  
        - **\( \mu_k \)**: Tâm cụm thứ \( k \).  
        - **\( C_k \)**: Tập các điểm thuộc cụm \( k \).  

        ---

        ### ✅ **Ưu điểm & ❌ Nhược điểm**
        ✅ **Ưu điểm:**  
        - Đơn giản, dễ hiểu, tốc độ nhanh.  
        - Hiệu quả trên tập dữ liệu lớn.  
        - Dễ triển khai và mở rộng.  

        ❌ **Nhược điểm:**  
        - Cần xác định số cụm \( K \) trước.  
        - Nhạy cảm với giá trị ngoại lai (**outliers**).  
        - Kết quả phụ thuộc vào cách khởi tạo ban đầu của các tâm cụm.  

        ---

        ### 🔍 **Một số cải tiến của K-Means**
        - **K-Means++**: Cải thiện cách chọn tâm cụm ban đầu để giảm thiểu hội tụ vào cực tiểu cục bộ.  
        - **Mini-batch K-Means**: Sử dụng tập mẫu nhỏ để cập nhật tâm cụm, giúp tăng tốc độ trên dữ liệu lớn.  
        - **K-Medoids**: Thay vì trung bình, sử dụng điểm thực tế làm tâm cụm để giảm ảnh hưởng của outliers.  

        📌 **Ứng dụng của K-Means:** Phân tích khách hàng, nhận diện mẫu, nén ảnh, phân cụm văn bản, v.v.  
        """)



    # 🔹 Định nghĩa hàm tính toán
    def euclidean_distance(a, b):
        return np.linalg.norm(a - b, axis=1)

    def generate_data(n_samples, n_clusters, cluster_std):
        np.random.seed(42)
        X = []
        centers = np.random.uniform(-10, 10, size=(n_clusters, 2))
        for c in centers:
            X.append(c + np.random.randn(n_samples // n_clusters, 2) * cluster_std)
        return np.vstack(X)

    def initialize_centroids(X, k):
        return X[np.random.choice(X.shape[0], k, replace=False)]

    def assign_clusters(X, centroids):
        return np.array([np.argmin(euclidean_distance(x, centroids)) for x in X])

    def update_centroids(X, labels, k):
        return np.array([X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else np.random.uniform(-10, 10, 2) for i in range(k)])

    # Giao diện Streamlit
    st.title("🎯 Minh họa thuật toán K-Means từng bước")

    num_samples_kmeans = st.slider("Số điểm dữ liệu", 50, 500, 200, step=10)
    cluster_kmeans = st.slider("Số cụm (K)", 2, 10, 3)
    spread_kmeans = st.slider("Độ rời rạc", 0.1, 2.0, 1.0)

    # if "X" not in st.session_state:
    #     st.session_state.X = generate_data(num_samples_kmeans, cluster_kmeans, spread_kmeans)

    # X = st.session_state.X

    # Kiểm tra và cập nhật dữ liệu khi tham số thay đổi
    if "data_params" not in st.session_state or st.session_state.data_params != (num_samples_kmeans, cluster_kmeans, spread_kmeans):
        st.session_state.data_params = (num_samples_kmeans, cluster_kmeans, spread_kmeans)
        st.session_state.X = generate_data(num_samples_kmeans, cluster_kmeans, spread_kmeans)
        st.session_state.centroids = initialize_centroids(st.session_state.X, cluster_kmeans)
        st.session_state.iteration = 0
        st.session_state.labels = assign_clusters(st.session_state.X, st.session_state.centroids)

    X = st.session_state.X


    if st.button("🔄 Reset"):
        st.session_state.X = generate_data(num_samples_kmeans, cluster_kmeans, spread_kmeans)
        st.session_state.centroids = initialize_centroids(st.session_state.X, cluster_kmeans)
        st.session_state.iteration = 0
        st.session_state.labels = assign_clusters(st.session_state.X, st.session_state.centroids)

    if st.button("🔄 Cập nhật vị trí tâm cụm"):
        st.session_state.labels = assign_clusters(X, st.session_state.centroids)
        new_centroids = update_centroids(X, st.session_state.labels, cluster_kmeans)
        
        # Kiểm tra hội tụ với sai số nhỏ
        if np.allclose(new_centroids, st.session_state.centroids, atol=1e-3):
            st.warning("⚠️ Tâm cụm không thay đổi đáng kể, thuật toán đã hội tụ!")
        else:
            st.session_state.centroids = new_centroids
            st.session_state.iteration += 1

    # 🔥 Thêm thanh trạng thái hiển thị tiến trình
    
    
    
    st.status(f"Lần cập nhật: {st.session_state.iteration} - Đang phân cụm...", state="running")
    st.markdown("### 📌 Tọa độ tâm cụm hiện tại:")
    num_centroids = st.session_state.centroids.shape[0]  # Số lượng tâm cụm thực tế
    centroid_df = pd.DataFrame(st.session_state.centroids, columns=["X", "Y"])
    centroid_df.index = [f"Tâm cụm {i}" for i in range(num_centroids)]  # Đảm bảo index khớp

    st.dataframe(centroid_df)
    
    
    
    
    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = st.session_state.labels
    centroids = st.session_state.centroids

    for i in range(cluster_kmeans):
        ax.scatter(X[labels == i][:, 0], X[labels == i][:, 1], label=f"Cụm {i}", alpha=0.6, edgecolors="k")

    ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c="red", marker="X", label="Tâm cụm")
    ax.set_title(f"K-Means Clustering")
    ax.legend()

    st.pyplot(fig)







# Hàm vẽ biểu đồ
import streamlit as st
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import mode  

import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import mode

def split_data():
    st.title("📌 Chia dữ liệu Train/Test")

    # Đọc dữ liệu
    Xmt = np.load("buoi2/X.npy")
    ymt = np.load("buoi2/y.npy")
    X = Xmt.reshape(Xmt.shape[0], -1)  # Giữ nguyên định dạng dữ liệu
    y = ymt.reshape(-1)  

    total_samples = X.shape[0]

    # Thanh kéo chọn số lượng ảnh để train
    num_samples = st.slider("Chọn số lượng ảnh để train:", min_value=1000, max_value=total_samples, value=10000)

    # Thanh kéo chọn tỷ lệ Train/Test
    test_size = st.slider("Chọn tỷ lệ test:", min_value=0.1, max_value=0.5, value=0.2)

    if st.button("✅ Xác nhận & Lưu"):
        # Chọn số lượng ảnh mong muốn
        X_selected, y_selected = X[:num_samples], y[:num_samples]

        # Chia train/test theo tỷ lệ đã chọn
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=test_size, random_state=42)

        # Lưu vào session_state để sử dụng sau
        st.session_state["X_train"] = X_train
        st.session_state["y_train"] = y_train
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test

        st.success(f"🔹 Dữ liệu đã được chia: Train ({len(X_train)}), Test ({len(X_test)})")

    if "X_train" in st.session_state:
        st.write("📌 Dữ liệu train/test đã sẵn sàng để sử dụng!")


import mlflow
import os
import time
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import mlflow
import mlflow.sklearn
import streamlit as st
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import mode

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin
import pandas as pd
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import io
import base64
from matplotlib.lines import Line2D

# Hàm tính toán cho K-means (dựa trên bài viết)
def kmeans_init_centers(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]

def kmeans_assign_labels(X, centers):
    D = pairwise_distances_argmin(X, centers)
    return D

def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k, :]
        if len(Xk) > 0:  # Kiểm tra để tránh lỗi khi tập con rỗng
            centers[k, :] = np.mean(Xk, axis=0)
    return centers

def has_converged(centers, new_centers):
    return set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers])

def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers) or it >= 100:  # Giới hạn vòng lặp
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)

# Hàm hiển thị dữ liệu
def kmeans_display(X, label, centers=None):
    K = np.max(label) + 1 if len(np.unique(label)) > 0 else 2
    colors = ['red' if l == 0 else 'blue' for l in label]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=80, edgecolors='k')
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='green', marker='x', s=200, label='Centers')
    plt.xlabel('X1')
    plt.ylabel('X2')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Cluster 0'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Cluster 1')
    ]
    if centers is not None:
        legend_elements.append(Line2D([0], [0], marker='x', color='w', markerfacecolor='green', markersize=10, label='Centers'))
    plt.legend(handles=legend_elements, title="Nhãn", loc='upper right')
    plt.axis('equal')

# Hàm tổng với nội dung
def ly_thuyet_kmeans():
    # Tiêu đề chính
    st.markdown('<h1 style="color:#FF4500; text-align:center;">🌟 K-means Clustering 🌟</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:24px; color:#4682B4;">📝 Tìm hiểu cách sử dụng K-means để phân cụm dữ liệu một cách trực quan và hiệu quả.</p>', unsafe_allow_html=True)

    # Chọn nguồn dữ liệu
    st.markdown('<h2 style="font-size:28px; color:#32CD32;">📊 Chọn nguồn dữ liệu</h2>', unsafe_allow_html=True)
    data_option = st.radio("Chọn loại dữ liệu:", ("Dữ liệu giả lập", "Dữ liệu tùy chỉnh"), key="dbscan_data_option_2")

    if data_option == "Dữ liệu giả lập":
        # Dữ liệu mẫu (phân bố tương tự ví dụ trước)
        np.random.seed(42)
        v1_class0 = np.random.normal(1.5, 0.2, 20)  # Cluster 0: X1 quanh 1.5
        v2_class0 = np.random.normal(2.4, 0.2, 20)  # Cluster 0: X2 quanh 2.4
        v1_class1 = np.random.normal(1.6, 0.2, 20)  # Cluster 1: X1 quanh 1.6
        v2_class1 = np.random.normal(2.6, 0.2, 20)  # Cluster 1: X2 quanh 2.6
        v1 = np.concatenate([v1_class0, v1_class1])
        v2 = np.concatenate([v2_class0, v2_class1])
        X = np.array([[x, y] for x, y in zip(v1, v2)])
        original_label = np.array([0] * 20 + [1] * 20)
    else:
        # Dữ liệu tùy chỉnh
        st.markdown('<p style="font-size:20px;">Thêm các cặp giá trị X1, X2 (nhãn sẽ được phân cụm bởi K-means):</p>', unsafe_allow_html=True)
        if 'custom_data' not in st.session_state:
            st.session_state.custom_data = {'X1': [], 'X2': []}

        col1, col2 = st.columns(2)
        with col1:
            x1_input = st.number_input("Giá trị X1:", value=1.2, step=0.1, key="x1_input")
        with col2:
            x2_input = st.number_input("Giá trị X2:", value=2.3, step=0.1, key="x2_input")

        if st.button("➕ Thêm điểm"):
            st.session_state.custom_data['X1'].append(x1_input)
            st.session_state.custom_data['X2'].append(x2_input)
            st.rerun()

        if st.session_state.custom_data['X1']:
            df = pd.DataFrame(st.session_state.custom_data)
            st.markdown('<p style="font-size:18px;">Dữ liệu đã nhập:</p>', unsafe_allow_html=True)
            st.dataframe(df)

            delete_index = st.selectbox("Chọn điểm để xóa (nếu cần):", options=range(len(st.session_state.custom_data['X1'])), format_func=lambda i: f"Điểm {i}: X1={st.session_state.custom_data['X1'][i]}, X2={st.session_state.custom_data['X2'][i]}")
            if st.button("🗑️ Xóa điểm"):
                st.session_state.custom_data['X1'].pop(delete_index)
                st.session_state.custom_data['X2'].pop(delete_index)
                st.rerun()

            X = np.array([st.session_state.custom_data['X1'], st.session_state.custom_data['X2']]).T
            original_label = np.zeros(len(X))  # Nhãn ban đầu không quan trọng
            if len(X) < 2:
                st.error("Vui lòng nhập ít nhất 2 cặp dữ liệu để phân cụm!")
                return
        else:
            st.warning("Chưa có dữ liệu nào được thêm. Hãy nhập ít nhất 2 cặp X1, X2 để tiếp tục!")
            return

    # Phần 1: Lý thuyết với ví dụ đơn giản
    st.markdown('<h2 style="font-size:32px; color:#32CD32;">📚 1. K-means là gì và cách hoạt động?</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:22px;">
    ❓ K-means là một thuật toán unsupervised learning dùng để phân cụm dữ liệu thành K nhóm dựa trên sự tương đồng.<br>
    🚀 <b>Các bước chính:</b><br>
    - Chọn ngẫu nhiên K tâm cụm (centers) ban đầu.<br>
    - Gán mỗi điểm dữ liệu vào cụm gần nhất dựa trên khoảng cách Euclidean.<br>
    - Cập nhật tâm cụm bằng trung bình của các điểm trong cụm.<br>
    - Lặp lại cho đến khi tâm cụm không đổi.<br>
    📐 <b>Thuật toán:</b> Tối ưu hóa bằng cách giảm tổng khoảng cách từ các điểm đến tâm cụm của chúng.<br>
    </p>
    """, unsafe_allow_html=True)

    # Ví dụ đơn giản với 2 vòng lặp
    st.markdown('<h3 style="font-size:26px; color:#4682B4;">📋 Ví dụ: Phân cụm 4 điểm với K=2</h3>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:22px;">
    Giả sử ta có 4 điểm 2D: <br>
    - A = (1, 1)<br>
    - B = (2, 1)<br>
    - C = (4, 3)<br>
    - D = (5, 4)<br>
    Mục tiêu: Phân thành 2 cụm (K=2) qua 2 vòng lặp.<br>
    </p>
    """, unsafe_allow_html=True)

    # Vòng lặp 1
    st.markdown('<h4 style="font-size:24px; color:#FF4500;">🔄 Vòng lặp 1:</h4>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:20px;">
    - <b>Khởi tạo tâm cụm:</b> Chọn ngẫu nhiên A=(1, 1) cho Cluster 0, C=(4, 3) cho Cluster 1.<br>
    - <b>Gán nhãn:</b> Tính khoảng cách Euclidean từ mỗi điểm đến tâm cụm:<br>
      - A đến (1, 1): 0 → Cluster 0<br>
      - B đến (1, 1): √((2-1)² + (1-1)²) = 1 → Cluster 0<br>
      - C đến (4, 3): 0 → Cluster 1<br>
      - D đến (4, 3): √((5-4)² + (4-3)²) = √2 ≈ 1.41 → Cluster 1<br>
    - <b>Cập nhật tâm cụm:</b> Trung bình của các điểm trong cụm:<br>
      - Cluster 0: (A, B) → ((1+2)/2, (1+1)/2) = (1.5, 1)<br>
      - Cluster 1: (C, D) → ((4+5)/2, (3+4)/2) = (4.5, 3.5)<br>
    </p>
    """, unsafe_allow_html=True)

    # Vòng lặp 2
    st.markdown('<h4 style="font-size:24px; color:#FF4500;">🔄 Vòng lặp 2:</h4>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:20px;">
    - <b>Gán nhãn:</b> Tính lại khoảng cách:<br>
      - A đến (1.5, 1): √((1-1.5)² + (1-1)²) = 0.5 → Cluster 0<br>
      - B đến (1.5, 1): √((2-1.5)² + (1-1)²) = 0.5 → Cluster 0<br>
      - C đến (4.5, 3.5): √((4-4.5)² + (3-3.5)²) = 0.707 → Cluster 1<br>
      - D đến (4.5, 3.5): √((5-4.5)² + (4-3.5)²) = 0.707 → Cluster 1<br>
    - <b>Cập nhật tâm cụm:</b> Trung bình:<br>
      - Cluster 0: (A, B) → ((1+2)/2, (1+1)/2) = (1.5, 1)<br>
      - Cluster 1: (C, D) → ((4+5)/2, (3+4)/2) = (4.5, 3.5)<br>
    - <b>Kết luận:</b> Tâm cụm không đổi, thuật toán hội tụ sau 2 vòng lặp.<br>
    </p>
    """, unsafe_allow_html=True)

    # Hiển thị biểu đồ ví dụ
    example_data = np.array([[1, 1], [2, 1], [4, 3], [5, 4]])
    example_labels = np.array([0, 0, 1, 1])  # Kết quả sau 2 vòng lặp
    example_centers = np.array([[1.5, 1], [4.5, 3.5]])
    fig, ax = plt.subplots()
    kmeans_display(example_data, example_labels, example_centers)
    ax.set_title('Ví dụ phân cụm 4 điểm (K=2)')
    st.pyplot(fig)

    st.markdown("""
    <p style="font-size:20px;">
    💡 <b>Lưu ý:</b> Bạn có thể tự tính toán các khoảng cách và trung bình để kiểm tra!<br>
    </p>
    """, unsafe_allow_html=True)

    # Phần 1.5: Hình động minh họa
    st.markdown('<h2 style="font-size:32px; color:#FFD700;">🎥 1.5. Quá trình phân cụm với hình động</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">📈 Xem quá trình K-means phân cụm dữ liệu qua các bước:</p>', unsafe_allow_html=True)

    # Tạo hình động
    K = 2  # Số cụm cố định là 2
    fig, ax = plt.subplots()
    colors = ['red' if l == 0 else 'blue' for l in original_label]
    scat = ax.scatter(X[:, 0], X[:, 1], c=colors, s=80, edgecolors='k')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Quá trình phân cụm của K-means')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Cluster 0'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Cluster 1')
    ]
    ax.legend(handles=legend_elements, title="Nhãn", loc='upper right')

    def init():
        ax.clear()
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=80, edgecolors='k')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title('Quá trình phân cụm của K-means')
        ax.legend(handles=legend_elements, title="Nhãn", loc='upper right')
        return ax,

    def update(frame, centers_list, labels_list):
        ax.clear()
        colors = ['red' if l == 0 else 'blue' for l in labels_list[frame]]
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=80, edgecolors='k')
        ax.scatter(centers_list[frame][:, 0], centers_list[frame][:, 1], c='green', marker='x', s=200, label='Centers')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title(f'Bước {frame + 1}')
        ax.legend(handles=legend_elements + [Line2D([0], [0], marker='x', color='w', markerfacecolor='green', markersize=10, label='Centers')], title="Nhãn", loc='upper right')
        return ax,

    # Chạy K-means để lấy dữ liệu cho hình động
    with st.spinner("🔄 Đang tính toán phân cụm..."):
        centers, labels, iterations = kmeans(X, K)
        frames = min(iterations + 1, 10)  # Giới hạn số khung hình

    # Tạo animation
    ani = animation.FuncAnimation(fig, update, init_func=init, fargs=(centers, labels), frames=frames, interval=2000, repeat=False, blit=False)

    # Lưu animation thành GIF
    try:
        writer = PillowWriter(fps=0.5)
        with writer.saving(fig, "kmeans_animation.gif", dpi=100):
            for i in range(frames):
                update(i, centers, labels)
                writer.grab_frame()
    except Exception as e:
        st.error(f"Lỗi khi lưu GIF: {e}")
        return

    # Hiển thị GIF trong Streamlit
    with open("kmeans_animation.gif", "rb") as file:
        gif_data = file.read()
    gif_base64 = base64.b64encode(gif_data).decode('utf-8')
    st.markdown(f'<img src="data:image/gif;base64,{gif_base64}" alt="animation">', unsafe_allow_html=True)

    # Phần 2: Trực quan hóa kết quả cuối cùng
    st.markdown('<h2 style="font-size:32px; color:#FFD700;">📈 2. Kết quả phân cụm</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">👀 Xem kết quả phân cụm sau khi K-means hoàn tất:</p>', unsafe_allow_html=True)

    fig, ax = plt.subplots()
    final_labels = labels[-1]
    kmeans_display(X, final_labels, centers[-1])
    ax.set_title('Kết quả phân cụm cuối cùng (K=2)')
    st.pyplot(fig)

    st.markdown(f'<p style="font-size:20px; color:#4682B4;">📊 Số bước lặp: {iterations}</p>', unsafe_allow_html=True)

    # Phần 3: Tùy chỉnh và dự đoán
    st.markdown('<h2 style="font-size:32px; color:#00CED1;">🎮 3. Thử nghiệm với điểm mới</h2>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:20px;">Nhập giá trị X1 ({min(X[:, 0]):.2f} đến {max(X[:, 0]):.2f}) và X2 ({min(X[:, 1]):.2f} đến {max(X[:, 1]):.2f}):</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        input_x1 = st.number_input("X1:", min_value=float(min(X[:, 0])), max_value=float(max(X[:, 0])), value=float(np.mean(X[:, 0])), step=0.1)
    with col2:
        input_x2 = st.number_input("X2:", min_value=float(min(X[:, 1])), max_value=float(max(X[:, 1])), value=float(np.mean(X[:, 1])), step=0.1)

    X_new = np.array([[input_x1, input_x2]])
    closest_center = pairwise_distances_argmin(X_new, centers[-1])[0]
    predicted_cluster = closest_center

    fig, ax = plt.subplots()
    kmeans_display(X, final_labels, centers[-1])
    ax.scatter([input_x1], [input_x2], c='green', marker='*', s=150, label='Điểm mới')
    ax.set_title('Dự đoán cụm cho điểm mới')
    ax.legend(handles=legend_elements + [Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=10, label='Điểm mới')], title="Nhãn", loc='upper right')
    st.pyplot(fig)

    st.markdown(f'<p style="font-size:20px;">📈 <b>Dự đoán:</b> Điểm (X1={input_x1:.2f}, X2={input_x2:.2f}) thuộc Cluster {predicted_cluster} ({"Đỏ" if predicted_cluster == 0 else "Xanh"})</p>', unsafe_allow_html=True)

    # Phần 4: Hạn chế
    st.markdown('<h2 style="font-size:32px; color:#FFA500;">⚠️ 4. Hạn chế của K-means</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:22px;">
    👍 <b>Ưu điểm:</b><br>
    - 🌟 Đơn giản, nhanh chóng với dữ liệu lớn.<br>
    - 📈 Hiệu quả khi cụm có hình dạng cầu.<br>
    👎 <b>Nhược điểm:</b><br>
    - 🚨 Nhạy cảm với giá trị khởi tạo ban đầu.<br>
    - ⚙️ Không hoạt động tốt với cụm không đều hoặc không phải hình cầu.<br>
    💡 <b>Gợi ý:</b> Thử nhiều lần khởi tạo hoặc sử dụng K-means++ để cải thiện.
    </p>
    """, unsafe_allow_html=True)

    # Phần 5: Tài liệu tham khảo
    st.markdown('<h2 style="font-size:32px; color:#1E90FF;">🔗 5. Tài liệu tham khảo</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">📖 Xem chi tiết về K-means tại <a href="https://machinelearningcoban.com/2017/01/01/kmeans/?form=MG0AV3">Machine Learning Cơ Bản</a>.</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">🙏 Cảm ơn bạn đã tham gia khám phá K-means!</p>', unsafe_allow_html=True)

def input_mlflow():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/PTToan250303/Linear_replication.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    os.environ["MLFLOW_TRACKING_USERNAME"] = "PTToan250303"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "5ca8caf353d564c358852da97c7487e64fc30a73"
    mlflow.set_experiment("Classification")

def train():
    st.header("⚙️ Chọn mô hình & Huấn luyện")

    if "X_train" not in st.session_state:
        st.warning("⚠️ Vui lòng chia dữ liệu trước khi train!")
        return

    X_train = st.session_state["X_train"]
    y_train = st.session_state["y_train"]

    X_train_norm = X_train / 255.0  # Chuẩn hóa

    model_choice = st.selectbox("Chọn mô hình:", ["K-Means", "DBSCAN"])

    if model_choice == "K-Means":
        st.markdown("🔹 **K-Means**")
        n_clusters = st.slider("🔢 Chọn số cụm (K):", 2, 20, 10)
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_norm)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    elif model_choice == "DBSCAN":
        st.markdown("🛠️ **DBSCAN**")
        eps = st.slider("📏 Bán kính lân cận (eps):", 0.1, 10.0, 0.5)
        min_samples = st.slider("👥 Số điểm tối thiểu trong cụm:", 2, 20, 5)
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_norm)
        model = DBSCAN(eps=eps, min_samples=min_samples)

    input_mlflow()
    if "experiment_name" not in st.session_state:
        st.session_state["experiment_name"] = "My_Experiment"

    experiment_name = st.text_input("🔹 Nhập tên Experiment:", st.session_state["experiment_name"], key="experiment_name_input")    

    if experiment_name:
        st.session_state["experiment_name"] = experiment_name

    mlflow.set_experiment(experiment_name)
    st.write(f"✅ Experiment Name: {experiment_name}")

    if st.button("🚀 Huấn luyện mô hình"):
        if "run_name" not in st.session_state:
            st.session_state["run_name"] = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # Đặt tên dựa vào thời gian

        with mlflow.start_run(run_name=st.session_state["run_name"]):
            model.fit(X_train_pca)
            st.success("✅ Huấn luyện thành công!")

            labels = model.labels_

            if model_choice == "K-Means":
                label_mapping = {}
                for i in range(n_clusters):
                    mask = labels == i
                    if np.sum(mask) > 0:
                        most_common_label = mode(y_train[mask], keepdims=True).mode[0]
                        label_mapping[i] = most_common_label

                predicted_labels = np.array([label_mapping[label] for label in labels])
                accuracy = np.mean(predicted_labels == y_train)
                st.write(f"🎯 **Độ chính xác của mô hình:** `{accuracy * 100:.2f}%`")

                # Log vào MLflow
                mlflow.log_param("model", "K-Means")
                mlflow.log_param("n_clusters", n_clusters)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.sklearn.log_model(model, "kmeans_model")

            elif model_choice == "DBSCAN":
                unique_clusters = set(labels) - {-1}
                n_clusters_found = len(unique_clusters)
                noise_ratio = np.sum(labels == -1) / len(labels)
                st.write(f"🔍 **Số cụm tìm thấy:** `{n_clusters_found}`")
                st.write(f"🚨 **Tỉ lệ nhiễu:** `{noise_ratio * 100:.2f}%`")

                # Log vào MLflow
                mlflow.log_param("model", "DBSCAN")
                mlflow.log_param("eps", eps)
                mlflow.log_param("min_samples", min_samples)
                mlflow.log_metric("n_clusters_found", n_clusters_found)
                mlflow.log_metric("noise_ratio", noise_ratio)
                mlflow.sklearn.log_model(model, "dbscan_model")

            if "models" not in st.session_state:
                st.session_state["models"] = []

            model_name = model_choice.lower().replace(" ", "_")
            count = 1
            new_model_name = model_name
            while any(m["name"] == new_model_name for m in st.session_state["models"]):
                new_model_name = f"{model_name}_{count}"
                count += 1

            st.session_state["models"].append({"name": new_model_name, "model": model})
            st.write(f"🔹 **Mô hình đã được lưu với tên:** `{new_model_name}`")
            st.write(f"📋 **Danh sách các mô hình:** {[m['name'] for m in st.session_state['models']]}")
            mlflow.end_run()
            st.success(f"✅ Đã log dữ liệu cho **Train_{st.session_state['run_name']}**!")
            st.markdown(f"### 🔗 [Truy cập MLflow DAGsHub]({st.session_state['mlflow_url']})")




import streamlit as st
import numpy as np
import random
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
from sklearn.decomposition import PCA

def preprocess_canvas_image(canvas_result):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data[:, :, 0].astype(np.uint8))
        img = img.resize((28, 28)).convert("L")  # Resize và chuyển thành grayscale
        img = np.array(img, dtype=np.float32) / 255.0  # Chuẩn hóa về [0, 1]
        return img.reshape(1, -1)  # Chuyển thành vector 1D
    return None


def du_doan():
    st.header("✍️ Vẽ dữ liệu để dự đoán cụm")

    # Ensure st.session_state["models"] is a list
    if "models" not in st.session_state or not isinstance(st.session_state["models"], list):
        st.session_state["models"] = []
        st.warning("⚠️ Danh sách mô hình trống! Hãy huấn luyện trước.")
        return

    # Kiểm tra danh sách mô hình đã huấn luyện
    if not st.session_state["models"]:
        st.warning("⚠️ Không có mô hình nào được lưu! Hãy huấn luyện trước.")
        return

    # Lấy danh sách tên mô hình
    model_names = [m["name"] for m in st.session_state["models"] if isinstance(m, dict)]

    # Kiểm tra danh sách có rỗng không
    if not model_names:
        st.warning("⚠️ Chưa có mô hình nào được huấn luyện.")
        return

    # 📌 Chọn mô hình
    model_option = st.selectbox("🔍 Chọn mô hình đã huấn luyện:", model_names)

    # Tìm mô hình tương ứng
    try:
        model = next(m["model"] for m in st.session_state["models"] if isinstance(m, dict) and m["name"] == model_option)
    except StopIteration:
        st.error(f"⚠️ Không tìm thấy mô hình với tên {model_option}!")
        return

    # 🆕 Cập nhật key cho canvas khi nhấn "Tải lại"
    if "key_value" not in st.session_state:
        st.session_state.key_value = str(random.randint(0, 1000000))

    if st.button("🔄 Tải lại"):
        st.session_state.key_value = str(random.randint(0, 1000000))
        st.rerun()

    # ✍️ Vẽ dữ liệu
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

    if st.button("Dự đoán cụm"):
        img = preprocess_canvas_image(canvas_result)

        if img is not None:
            X_train = st.session_state["X_train"]
            # Hiển thị ảnh sau xử lý
            st.image(Image.fromarray((img.reshape(28, 28) * 255).astype(np.uint8)), caption="Ảnh sau xử lý", width=100)
            
            pca = PCA(n_components=2)
            pca.fit(X_train)
            img_reduced = pca.transform(img.reshape(1, -1))  # Đã sửa lỗi

            # Dự đoán với K-Means hoặc DBSCAN
            if isinstance(model, KMeans):
                predicted_cluster = model.predict(img_reduced)[0]  # Dự đoán từ ảnh đã PCA
                st.subheader(f"🔢 Cụm dự đoán: {predicted_cluster}")

            elif isinstance(model, DBSCAN):
                model.fit(X_train)  # Fit trước với tập huấn luyện
                predicted_cluster = model.fit_predict(img_reduced)[0]
                if predicted_cluster == -1:
                    st.subheader("⚠️ Điểm này không thuộc cụm nào!")
                else:
                    st.subheader(f"🔢 Cụm dự đoán: {predicted_cluster}")

        else:
            st.error("⚠️ Hãy vẽ một số trước khi bấm Dự đoán!")

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




def ClusteringAlgorithms():
  
    st.markdown("""
            <style>
            .title { font-size: 48px; font-weight: bold; text-align: center; color: #4682B4; margin-top: 50px; }
            .subtitle { font-size: 24px; text-align: center; color: #4A4A4A; }
            hr { border: 1px solid #ddd; }
            </style>
            <div class="title">MNIST Clustering Algorithms App</div>
            <hr>
        """, unsafe_allow_html=True)
    
   
    # === Sidebar để chọn trang ===
    # === Tạo Tabs ===
    tab1, tab2, tab3, tab4,tab5 ,tab6= st.tabs(["📘 Lý thuyết K-means", "📘 Lý thuyết DBSCAN", "📘 Data" ,"⚙️ Huấn luyện", "🔢 Dự đoán","🔥 Mlflow"])

    with tab1:
        ly_thuyet_kmeans()

    with tab2:
        ly_thuyet_dbscan()
    
    with tab3:
        data()
        
    with tab4:
       
        
        
        
        split_data()
        train()
        
    
    with tab5:
        
        du_doan() 
    with tab6:
        
        show_experiment_selector() 




            
if __name__ == "__main__":
    ClusteringAlgorithms()