import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import joblib
import pandas as pd
import os
import mlflow
from mlflow.tracking import MlflowClient
import random
from datetime import datetime  
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import base64
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D

# Hàm chuẩn hóa dữ liệu với scaler lưu trữ
@st.cache_data
def standardize_data(X, fit=True, _scaler=None):
    if fit or _scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler
    else:
        return _scaler.transform(X), _scaler

# Hàm hiển thị dữ liệu và đường biên quyết định của SVM
@st.cache_data
def svm_display(X, y, _clf, title="Phân lớp với SVM", new_point=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Class 0', s=80, edgecolors='k')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Class 1', s=80, edgecolors='k')

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = _clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'], 
               labels=['Lề -1', 'Biên quyết định', 'Lề +1'])

    ax.scatter(_clf.support_vectors_[:, 0], _clf.support_vectors_[:, 1], s=200, facecolors='none', 
               edgecolors='green', label='Support Vectors')

    if new_point is not None:
        ax.scatter(new_point[0], new_point[1], c='yellow', marker='*', s=150, label='Điểm mới')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.axis('equal')
    return fig

# Hàm dự đoán điểm mới với SVM
def predict_new_point_svm(clf, new_point):
    return clf.predict([new_point])[0]

# Hàm chính
def ly_thuyet_svm():
    st.markdown('<h1 style="color:#FF4500; text-align:center;">Support Vector Machine (SVM)</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:24px; color:#4682B4;">📝 Tìm hiểu thuật toán SVM để phân lớp dữ liệu.</p>', unsafe_allow_html=True)

    # Chọn nguồn dữ liệu
    st.markdown('<h2 style="font-size:28px; color:#32CD32;">📊 Chọn nguồn dữ liệu</h2>', unsafe_allow_html=True)
    data_option = st.radio("Chọn loại dữ liệu:", ("Dữ liệu giả lập", "Dữ liệu tùy chỉnh"),key="SVM_data_option_1")

    if data_option == "Dữ liệu giả lập":
        X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=42)
        X, scaler = standardize_data(X, fit=True)
    else:
        st.markdown('<p style="font-size:20px;">Thêm các cặp giá trị X1, X2 và nhãn (0 hoặc 1):</p>', unsafe_allow_html=True)
        if 'custom_data' not in st.session_state:
            st.session_state.custom_data = {'X1': [], 'X2': [], 'Label': []}

        col1, col2, col3 = st.columns(3)
        with col1:
            x1_input = st.number_input("Giá trị X1:", value=0.0, step=0.1, key="x1_input")
        with col2:
            x2_input = st.number_input("Giá trị X2:", value=0.0, step=0.1, key="x2_input")
        with col3:
            label_input = st.selectbox("Nhãn (0 hoặc 1):", [0, 1], key="label_input")

        if st.button("➕ Thêm điểm"):
            st.session_state.custom_data['X1'].append(x1_input)
            st.session_state.custom_data['X2'].append(x2_input)
            st.session_state.custom_data['Label'].append(label_input)
            st.rerun()

        if st.session_state.custom_data['X1']:
            df = pd.DataFrame(st.session_state.custom_data)
            st.markdown('<p style="font-size:18px;">Dữ liệu đã nhập:</p>', unsafe_allow_html=True)
            st.dataframe(df)

            delete_index = st.selectbox("Chọn điểm để xóa (nếu cần):", options=range(len(df)), 
                                        format_func=lambda i: f"Điểm {i}: X1={df['X1'][i]}, X2={df['X2'][i]}, Label={df['Label'][i]}")
            if st.button("🗑️ Xóa điểm"):
                for key in st.session_state.custom_data:
                    st.session_state.custom_data[key].pop(delete_index)
                st.rerun()

            X = np.array([df['X1'], df['X2']]).T
            y = np.array(df['Label'])
            if len(np.unique(y)) < 2 or len(X) < 2:
                st.error("Cần ít nhất 2 điểm với 2 nhãn khác nhau để phân lớp!")
                return
            X, scaler = standardize_data(X, fit=True)
        else:
            st.warning("Chưa có dữ liệu nào được thêm. Hãy nhập ít nhất 2 điểm với nhãn 0 và 1!")
            return

    # Phần 1: Lý thuyết chi tiết
    st.markdown('<h2 style="font-size:32px; color:#32CD32;">📚 1. SVM là gì và cách hoạt động?</h2>', unsafe_allow_html=True)

    st.markdown("""
    ❓ Support Vector Machine (SVM) là một thuật toán học máy có giám sát dùng để phân lớp (classification) hoặc hồi quy (regression). Trong bài này, chúng ta tập trung vào phân lớp nhị phân. \n\n
    🚀 Ý tưởng chính: SVM tìm một siêu phẳng (hyperplane) trong không gian đặc trưng để phân tách hai lớp dữ liệu sao cho khoảng cách từ siêu phẳng đến các điểm gần nhất của mỗi lớp (lề - margin) là lớn nhất.
    """)


    st.markdown(r"""
    ### 🔹 **Các khái niệm cơ bản**
    - **Siêu phẳng (Hyperplane)**: Trong không gian $$ d $$-chiều, siêu phẳng là một mặt phẳng $$(d-1)$$-chiều được định nghĩa bởi phương trình:
      $$ w^T x + b = 0 $$
      Trong đó:
      - $$ w $$: Vector pháp tuyến của siêu phẳng (quyết định hướng).
      - $$ x $$: Vector dữ liệu.
      - $$ b $$: Hệ số chặn (quyết định vị trí siêu phẳng so với gốc tọa độ).

    - **Lề (Margin)**: Khoảng cách từ siêu phẳng đến các điểm dữ liệu gần nhất (support vectors). SVM tối ưu hóa để lề này lớn nhất, giúp mô hình ít nhạy cảm với nhiễu.

    - **Support Vectors**: Các điểm dữ liệu nằm trên ranh giới của lề, tức là các điểm thỏa mãn $$|w^T x_i + b| = 1$$. Đây là các điểm quan trọng nhất quyết định siêu phẳng.

    ### 🔹 **SVM Hard Margin**
    Khi dữ liệu có thể phân tách tuyến tính hoàn toàn (linearly separable), SVM tìm siêu phẳng sao cho không có điểm dữ liệu nào nằm trong lề. Bài toán tối ưu hóa được định nghĩa như sau:

    **Mục tiêu**: Tối đa hóa lề, tức là tối đa hóa $$\frac{2}{||w||}$$ (vì lề tỷ lệ nghịch với độ dài của $$w$$.  
    Để dễ tính toán, ta chuyển thành bài toán tối thiểu hóa:
    $$ \min_{w, b} \frac{1}{2} ||w||^2 $$
    Với ràng buộc:
    $$ y_i (w^T x_i + b) \geq 1, \quad \forall i $$
    Trong đó:
    - $$y_i$$: Nhãn của điểm $$(x_i)$$ ($$(y_i = -1)$$ hoặc 1).
    - $$w^T x_i + b \geq 1$$ nếu $$y_i = 1$$, và $$w^T x_i + b \leq -1$$ nếu $$y_i = -1$$.

    **Giải thích công thức**:
    - $$||w||$$: Độ dài của vector pháp tuyến $$w$$, tính bằng $$\sqrt{w_1^2 + w_2^2 + \dots + w_d^2}$$.
    - Lề được tính là $$\frac{2}{||w||}$$, vì khoảng cách từ siêu phẳng đến các điểm trên ranh giới lề là $$\frac{1}{||w||}$$ ở mỗi phía.
    - Ràng buộc $$y_i (w^T x_i + b) \geq 1$$ đảm bảo tất cả điểm dữ liệu nằm ngoài lề hoặc trên ranh giới lề.

    ### 🔹 **SVM Soft Margin**
    Trong thực tế, dữ liệu thường không phân tách tuyến tính hoàn toàn. SVM Soft Margin cho phép một số điểm nằm trong lề hoặc thậm chí bị phân lớp sai, bằng cách thêm biến chùng (slack variables) \(\xi_i\). Bài toán tối ưu hóa trở thành:
    $$ \min_{w, b, \xi} \frac{1}{2} ||w||^2 + C \sum_{i=1}^n \xi_i $$
    Với ràng buộc:
    $$ y_i (w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad \forall i $$
    Trong đó:
    - $$\xi_i$$: Độ sai lệch của điểm $$x_i$$ so với lề (nếu $$xi_i = 0$$, điểm nằm ngoài lề; nếu $$0 < \xi_i \leq 1$$, điểm nằm trong lề; nếu $$xi_i > 1$$, điểm bị phân lớp sai).
    - $$C$$: Tham số điều chỉnh, cân bằng giữa việc tối đa hóa lề ($$||w||^2$$ nhỏ) và giảm lỗi phân lớp ($$sum \xi_i$$ nhỏ).
        - $$C$$ lớn: Ưu tiên giảm lỗi phân lớp, lề hẹp hơn.
        - $$C$$ nhỏ: Ưu tiên lề lớn, chấp nhận nhiều lỗi hơn.

    ### 🔹 **Kernel Trick**
    Khi dữ liệu không thể phân tách tuyến tính trong không gian ban đầu, SVM sử dụng hàm kernel để ánh xạ dữ liệu lên không gian chiều cao hơn. Siêu phẳng trong không gian mới có thể là tuyến tính, tương ứng với một ranh giới phi tuyến trong không gian gốc.  
    Hàm kernel phổ biến:
    - **Linear**: $$K(x_i, x_j) = x_i^T x_j$$.
    - **Polynomial**: $$K(x_i, x_j) = (x_i^T x_j + 1)^d$$.
    - **RBF (Radial Basis Function)**: $$K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$$.

    **Công thức dự đoán**: Với kernel, hàm quyết định trở thành:
    $$ f(x) = \text{sign} \left( \sum_{i \in SV} \alpha_i y_i K(x_i, x) + b \right) $$
    Trong đó:
    - $$SV$$: Tập hợp các support vectors.
    - $$\alpha_i$$: Hệ số Lagrange từ bài toán đối ngẫu (dual problem).
    """, unsafe_allow_html=True)

    # Ví dụ đơn giản
    st.markdown('<h3 style="font-size:26px; color:#4682B4;">📋 Ví dụ: Phân lớp 4 điểm với Hard Margin</h3>', unsafe_allow_html=True)
    example_data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    example_labels = np.array([0, 0, 1, 1])
    example_data, _ = standardize_data(example_data, fit=True)
    clf_example = SVC(kernel='linear', C=1e6)
    clf_example.fit(example_data, example_labels)
    st.pyplot(svm_display(example_data, example_labels, clf_example, "Ví dụ phân lớp 4 điểm với SVM Hard Margin"))

    # Phần 2: Kết quả phân lớp
    st.markdown('<h2 style="font-size:32px; color:#FFD700;">📈 2. Kết quả phân lớp với SVM</h2>', unsafe_allow_html=True)
    kernel = st.selectbox("Chọn kernel:", ['linear', 'rbf', 'poly'], index=0)
    C = st.slider("Tham số C (độ mềm của lề):", 0.1, 10.0, 1.0, 0.1)
    clf = SVC(kernel=kernel, C=C, random_state=42)
    clf.fit(X, y)
    st.pyplot(svm_display(X, y, clf, f"Kết quả phân lớp (kernel={kernel}, C={C})"))
    st.markdown(f'<p style="font-size:20px; color:#4682B4;">📊 Số support vectors: {len(clf.support_vectors_)}</p>', unsafe_allow_html=True)

    # Phần 3: Dự đoán điểm mới
    st.markdown('<h2 style="font-size:32px; color:#00CED1;">🎮 3. Thử nghiệm với điểm mới</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        input_x1 = st.number_input("X1:", min_value=float(min(X[:, 0])), max_value=float(max(X[:, 0])), value=float(np.mean(X[:, 0])), step=0.1)
    with col2:
        input_x2 = st.number_input("X2:", min_value=float(min(X[:, 1])), max_value=float(max(X[:, 1])), value=float(np.mean(X[:, 1])), step=0.1)

    X_new = np.array([[input_x1, input_x2]])
    X_new_scaled, _ = standardize_data(X_new, fit=False, _scaler=scaler)
    predicted_label = predict_new_point_svm(clf, X_new_scaled[0])
    st.pyplot(svm_display(X, y, clf, "Dự đoán điểm mới với SVM", new_point=X_new_scaled[0]))
    st.markdown(f'<p style="font-size:20px;">📈 <b>Dự đoán:</b> Điểm (X1={input_x1:.2f}, X2={input_x2:.2f}) thuộc lớp {predicted_label}</p>', unsafe_allow_html=True)

    # Phần 4: Ưu điểm và hạn chế
    st.markdown('<h2 style="font-size:32px; color:#FFA500;">⚠️ 4. Ưu điểm và hạn chế</h2>', unsafe_allow_html=True)
    st.markdown("""
    👍 **Ưu điểm:** 
    - Hiệu quả với dữ liệu chiều cao nhờ kernel trick.
    - Tối ưu lề lớn, giảm nguy cơ overfitting.
    - Linh hoạt với dữ liệu không tuyến tính. \n\n
    👎 **Hạn chế:**
    - Nhạy cảm với tham số $$C$$ và lựa chọn kernel.
    - Tính toán phức tạp, chậm với dữ liệu lớn.
    - Khó giải thích kết quả trong không gian kernel.
    """)
    st.markdown('<h2 style="font-size:32px; color:#1E90FF;">🔗 5. Tài liệu tham khảo</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">📖 Xem chi tiết về SVM tại <a href="https://machinelearningcoban.com/2017/04/09/smv/?form=MG0AV3">Machine Learning cơ bản - Bài 19: Support Vector Machine</a>.</p>', unsafe_allow_html=True)
# Hàm tính Entropy
def entropy(y):
    if len(y) == 0:
        return 0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Thêm 1e-10 để tránh log(0)

# Hàm tổng với nội dung
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import base64
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from matplotlib.lines import Line2D

# Hàm lý thuyết Decision Tree
def ly_thuyet_Decision():
    # Tiêu đề chính
    st.markdown('<h1 style="color:#FF4500; text-align:center;">🌟 Decision Tree 🌟</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:24px; color:#4682B4;">📝 Tìm hiểu cách sử dụng Decision Tree để phân loại dữ liệu một cách trực quan và hiệu quả.</p>', unsafe_allow_html=True)

    # Chọn nguồn dữ liệu
    st.markdown('<h2 style="font-size:28px; color:#32CD32;">📊 Chọn nguồn dữ liệu</h2>', unsafe_allow_html=True)
    data_option = st.radio("Chọn loại dữ liệu:", ("Dữ liệu giả lập", "Dữ liệu tùy chỉnh"), key="SVM_data_option_2")

    if data_option == "Dữ liệu giả lập":
        # Dữ liệu mẫu (phân bố tương tự ví dụ trước)
        np.random.seed(42)
        v1_class0 = np.random.normal(1.5, 0.1, 16)  # Lớp 0: X1 quanh 1.5
        v2_class0 = np.random.normal(2.4, 0.1, 16)  # Lớp 0: X2 quanh 2.4
        v1_class1 = np.random.normal(1.6, 0.1, 4)  # Lớp 1: X1 quanh 1.6
        v2_class1 = np.random.normal(2.55, 0.05, 4)  # Lớp 1: X2 quanh 2.55
        v1 = np.concatenate([v1_class0, v1_class1])
        v2 = np.concatenate([v2_class0, v2_class1])
        X = np.array([[x, y] for x, y in zip(v1, v2)])
        y = np.array([0] * 16 + [1] * 4)  # 16 mẫu lớp 0, 4 mẫu lớp 1
    else:
        # Dữ liệu tùy chỉnh với giao diện đơn giản
        st.markdown('<p style="font-size:20px;">Thêm các cặp giá trị X1, X2 và nhãn Y (0 hoặc 1):</p>', unsafe_allow_html=True)

        # Khởi tạo session state để lưu dữ liệu
        if 'custom_data' not in st.session_state:
            st.session_state.custom_data = {'X1': [], 'X2': [], 'Y': []}

        # Ba cột để nhập X1, X2, Y
        col1, col2, col3 = st.columns(3)
        with col1:
            x1_input = st.number_input("Giá trị X1:", value=1.2, step=0.1, key="x1_input")
        with col2:
            x2_input = st.number_input("Giá trị X2:", value=2.3, step=0.1, key="x2_input")
        with col3:
            y_input = st.selectbox("Nhãn Y:", [0, 1], key="y_input")

        # Nút thêm điểm
        if st.button("➕ Thêm điểm"):
            st.session_state.custom_data['X1'].append(x1_input)
            st.session_state.custom_data['X2'].append(x2_input)
            st.session_state.custom_data['Y'].append(y_input)

        # Hiển thị dữ liệu đã nhập dưới dạng bảng
        if st.session_state.custom_data['X1']:
            df = pd.DataFrame(st.session_state.custom_data)
            st.markdown('<p style="font-size:18px;">Dữ liệu đã nhập:</p>', unsafe_allow_html=True)
            st.dataframe(df)

            # Tùy chọn xóa điểm
            delete_index = st.selectbox("Chọn điểm để xóa (nếu cần):", options=range(len(st.session_state.custom_data['X1'])), format_func=lambda i: f"Điểm {i}: X1={st.session_state.custom_data['X1'][i]}, X2={st.session_state.custom_data['X2'][i]}, Y={st.session_state.custom_data['Y'][i]}")
            if st.button("🗑️ Xóa điểm"):
                st.session_state.custom_data['X1'].pop(delete_index)
                st.session_state.custom_data['X2'].pop(delete_index)
                st.session_state.custom_data['Y'].pop(delete_index)
                st.rerun()

            # Chuyển dữ liệu thành numpy array
            X = np.array([st.session_state.custom_data['X1'], st.session_state.custom_data['X2']]).T
            y = np.array(st.session_state.custom_data['Y'])

            if len(X) < 2:
                st.error("Vui lòng nhập ít nhất 2 cặp dữ liệu để mô hình hóa!")
                return
        else:
            st.warning("Chưa có dữ liệu nào được thêm. Hãy nhập ít nhất 2 cặp X1, X2, Y để tiếp tục!")
            return

    # Định nghĩa legend_elements ở cấp độ toàn cục trong hàm
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='0'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='1')
    ]

    # Phần 1: Giới thiệu và lý thuyết
    st.markdown('<h2 style="font-size:32px; color:#32CD32;">📚 1. Decision Tree là gì và cách xây dựng?</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:22px;">
    ❓ Decision Tree (Cây quyết định) là một mô hình học máy phân loại hoặc hồi quy bằng cách chia không gian dữ liệu thành các vùng dựa trên các đặc trưng.<br>
    🚀 <b>Các khái niệm chính:</b><br>
    - Nút gốc (Root Node): Đại diện toàn bộ tập dữ liệu ban đầu.<br>
    - Nút bên trong (Internal Node): Đại diện cho một đặc trưng và ngưỡng phân chia.<br>
    - Nút lá (Leaf Node): Đại diện cho một lớp hoặc giá trị dự đoán.<br>
    📐 <b>Entropy và Information Gain:</b><br>
    </p>
    """, unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:22px;">
    - Entropy: Đo độ hỗn tạp của tập dữ liệu.<br>
    </p>
    """, unsafe_allow_html=True)
    # Hiển thị công thức Entropy
    st.latex(r'H(S) = - \sum p(i) \log_2 p(i)')

    st.markdown("""
    Trong đó:
    - $H(S)$: Entropy của tập dữ liệu ban đầu.
    - $p(i)$: Tỷ lệ của lớp $i$ trong tập dữ liệu.
    """)
    st.markdown("""
    <p style="font-size:22px;">
    - Information Gain: Đo mức độ giảm Entropy sau khi phân chia.<br>
    </p>
    """, unsafe_allow_html=True)
    # Hiển thị công thức Information Gain
    st.latex(r'\text{Gain}(S, A) = H(S) - \sum \frac{|S_v|}{|S|} H(S_v)')

    st.markdown("""
        Trong đó:
        - $H(S)$: Entropy của tập dữ liệu ban đầu.
        - $|S_v|$: Số lượng mẫu trong tập con sau phân chia theo đặc trưng $A$.
        - $|S|$: Tổng số mẫu ban đầu.
        - $H(S_v)$: Entropy của tập con sau phân chia.
    """)
    st.markdown("""
    <p style="font-size:22px;">
        💡 <b>Cách chọn ngưỡng:</b><br>
        1. Tính Entropy ban đầu của tập dữ liệu.<br>
        2. Thử các ngưỡng trên từng đặc trưng, tính Entropy sau phân chia và Information Gain.<br>
        3. Chọn đặc trưng và ngưỡng có Information Gain cao nhất để phân chia.<br>
        4. Lặp lại cho đến khi đạt độ sâu tối đa hoặc Entropy = 0.<br>
        📊 <b>Ví dụ từ tài liệu:</b> Phân loại khách hàng mua máy tính với 14 mẫu (9 mua, 5 không mua).<br>
        </p>
    """, unsafe_allow_html=True)

    # Hiển thị ví dụ Entropy ban đầu
    st.latex(r'H(S) = -\frac{9}{14} \log_2 \frac{9}{14} - \frac{5}{14} \log_2 \frac{5}{14} \approx 0.94')

    st.markdown("""
    <p style="font-size:22px;">
    Phân chia theo đặc trưng "Sinh viên":<br>
    - Tập con "Không" (5 mua, 2 không):<br>
    </p>
    """, unsafe_allow_html=True)

    # Hiển thị Entropy tập con "Không"
    st.latex(r'H(S_v) = -\frac{5}{7} \log_2 \frac{5}{7} - \frac{2}{7} \log_2 \frac{2}{7} \approx 0.863')

    st.markdown("""
    <p style="font-size:22px;">
    - Tập con "Có" (4 không mua):<br>
    </p>
    """, unsafe_allow_html=True)

    # Hiển thị Entropy tập con "Có"
    st.latex(r'H(S_v) = 0')

    st.markdown("""
    <p style="font-size:22px;">
    - Gain <br>
    </p>
    """, unsafe_allow_html=True)

    # Hiển thị Gain
    st.latex(r'= 0.94 - \left(\frac{7}{14} \times 0.863 + \frac{7}{14} \times 0\right) \approx 0.151')

    st.markdown("""
    <p style="font-size:22px;">
    Chọn "Sinh viên" để phân chia.<br>
    </p>
    """, unsafe_allow_html=True)

    # Phần 1.5: Ví dụ cố định với hình động (chỉ hiển thị khi chọn dữ liệu giả lập)
    if data_option == "Dữ liệu giả lập":
        st.markdown('<h2 style="font-size:32px; color:#FFD700;">🎥 Ví dụ cố định: Quá trình phân lớp với hình động</h2>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:22px;">📈 Xem quá trình Decision Tree phân chia dữ liệu qua 2 bước:</p>', unsafe_allow_html=True)

        # Dữ liệu cố định cho ví dụ
        np.random.seed(42)
        X_fixed = X
        y_fixed = y

        # Tạo lưới để dự đoán
        x1_range = np.linspace(min(X_fixed[:, 0]) - 0.1, max(X_fixed[:, 0]) + 0.1, 100)
        x2_range = np.linspace(min(X_fixed[:, 1]) - 0.1, max(X_fixed[:, 1]) + 0.1, 100)
        X_grid = np.array([[x1, x2] for x1 in x1_range for x2 in x2_range])

        # Tạo hình động
        fig, ax = plt.subplots()
        # Sử dụng màu sắc tùy chỉnh: 0 là đỏ, 1 là xanh
        colors = ['red' if label == 0 else 'blue' for label in y_fixed]
        scat = ax.scatter(X_fixed[:, 0], X_fixed[:, 1], c=colors, s=80)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title('Quá trình phân lớp của Decision Tree')

        def init():
            ax.clear()
            ax.scatter(X_fixed[:, 0], X_fixed[:, 1], c=colors, s=80)
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_title('Quá trình phân lớp của Decision Tree')
            ax.legend(handles=legend_elements, title="Nhãn", loc='upper right')
            return ax,

        def update(frame):
            ax.clear()
            ax.scatter(X_fixed[:, 0], X_fixed[:, 1], c=colors, s=80)
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_title(f'Bước phân lớp {frame + 1}')

            # Dự đoán vùng phân lớp cho từng bước
            if frame == 0:
                # Bước 1: Phân chia ngang theo X2 = 2.445
                y_pred_grid = np.zeros((100, 100))
                for i in range(100):
                    for j in range(100):
                        if x2_range[j] > 2.445:
                            y_pred_grid[j, i] = 1
                ax.axhline(y=2.445, color='black', linestyle='--', label='Bước 1: X2 = 2.445')
            elif frame == 1:
                # Bước 2: Phân chia dọc theo X1 = 1.477 trong vùng X2 > 2.445
                y_pred_grid = np.zeros((100, 100))
                for i in range(100):
                    for j in range(100):
                        if x2_range[j] > 2.445 and x1_range[i] > 1.477:
                            y_pred_grid[j, i] = 1
                        elif x2_range[j] <= 2.445:
                            y_pred_grid[j, i] = 0
                ax.axhline(y=2.445, color='black', linestyle='--', label='Bước 1: X2 = 2.445')
                ax.axvline(x=1.477, color='black', linestyle='--', label='Bước 2: X1 = 1.477')
                ax.legend()

            ax.contourf(x1_range, x2_range, y_pred_grid, alpha=0.4, cmap='Oranges', levels=[-0.5, 0.5, 1.5])  # Sử dụng màu cam
            ax.legend(handles=legend_elements, title="Nhãn", loc='upper right')
            return ax,

        # Tạo animation
        ani = FuncAnimation(fig, update, init_func=init, frames=2, interval=2000, repeat=False, blit=False)

        # Lưu animation thành GIF
        try:
            writer = PillowWriter(fps=0.5)
            with writer.saving(fig, "decision_tree_animation.gif", dpi=100):
                for i in range(2):  # Số khung hình
                    update(i)
                    writer.grab_frame()
        except Exception as e:
            st.error(f"Lỗi khi lưu GIF: {e}")
            return

        # Hiển thị GIF trong Streamlit
        with open("decision_tree_animation.gif", "rb") as file:
            gif_data = file.read()
        gif_base64 = base64.b64encode(gif_data).decode('utf-8')
        st.markdown(f'<img src="data:image/gif;base64,{gif_base64}" alt="animation">', unsafe_allow_html=True)

    # Phần 2: Trực quan hóa ý tưởng
    st.markdown('<h2 style="font-size:32px; color:#FFD700;">📈 2. Cơ chế hoạt động của Decision Tree</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">👀 Quan sát cách Decision Tree phân chia dữ liệu qua từng bước:</p>', unsafe_allow_html=True)

    # Biểu đồ dữ liệu gốc
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=['red' if label == 0 else 'blue' for label in y], label='Dữ liệu thực tế', s=80)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Dữ liệu phân loại ban đầu')
    ax.legend(handles=legend_elements, title="Nhãn", loc='upper right')
    st.pyplot(fig)

    st.markdown('<p style="font-size:22px;">🔍 Decision Tree sẽ phân chia dữ liệu qua các bước dựa trên ngưỡng tối ưu.</p>', unsafe_allow_html=True)

    # Phần 3: Thực hành với Decision Tree
    st.markdown('<h2 style="font-size:32px; color:#00CED1;">🎮 3. Thực hành với Decision Tree</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">🛠️ Xây dựng mô hình Decision Tree, hiển thị từng bước phân chia, và dự đoán nhãn:</p>', unsafe_allow_html=True)

    # Tùy chọn độ sâu tối đa
    max_depth = st.slider("📏 Chọn độ sâu tối đa của cây (max_depth):", min_value=1, max_value=5, value=2)

    # Huấn luyện mô hình Decision Tree
    with st.spinner("🔄 Đang xây dựng mô hình..."):
        model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=42)
        model.fit(X, y)

        # Lấy thông tin phân chia từ cây
        n_nodes = model.tree_.node_count
        children_left = model.tree_.children_left
        children_right = model.tree_.children_right
        feature = model.tree_.feature
        threshold = model.tree_.threshold
        samples = model.tree_.n_node_samples
        values = model.tree_.value

        # Tạo danh sách các bước phân chia
        splits = []
        def traverse_tree(node_id=0, depth=0):
            if node_id != -1 and depth < max_depth:
                entropy_value = model.tree_.impurity[node_id]
                if children_left[node_id] != -1:
                    splits.append((depth, feature[node_id], threshold[node_id], entropy_value, samples[node_id], values[node_id]))
                    traverse_tree(children_left[node_id], depth + 1)
                    traverse_tree(children_right[node_id], depth + 1)
        traverse_tree()

        # Dự đoán trên toàn bộ không gian
        x1_range = np.linspace(min(X[:, 0]) - 0.1, max(X[:, 0]) + 0.1, 100)
        x2_range = np.linspace(min(X[:, 1]) - 0.1, max(X[:, 1]) + 0.1, 100)
        X_grid = np.array([[x1, x2] for x1 in x1_range for x2 in x2_range])
        y_pred_grid = model.predict(X_grid).reshape(100, 100)

        # Tính accuracy
        y_pred_full = model.predict(X)
        accuracy = accuracy_score(y, y_pred_full)

        # Hiển thị từng bước phân chia
        st.markdown('<h3 style="font-size:26px; color:#4682B4;">🔍 Các bước phân chia:</h3>', unsafe_allow_html=True)
        current_step = 0

        # Biểu đồ dữ liệu ban đầu trước khi phân chia (thêm màu nền cam)
        fig, ax = plt.subplots()
        y_pred_grid_step = np.zeros((100, 100))  # Ban đầu tất cả là lớp 0
        ax.contourf(x1_range, x2_range, y_pred_grid_step, alpha=0.4, cmap='Oranges', levels=[-0.5, 0.5, 1.5])  # Màu nền cam
        ax.scatter(X[:, 0], X[:, 1], c=['red' if label == 0 else 'blue' for label in y], edgecolors='k', s=80)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title('Dữ liệu ban đầu trước khi phân chia')
        ax.legend(handles=legend_elements, title="Nhãn", loc='upper right')
        st.pyplot(fig)

        # Tạo vùng phân lớp tại mỗi bước
        for i, (depth, feat_idx, thresh, entropy_value, sample_count, value) in enumerate(splits):
            current_step += 1

            # Tính toán vùng phân lớp tại bước hiện tại
            y_pred_grid_step = np.zeros((100, 100))  # Khởi tạo lại lưới dự đoán
            for idx in range(len(X_grid)):
                x1, x2 = X_grid[idx]
                # Khởi tạo nhãn mặc định là 0 (lớp đa số)
                label = 0

                # Áp dụng các phân chia từ bước 1 đến bước hiện tại
                for j in range(i + 1):
                    past_depth, past_feat_idx, past_thresh, _, _, past_value = splits[j]
                    if past_feat_idx == 0:  # Phân chia theo X1
                        if j == 0:  # Bước 1
                            if x1 <= past_thresh:
                                label = np.argmax(past_value[0]) if past_value[0][0] > past_value[0][1] else 0
                            else:
                                label = np.argmax(past_value[0]) if past_value[0][0] > past_value[0][1] else 1
                        else:  # Bước tiếp theo
                            if label == 1:  # Chỉ áp dụng cho nhánh bên phải của bước trước
                                if x1 <= past_thresh:
                                    label = np.argmax(past_value[0]) if past_value[0][0] > past_value[0][1] else 0
                                else:
                                    label = np.argmax(past_value[0]) if past_value[0][0] > past_value[0][1] else 1
                    else:  # Phân chia theo X2
                        if j == 0:  # Bước 1
                            if x2 <= past_thresh:
                                label = np.argmax(past_value[0]) if past_value[0][0] > past_value[0][1] else 0
                            else:
                                label = np.argmax(past_value[0]) if past_value[0][0] > past_value[0][1] else 1
                        else:  # Bước tiếp theo
                            if label == 1:  # Chỉ áp dụng cho nhánh bên trên của bước trước
                                if x2 <= past_thresh:
                                    label = np.argmax(past_value[0]) if past_value[0][0] > past_value[0][1] else 0
                                else:
                                    label = np.argmax(past_value[0]) if past_value[0][0] > past_value[0][1] else 1

                # Gán nhãn cho lưới
                row = idx // 100
                col = idx % 100
                y_pred_grid_step[row, col] = label

            # Vẽ biểu đồ cho bước hiện tại (thêm màu nền cam)
            fig, ax = plt.subplots()
            ax.contourf(x1_range, x2_range, y_pred_grid_step, alpha=0.4, cmap='Oranges', levels=[-0.5, 0.5, 1.5])  # Màu nền cam
            ax.scatter(X[:, 0], X[:, 1], c=['red' if label == 0 else 'blue' for label in y], edgecolors='k', s=80)
            # Vẽ các đường phân chia từ các bước trước
            for j in range(i + 1):
                past_depth, past_feat_idx, past_thresh, _, _, _ = splits[j]
                if past_feat_idx == 0:  # Phân chia theo X1
                    ax.axvline(x=past_thresh, color='black', linestyle='--', alpha=0.5)
                else:  # Phân chia theo X2
                    ax.axhline(y=past_thresh, color='black', linestyle='--', alpha=0.5)
            if feat_idx == 0:  # Phân chia theo X1
                ax.axvline(x=thresh, color='black', linestyle='--', label=f'Bước {current_step}: X1 = {thresh:.3f}')
            else:  # Phân chia theo X2
                ax.axhline(y=thresh, color='black', linestyle='--', label=f'Bước {current_step}: X2 = {thresh:.3f}')
            class_label = np.argmax(value[0])
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_title(f'Bước phân chia {current_step} (max_depth={max_depth})')
            ax.legend(handles=legend_elements + [Line2D([0], [0], color='black', linestyle='--', label=f'Bước {current_step}: {"X1" if feat_idx == 0 else "X2"} = {thresh:.3f}')], title="Nhãn", loc='upper right')
            st.pyplot(fig)
            st.markdown(f'<p style="font-size:18px;">📋 Thông tin bước {current_step}: Entropy = {entropy_value:.3f}, Samples = {sample_count}, Value = {value[0].astype(int)}, Class = {class_label}</p>', unsafe_allow_html=True)

        # Trực quan hóa cây quyết định cuối cùng
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(model, feature_names=['X1', 'X2'], class_names=['0', '1'], filled=True, impurity=True, ax=ax)
        st.pyplot(fig)

    st.markdown(f'<p style="font-size:24px; color:#FF6347;">✅ Kết quả: Mô hình Decision Tree (max_depth={max_depth}) đã được xây dựng.</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:20px; color:#4682B4;">📊 <b>Độ chính xác (Accuracy):</b> {accuracy:.2f}</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:18px;">💡 Accuracy đo tỷ lệ dự đoán đúng trên dữ liệu đã cho. Giá trị gần 1 cho thấy mô hình tốt.</p>', unsafe_allow_html=True)

    # Phần dự đoán giá trị cụ thể
    st.markdown('<h3 style="font-size:26px; color:#4682B4;">🔍 Dự đoán nhãn Y từ X1, X2</h3>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:20px;">Nhập giá trị X1 ({min(X[:, 0]):.2f} đến {max(X[:, 0]):.2f}) và X2 ({min(X[:, 1]):.2f} đến {max(X[:, 1]):.2f}):</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        input_x1 = st.number_input("X1:", min_value=float(min(X[:, 0])), max_value=float(max(X[:, 0])), value=float(X[len(X)//2, 0]), step=0.1)
    with col2:
        input_x2 = st.number_input("X2:", min_value=float(min(X[:, 1])), max_value=float(max(X[:, 1])), value=float(X[len(X)//2, 1]), step=0.1)

    # Dự đoán nhãn từ X1, X2 nhập vào
    X_input = np.array([[input_x1, input_x2]])
    y_pred_input = model.predict(X_input)[0]

    # Tìm nhãn thực tế gần nhất trong dữ liệu mẫu để so sánh
    idx_closest = np.argmin(np.sqrt((X[:, 0] - input_x1)**2 + (X[:, 1] - input_x2)**2))
    y_true_closest = y[idx_closest]

    # Vẽ điểm dự đoán lên biểu đồ cuối cùng (thêm màu nền cam)
    fig, ax = plt.subplots()
    ax.contourf(x1_range, x2_range, y_pred_grid, alpha=0.4, cmap='Oranges', levels=[-0.5, 0.5, 1.5])  # Màu nền cam
    ax.scatter(X[:, 0], X[:, 1], c=['red' if label == 0 else 'blue' for label in y], edgecolors='k', s=80)
    ax.scatter([input_x1], [input_x2], color='green', s=150, marker='*', label='Điểm dự đoán')
    for _, feat_idx, thresh, _, _, _ in splits:
        if feat_idx == 0:
            ax.axvline(x=thresh, color='black', linestyle='--', alpha=0.5)
        else:
            ax.axhline(y=thresh, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title(f'Vùng phân loại với điểm dự đoán (max_depth={max_depth})')
    ax.legend(handles=legend_elements + [Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=10, label='Điểm dự đoán')], title="Nhãn", loc='upper right')
    st.pyplot(fig)

    # Hiển thị thông tin dự đoán bên ngoài biểu đồ
    st.markdown(f'<p style="font-size:20px;">📈 <b>Thông tin dự đoán:</b> Điểm (X1={input_x1:.2f}, X2={input_x2:.2f}), Nhãn dự đoán Y = {y_pred_input} ({"Đỏ" if y_pred_input == 0 else "Xanh"})</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:20px;">🔎 <b>Nhãn thực tế gần nhất (X1={X[idx_closest, 0]:.2f}, X2={X[idx_closest, 1]:.2f}):</b> {y_true_closest} ({"Đỏ" if y_true_closest == 0 else "Xanh"})</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:20px; color:#FF4500;">⚠️ <b>Kết quả so sánh:</b> {"Khớp" if y_pred_input == y_true_closest else "Không khớp"}</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:18px;">💡 Kết quả so sánh cho thấy khả năng phân loại của mô hình tại điểm cụ thể.</p>', unsafe_allow_html=True)

    # Phần 4: Liên hệ với hạn chế
    st.markdown('<h2 style="font-size:32px; color:#FFA500;">⚠️ 4. Decision Tree và các hạn chế</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:22px;">
    👍 <b>Ưu điểm:</b><br>
    - 🌟 Trực quan, dễ hiểu qua cấu trúc cây.<br>
    - 📈 Không cần chuẩn hóa dữ liệu, linh hoạt với nhiều loại đặc trưng.<br>
    👎 <b>Nhược điểm:</b><br>
    - 🚨 Dễ bị Overfitting nếu cây quá sâu hoặc dữ liệu nhiễu.<br>
    - ⚙️ Nhạy cảm với thay đổi nhỏ trong dữ liệu (thiếu ổn định).<br>
    💡 <b>Gợi ý:</b> Điều chỉnh độ sâu tối đa (max_depth) để cân bằng giữa Underfitting và Overfitting.
    </p>
    """, unsafe_allow_html=True)

    # Phần 5: Tài liệu tham khảo
    st.markdown('<h2 style="font-size:32px; color:#1E90FF;">🔗 5. Tài liệu tham khảo</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">📖 Xem chi tiết về Decision Tree tại <a href="https://machinelearningcoban.com/tabml_book/ch_model/decision_tree.html?form=MG0AV3">Machine Learning Cơ Bản</a>.</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">🙏 Cảm ơn bạn đã tham gia khám phá Decision Tree!</p>', unsafe_allow_html=True)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Hàm hiển thị thông tin về MNIST
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
    image_path = os.path.join(os.path.dirname(__file__), "img3.png")
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

    # Bảng kết quả với giao diện đẹp
    st.markdown("<h2 style='color: #8A2BE2; font-size: 32px;'>🏆 Kết Quả Hiệu Suất Trên MNIST</h2>", unsafe_allow_html=True)
    data = {
        "Mô hình": ["Decision Tree", "SVM (Linear)", "SVM (Poly)", "SVM (Sigmoid)", "SVM (RBF)"],
        "Độ chính xác": ["0.8574", "0.9253", "0.9774", "0.7656", "0.9823"]
    }
    df = pd.DataFrame(data)
    st.table(df.style.set_properties(**{
        'background-color': '#F5F5F5',
        'border-color': '#DDDDDD',
        'border-style': 'solid',
        'border-width': '1px',
        'text-align': 'center',
        'font-size': '18px'
    }).set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#3498DB'), ('color', 'white'), ('font-weight', 'bold')]}
    ]))
import os
def load_mnist_data():
    base_dir = os.path.dirname(__file__)  # Lấy thư mục chứa file MNIST.py
    file_path_X = os.path.join(base_dir, "X.npy")
    file_path_y = os.path.join(base_dir, "y.npy")

    if not os.path.exists(file_path_X) or not os.path.exists(file_path_y):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {file_path_X} hoặc {file_path_y}")

    X = np.load(file_path_X)
    y = np.load(file_path_y)
    
    return X, y

# Hàm chia dữ liệu
def split_data():
    st.title("📌 Chia dữ liệu Train/Test")

    # Đọc dữ liệu
    X, y = load_mnist_data()
    total_samples = X.shape[0]
    num_classes = len(np.unique(y))  # Số lớp (10 trong trường hợp MNIST)

    # Nếu chưa có cờ "data_split_done", đặt mặc định là False
    if "data_split_done" not in st.session_state:
        st.session_state.data_split_done = False  

    # Thanh kéo chọn số lượng ảnh để train
    max_samples = total_samples - num_classes
    num_samples = st.slider("📌 Chọn số lượng ảnh để train:", 1000, max_samples, 10000)

    # Thanh kéo chọn tỷ lệ Train/Test
    test_size_percent = st.slider("📌 Chọn % dữ liệu Test", 10, 80, 10)
    test_size = test_size_percent / 100  # Tỷ lệ test hợp lệ (0.1 đến 0.8)
    remaining_size = 100 - test_size_percent
    val_size_percent = st.slider("📌 Chọn % dữ liệu Validation (trong phần Train)", 0, min(80, remaining_size), 0)
    val_size = val_size_percent / 100  # Tỷ lệ validation hợp lệ (0.0 đến 0.8)

    # Tính số lượng mẫu trong tập test và validation
    test_samples = int(num_samples * test_size)
    train_val_samples = num_samples - test_samples
    val_samples = int(train_val_samples * (val_size_percent / remaining_size)) if val_size_percent > 0 else 0

    # Kiểm tra số lượng mẫu tối thiểu
    if test_samples < num_classes:
        st.error(f"❌ Số lượng mẫu trong tập Test ({test_samples}) phải lớn hơn hoặc bằng số lớp ({num_classes}). Vui lòng giảm % Test hoặc tăng số lượng ảnh.")
        return
    if val_samples < num_classes and val_size_percent > 0:
        st.error(f"❌ Số lượng mẫu trong tập Validation ({val_samples}) phải lớn hơn hoặc bằng số lớp ({num_classes}). Vui lòng giảm % Validation hoặc tăng số lượng ảnh.")
        return

    # Cảnh báo nếu tập train quá nhỏ
    train_percent = remaining_size - val_size_percent
    if train_percent < 30:
        st.warning(f"⚠️ Tỷ lệ Train chỉ còn {train_percent}%! Điều này có thể ảnh hưởng đến hiệu suất mô hình. Hãy cân nhắc giảm % Test hoặc Validation.")

    st.write(f"📌 **Tỷ lệ phân chia:** Test={test_size_percent}%, Validation={val_size_percent}%, Train={train_percent}%")

    # Nút reset để cho phép chia lại
    if st.session_state.data_split_done:
        if st.button("🔄 Reset & Chia lại"):
            st.session_state.data_split_done = False
            st.rerun()

    if st.button("✅ Xác nhận & Lưu"):
        st.session_state.data_split_done = True
        
        # Chia dữ liệu theo số lượng mẫu đã chọn
        X_selected, _, y_selected, _ = train_test_split(
            X, y, train_size=num_samples, stratify=y, random_state=42
        )

        # Chia train/test
        stratify_option = y_selected if test_samples >= num_classes else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_selected, y_selected, test_size=test_size, stratify=stratify_option, random_state=42
        )

        # Chia train/val (nếu val_size > 0)
        if val_size_percent > 0:
            stratify_option = y_train_full if val_samples >= num_classes else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=val_size, stratify=stratify_option, random_state=42
            )
        else:
            X_train, X_val, y_train, y_val = X_train_full, np.array([]), y_train_full, np.array([])

        # Chuẩn hóa dữ liệu tại đây
        X_train = X_train.reshape(-1, 28 * 28) / 255.0
        X_test = X_test.reshape(-1, 28 * 28) / 255.0
        X_val = X_val.reshape(-1, 28 * 28) / 255.0 if val_size_percent > 0 else X_val

        # Lưu dữ liệu vào session_state
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

        # Hiển thị thông tin chia dữ liệu
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
        
import streamlit as st
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import time

# Hàm huấn luyện mô hình
def train():
    # 📥 **Tải dữ liệu MNIST**
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

    st.header("⚙️ Chọn mô hình & Huấn luyện")

    # 📌 **Chọn mô hình**
    model_choice = st.selectbox("Chọn mô hình:", ["Decision Tree", "SVM"])

    if model_choice == "Decision Tree":
        st.markdown("""
        - **🌳 Decision Tree (Cây quyết định)** giúp chia dữ liệu thành các nhóm bằng cách đặt câu hỏi nhị phân dựa trên đặc trưng.
        - **Tham số cần chọn:**  
            - **max_depth**: Giới hạn độ sâu tối đa của cây.  
        """)
        max_depth = st.slider("max_depth", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

    elif model_choice == "SVM":
        st.markdown("""
        - **🛠️ SVM (Support Vector Machine)** là mô hình tìm siêu phẳng tốt nhất để phân tách dữ liệu.
        - **Lưu ý:** Kernel 'linear' thường nhanh hơn 'rbf', 'poly', 'sigmoid' với dữ liệu lớn.
        """)
        C = st.slider("C (Regularization)", 0.1, 10.0, 1.0, step=0.1)
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
        model = SVC(C=C, kernel=kernel, random_state=42)

    n_folds = st.slider("Chọn số folds (KFold Cross-Validation):", min_value=2, max_value=10, value=5)

    # Chỉ nhập tên Experiment (Không có phần nhập tên Run)
    if "experiment_name" not in st.session_state:
        st.session_state["experiment_name"] = "My_Experiment"

    experiment_name = st.text_input("🔹 Nhập tên Experiment:", st.session_state["experiment_name"], key="experiment_name_input")    

    if experiment_name:
        st.session_state["experiment_name"] = experiment_name

    mlflow.set_experiment(experiment_name)
    st.write(f"✅ Experiment Name: {experiment_name}")

    
    if st.button("Huấn luyện mô hình"):
        # Thanh tiến trình
        progress_bar = st.progress(0)
        status_text = st.empty()
        if "run_name" not in st.session_state:
            st.session_state["run_name"] = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # Đặt tên dựa vào thời gian

        with mlflow.start_run(run_name=f"Train_{st.session_state['run_name']}"):
            try:
                # Cập nhật tiến trình
                status_text.text("🔄 Ghi log tham số vào MLflow...")
                progress_bar.progress(10)

                # Log các tham số liên quan đến dữ liệu
                mlflow.log_param("test_size", st.session_state.get("test_size", 0))
                mlflow.log_param("val_size", st.session_state.get("val_size", 0))
                mlflow.log_param("train_size", st.session_state.get("train_size", 0))
                mlflow.log_param("num_samples", st.session_state.get("total_samples", 0))

                # 🏆 **Huấn luyện với Cross Validation**
                status_text.text("⏳ Đang chạy Cross-Validation...")
                progress_bar.progress(40)
                cv_scores = cross_val_score(model, X_train, y_train, cv=n_folds, n_jobs=-1)  # Song song hóa
                mean_cv_score = cv_scores.mean()
                std_cv_score = cv_scores.std()

                st.success(f"📊 **Cross-Validation Accuracy**: {mean_cv_score:.4f} ± {std_cv_score:.4f}")

                # Huấn luyện mô hình trên tập train chính
                status_text.text("⏳ Đang huấn luyện mô hình...")
                progress_bar.progress(70)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                st.success(f"✅ Độ chính xác trên test set: {acc:.4f}")

                # 📝 Ghi log vào MLflow
                status_text.text("🔄 Ghi log kết quả vào MLflow...")
                progress_bar.progress(90)
                mlflow.log_param("model", model_choice)
                if model_choice == "Decision Tree":
                    mlflow.log_param("max_depth", max_depth)
                elif model_choice == "SVM":
                    mlflow.log_param("C", C)
                    mlflow.log_param("kernel", kernel)

                mlflow.log_metric("test_accuracy", acc)
                mlflow.log_metric("cv_accuracy_mean", mean_cv_score)
                mlflow.log_metric("cv_accuracy_std", std_cv_score)
                mlflow.sklearn.log_model(model, f"model_{model_choice.lower().replace(' ', '_')}")

                # Lưu mô hình vào session_state với dictionary
                if "models" not in st.session_state:
                    st.session_state["models"] = {}

                model_name = f"{model_choice.lower().replace(' ', '_')}_{kernel if model_choice == 'SVM' else max_depth}"
                count = 1
                base_model_name = model_name
                while model_name in st.session_state["models"]:
                    model_name = f"{base_model_name}_{count}"
                    count += 1

                st.session_state["models"][model_name] = model
                st.write(f"🔹 Mô hình đã được lưu với tên: {model_name}")
                st.write(f"Tổng số mô hình hiện tại: {len(st.session_state['models'])}")
                st.session_state["last_trained_model"] = model_name

                # Hiển thị danh sách mô hình
                st.write("📋 Danh sách các mô hình đã lưu:")
                model_names = list(st.session_state["models"].keys())
                st.write(", ".join(model_names))

                status_text.text("✅ Hoàn tất huấn luyện!")
                progress_bar.progress(100)
                st.success(f"✅ Đã log dữ liệu cho **Train_{st.session_state['run_name']}**!")
                if "mlflow_url" in st.session_state:
                    st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")
                else:
                    st.warning("⚠️ URL MLflow chưa được thiết lập. Vui lòng kiểm tra cấu hình MLflow.")

            except Exception as e:
                st.error(f"❌ Lỗi khi huấn luyện: {str(e)}")
                mlflow.end_run()
                progress_bar.progress(0)
                status_text.text("❌ Huấn luyện thất bại!")

def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"⚠️ Không tìm thấy mô hình tại `{path}`")
        st.stop()
# ✅ Xử lý ảnh từ canvas (chuẩn 28x28 cho MNIST)
def preprocess_canvas_image(canvas_result):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data[:, :, 0].astype(np.uint8))
        img = img.resize((28, 28)).convert("L")  # Resize và chuyển thành grayscale
        img = np.array(img, dtype=np.float32) / 255.0  # Chuẩn hóa về [0, 1]
        return img.reshape(1, -1)  # Chuyển thành vector 1D
    return None

def du_doan():
    st.header("✍️ Vẽ số để dự đoán")

    # 🔹 Danh sách mô hình có sẵn
    models = {
        "SVM Linear": "svm_mnist_linear.joblib",
        "SVM Poly": "svm_mnist_poly.joblib",
        "SVM Sigmoid": "svm_mnist_sigmoid.joblib",
        "SVM RBF": "svm_mnist_rbf.joblib",
    }

    # Lấy tên mô hình từ session_state
    model_names = list(st.session_state.get("models", {}).keys())

    # 📌 Chọn mô hình
    if model_names:
        model_option = st.selectbox("🔍 Chọn mô hình:", model_names)
    else:
        st.warning("⚠️ Chưa có mô hình nào được huấn luyện.")
        return


    # Nếu chọn mô hình đã được huấn luyện và lưu trong session_state
    if model_option in model_names:
        model = st.session_state["models"][model_option]  # Truy cập trực tiếp từ dictionary
        st.success(f"✅ Đã chọn mô hình từ session_state: {model_option}")
    else:
        # Nếu chọn mô hình có sẵn (các mô hình đã được huấn luyện và lưu trữ dưới dạng file)
        try:
            model = load_model(models[model_option])
            st.success(f"✅ Đã tải mô hình từ file: {model_option}")
        except Exception as e:
            st.error(f"❌ Lỗi khi tải mô hình {model_option}: {str(e)}")
            return

    # 🆕 Cập nhật key cho canvas khi nhấn "Tải lại"
    if "key_value" not in st.session_state:
        st.session_state.key_value = str(random.randint(0, 1000000))  # Đổi key thành string

    if st.button("🔄 Tải lại nếu không thấy canvas"):
        st.session_state.key_value = str(random.randint(0, 1000000))  # Đổi key thành string
        st.rerun()  # Cập nhật lại giao diện để vùng vẽ được làm mới
    
    # ✍️ Vẽ số
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=150,
        width=150,
        drawing_mode="freedraw",
        key=st.session_state.key_value,  # Đảm bảo key là string
        update_streamlit=True
    )

    if st.button("Dự đoán số"):
        img = preprocess_canvas_image(canvas_result)

        if img is not None:
            # Hiển thị ảnh sau xử lý
            st.image(Image.fromarray((img.reshape(28, 28) * 255).astype(np.uint8)), caption="Ảnh sau xử lý", width=100)

            # Dự đoán
            prediction = model.predict(img)
            st.subheader(f"🔢 Dự đoán: {prediction[0]}")
        else:
            st.error("⚠️ Hãy vẽ một số trước khi bấm Dự đoán!")

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


def Classification():
    """#+
    Main function for the MNIST Classification App.#+

    This function sets up the MLflow tracking, initializes the Streamlit interface,#+
    and creates tabs for different sections of the application including theory,#+
    data visualization, model training, prediction, and MLflow experiment tracking.#+
#+
    The function performs the following tasks:#+
    1. Initializes MLflow tracking if not already done.#+
    2. Sets up the Streamlit interface with custom CSS.#+
    3. Creates tabs for different sections of the app.#+
    4. Calls appropriate functions for each tab.#+
#+
    Parameters:#+
    None#+
#+
    Returns:#+
    None#+
#+
    Note:#+
    This function relies on several global variables and functions that should be#+
    defined elsewhere in the code, such as ly_thuyet_Decision(), ly_thuyet_svm(),#+
    data(), split_data(), train(), du_doan(), and show_experiment_selector().#+
    """#+
    if "mlflow_initialized" not in st.session_state:   
        DAGSHUB_MLFLOW_URI = "https://dagshub.com/PTToan250303/Linear_replication.mlflow"
        mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
        st.session_state['mlflow_url']=DAGSHUB_MLFLOW_URI

        st.session_state['mlflow_url']=DAGSHUB_MLFLOW_URI
        os.environ["MLFLOW_TRACKING_USERNAME"] = "PTToan250303"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "5ca8caf353d564c358852da97c7487e64fc30a73"
        mlflow.set_experiment("Classification")   
    st.markdown("""
        <style>
        .title {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            color: #4682B4;
            margin-top: 50px;
        }
        .subtitle {
            font-size: 24px;
            text-align: center;
            color: #4A4A4A;
        }
        hr {
            border: 1px solid #ddd;
        }
        </style>
        <div class="title">MNIST Classification App</div>
        <hr>
    """, unsafe_allow_html=True)    

    #st.session_state.clear()#-
    ### **Phần 1: Hiển thị dữ liệu MNIST**#-
#-
    ### **Phần 2: Trình bày lý thuyết về Decision Tree & SVM*#-
#-
    # 1️⃣ Phần giới thiệu#-
#-
    # === Sidebar để chọn trang ==#-
    # === Tạo Tabs ===#-
    tab1, tab2, tab3, tab4,tab5 ,tab6= st.tabs(["📘 Lý thuyết Decision Tree", "📘 Lý thuyết SVM", "📘 Data" ,"⚙️ Huấn luyện", "🔢 Dự đoán","🔥Mlflow"])

    with tab1:
#-
        ly_thuyet_Decision()
    with tab2:
        ly_thuyet_svm()
    with tab3:
        data()
#-
    with tab4:  
        split_data()
        train()
    with tab5: 
        du_doan()   
    with tab6:
#-
        show_experiment_selector()



            
if __name__ == "__main__":
    Classification()