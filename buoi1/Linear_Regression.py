import streamlit as st
import pandas as pd
from scipy.stats import zscore
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import  StandardScaler,PolynomialFeatures,OneHotEncoder, MinMaxScaler
from scipy.stats import zscore
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error
import mlflow
import io
from sklearn.model_selection import KFold,cross_val_score
import time

import os
from mlflow.tracking import MlflowClient
from scipy.stats import zscore
def run_polynomial_regression_app():
    # Tiêu đề chính với màu sắc
    st.markdown('<h1 style="color:#FF4500; text-align:center;">🌟 Polynomial Regression🌟</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:24px; color:#4682B4;">📝 Tìm hiểu cách mở rộng Linear Regression để mô hình hóa dữ liệu phi tuyến một cách hiệu quả.</p>', unsafe_allow_html=True)

    # Chọn nguồn dữ liệu
    st.markdown('<h2 style="font-size:28px; color:#32CD32;">📊 Chọn nguồn dữ liệu</h2>', unsafe_allow_html=True)
    data_option = st.radio("Chọn loại dữ liệu:", ["Dữ liệu giả lập", "Dữ liệu tùy chỉnh"], key="data_option_selection")

    if data_option == "Dữ liệu giả lập":
        # Dữ liệu mẫu (phi tuyến)
        np.random.seed(42)
        X = np.linspace(0, 4, 20).reshape(-1, 1)
        y = 1 + 2 * X + 3 * X**2 + np.random.normal(0, 2, (20, 1))
    else:
        # Dữ liệu tùy chỉnh với giao diện đơn giản
        st.markdown('<p style="font-size:20px;">Thêm các cặp giá trị X và Y:</p>', unsafe_allow_html=True)

        # Khởi tạo session state để lưu dữ liệu
        if 'custom_data' not in st.session_state:
            st.session_state.custom_data = {'X': [], 'Y': []}

        # Hai cột để nhập X và Y
        col1, col2 = st.columns(2)
        with col1:
            x_input = st.number_input("Giá trị X:", value=0.0, step=0.1, key="x_input")
        with col2:
            y_input = st.number_input("Giá trị Y:", value=0.0, step=0.1, key="y_input")

        # Nút thêm điểm
        if st.button("➕ Thêm điểm"):
            st.session_state.custom_data['X'].append(x_input)
            st.session_state.custom_data['Y'].append(y_input)

        # Hiển thị dữ liệu đã nhập dưới dạng bảng
        if st.session_state.custom_data['X']:
            df = pd.DataFrame(st.session_state.custom_data)
            st.markdown('<p style="font-size:18px;">Dữ liệu đã nhập:</p>', unsafe_allow_html=True)
            st.dataframe(df)

            # Tùy chọn xóa điểm
            delete_index = st.selectbox("Chọn điểm để xóa (nếu cần):", options=range(len(st.session_state.custom_data['X'])), format_func=lambda i: f"Điểm {i}: X={st.session_state.custom_data['X'][i]}, Y={st.session_state.custom_data['Y'][i]}")
            if st.button("🗑️ Xóa điểm"):
                st.session_state.custom_data['X'].pop(delete_index)
                st.session_state.custom_data['Y'].pop(delete_index)
                st.rerun()  # Thay st.experimental_rerun() bằng st.rerun()

            # Chuyển dữ liệu thành numpy array
            X = np.array(st.session_state.custom_data['X']).reshape(-1, 1)
            y = np.array(st.session_state.custom_data['Y']).reshape(-1, 1)

            if len(X) < 2:
                st.error("Vui lòng nhập ít nhất 2 cặp dữ liệu để mô hình hóa!")
                return
        else:
            st.warning("Chưa có dữ liệu nào được thêm. Hãy nhập ít nhất 2 cặp X, Y để tiếp tục!")
            return

    # Phần 1: Giới thiệu
    st.markdown('<h2 style="font-size:32px; color:#32CD32;">📚 1. Polynomial Regression là gì?</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:22px;">
    ❓ Linear Regression chỉ phù hợp với dữ liệu có mối quan hệ tuyến tính. Nhưng với dữ liệu phi tuyến thì sao?<br>
    🚀 Polynomial Regression là một giải pháp mở rộng, sử dụng các lũy thừa của biến đầu vào để mô tả các mối quan hệ phức tạp hơn.<br>
    📐 Công thức cơ bản:
    </p>
    <div style="text-align: center;">
        <p style="font-size:28px; color:#FF69B4;"><b>y = w₀ + w₁x + w₂x² + w₃x³ + ... + wₙxⁿ</b></p>
    </div>
    <p style="font-size:22px;">
    - <b>y</b>: Giá trị cần dự đoán (output).<br>
    - <b>x</b>: Biến độc lập (input).<br>
    - <b>w</b>: Hệ số của từng bậc (weight).<br>
    - <b>n</b>: Bậc của đa thức (degree).<br>
    💡 Phương pháp này cho phép mô hình hóa các đường cong, mở rộng khả năng của hồi quy tuyến tính truyền thống.
    </p>
    """, unsafe_allow_html=True)

    # Phần 2: Trực quan hóa ý tưởng
    st.markdown('<h2 style="font-size:32px; color:#FFD700;">📈 2. Cơ chế hoạt động của Polynomial Regression</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">👀 Hãy quan sát cách Polynomial Regression mô phỏng dữ liệu phi tuyến:</p>', unsafe_allow_html=True)

    # Biểu đồ dữ liệu gốc
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='red', label='Dữ liệu thực tế', s=80)
    ax.set_xlabel('X (Biến độc lập)')
    ax.set_ylabel('Y (Biến phụ thuộc)')
    ax.set_title('Dữ liệu phi tuyến')
    ax.legend()
    st.pyplot(fig)

    st.markdown('<p style="font-size:22px;">🔍 Polynomial Regression sẽ tìm một đường cong tối ưu để biểu diễn mối quan hệ này.</p>', unsafe_allow_html=True)

    # Phần 3: Tương tác nâng cao với dự đoán
    st.markdown('<h2 style="font-size:32px; color:#00CED1;">🎮 3. Thực hành với Polynomial Regression</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">🛠️ Điều chỉnh bậc đa thức, so sánh với Linear Regression, và dự đoán giá trị cụ thể:</p>', unsafe_allow_html=True)

    # Tùy chọn bậc đa thức
    degree = st.slider("📏 Chọn bậc đa thức (degree):", min_value=1, max_value=10, value=2)
    compare_linear = st.checkbox("🔄 Hiển thị so sánh với Linear Regression", value=False)

    # Tạo hiệu ứng "đang tính toán"
    with st.spinner("🔄 Đang xây dựng mô hình..."):
        time.sleep(0.5)  # Giả lập thời gian chờ

        # Tạo đặc trưng đa thức
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)

        # Huấn luyện mô hình Polynomial
        model_poly = LinearRegression()
        model_poly.fit(X_poly, y)

        # Dự đoán Polynomial
        X_smooth = np.linspace(min(X), max(X), 100).reshape(-1, 1)
        X_smooth_poly = poly.transform(X_smooth)
        y_pred_poly = model_poly.predict(X_smooth_poly)

        # Tính MSE
        y_pred_full = model_poly.predict(X_poly)
        mse = mean_squared_error(y, y_pred_full)

        # Tạo biểu đồ
        fig, ax = plt.subplots()
        ax.scatter(X, y, color='red', label='Dữ liệu thực tế', s=80)
        ax.plot(X_smooth, y_pred_poly, 'b-', label=f'Polynomial (bậc {degree})', linewidth=2)

        # So sánh với Linear Regression nếu bật
        if compare_linear:
            model_linear = LinearRegression()
            model_linear.fit(X, y)
            y_pred_linear = model_linear.predict(X_smooth)
            ax.plot(X_smooth, y_pred_linear, 'g--', label='Linear Regression', linewidth=2)

        ax.set_xlabel('X (Biến độc lập)')
        ax.set_ylabel('Y (Biến phụ thuộc)')
        ax.set_title(f'Mô hình Polynomial Regression (bậc {degree})')
        ax.legend()
        st.pyplot(fig)

    st.markdown(f'<p style="font-size:24px; color:#FF6347;">✅ Kết quả: Mô hình bậc {degree} đã được xây dựng.</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:20px; color:#4682B4;">📊 <b>Mean Squared Error (MSE):</b> {mse:.2f}</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:18px;">💡 MSE đo lường sai số trung bình bình phương giữa giá trị thực tế và dự đoán. MSE nhỏ hơn thường cho thấy mô hình tốt hơn.</p>', unsafe_allow_html=True)
    st.markdown(
    r"""
    ### 📌 Công thức tính MSE:
    """)
    st.latex(r"""
    MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
    """)
    st.markdown(
    """
    Trong đó:
    - $$y_i$$ là giá trị thực tế.
    - $$\hat{y}_i$$ là giá trị dự đoán.
    - $$n$$ là số lượng mẫu dữ liệu.
    
    Ý nghĩa:
    - MSE càng nhỏ → Mô hình dự đoán càng chính xác.
    - MSE = 0 → Mô hình hoàn hảo (hiếm khi xảy ra).
    - MSE lớn có thể cho thấy sự chênh lệch lớn giữa dự đoán và thực tế.
    """)
    # Phần dự đoán giá trị cụ thể
    st.markdown('<h3 style="font-size:26px; color:#4682B4;">🔍 Dự đoán giá trị Y từ X</h3>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:20px;">Nhập giá trị X (trong khoảng {min(X)[0]:.2f} đến {max(X)[0]:.2f}) để xem dự đoán và sai số:</p>', unsafe_allow_html=True)
    input_x = st.number_input("Nhập giá trị X:", min_value=float(min(X)), max_value=float(max(X)), value=float(X[len(X)//2]), step=0.1)

    # Dự đoán Y từ X nhập vào
    X_input = np.array([[input_x]])
    X_input_poly = poly.transform(X_input)
    y_pred_input = model_poly.predict(X_input_poly)[0][0]

    # Tìm giá trị thực tế gần nhất trong dữ liệu mẫu để tính sai số
    idx_closest = np.argmin(np.abs(X - input_x))
    y_true_closest = y[idx_closest][0]
    error = abs(y_pred_input - y_true_closest)

    # Vẽ điểm dự đoán lên biểu đồ
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='red', label='Dữ liệu thực tế', s=80)
    ax.plot(X_smooth, y_pred_poly, 'b-', label=f'Polynomial (bậc {degree})', linewidth=2)
    ax.scatter([input_x], [y_pred_input], color='green', s=150, marker='*', label=f'Dự đoán (X={input_x:.2f}, Y={y_pred_input:.2f})')
    ax.set_xlabel('X (Biến độc lập)')
    ax.set_ylabel('Y (Biến phụ thuộc)')
    ax.set_title(f'Mô hình với điểm dự đoán (bậc {degree})')
    ax.legend()
    st.pyplot(fig)

    st.markdown(f'<p style="font-size:20px;">📈 <b>Giá trị dự đoán Y:</b> {y_pred_input:.2f}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:20px;">🔎 <b>Giá trị thực tế gần nhất (X={X[idx_closest][0]:.2f}):</b> {y_true_closest:.2f}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:20px; color:#FF4500;">⚠️ <b>Sai số tuyệt đối:</b> {error:.2f}</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:18px;">💡 Sai số cho thấy mức độ chênh lệch giữa dự đoán và thực tế. Sai số lớn có thể do bậc đa thức chưa phù hợp.</p>', unsafe_allow_html=True)

    # Phần 5: Liên hệ với Overfitting
    st.markdown('<h2 style="font-size:32px; color:#FFA500;">⚠️ 5. Polynomial Regression và vấn đề Overfitting</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:22px;">
    👍 <b>Ưu điểm:</b><br>
    - 🌟 Linh hoạt hơn Linear Regression, phù hợp với dữ liệu phi tuyến.<br>
    - 📈 Có khả năng mô phỏng các mối quan hệ phức tạp.<br>
    👎 <b>Nhược điểm:</b><br>
    - 🚨 Khi bậc đa thức quá cao, mô hình có nguy cơ Overfitting, tức là quá khớp với dữ liệu huấn luyện.<br>
    - ⚙️ Việc chọn bậc không phù hợp có thể làm giảm hiệu suất của mô hình.<br>
    💡 <b>Gợi ý:</b> Thử nghiệm với các bậc khác nhau để hiểu tác động của chúng.
    </p>
    """, unsafe_allow_html=True)

    # Phần 6: Kết thúc
    st.markdown('<h2 style="font-size:32px; color:#1E90FF;">🔗 6. Tài liệu tham khảo</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">📖 Tìm hiểu thêm về Overfitting và Polynomial Regression tại <a href="https://machinelearningcoban.com/2017/03/04/overfitting/">Machine Learning Cơ Bản</a>.</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">🙏 Cảm ơn bạn đã tham gia khám phá Polynomial Regression!</p>', unsafe_allow_html=True)

def run_linear_regression_app():
    # Tiêu đề chính
    st.markdown('<h1 style="color:#FF4500; text-align:center;">🌟 Linear Regression🌟</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:24px; color:#4682B4;">📝 Tìm hiểu cách sử dụng Linear Regression để mô hình hóa mối quan hệ tuyến tính giữa các biến.</p>', unsafe_allow_html=True)

    # Chọn nguồn dữ liệu
    st.markdown('<h2 style="font-size:28px; color:#32CD32;">📊 Chọn nguồn dữ liệu</h2>', unsafe_allow_html=True)
    data_option = st.radio("Chọn loại dữ liệu:", ("Dữ liệu giả lập", "Dữ liệu tùy chỉnh"))

    if data_option == "Dữ liệu giả lập":
        # Dữ liệu mẫu (tuyến tính)
        np.random.seed(42)
        X = np.linspace(0, 10, 20).reshape(-1, 1)
        y = 2 * X + 1 + np.random.normal(0, 2, (20, 1))  # y = 2x + 1 + nhiễu
    else:
        # Dữ liệu tùy chỉnh với giao diện đơn giản
        st.markdown('<p style="font-size:20px;">Thêm các cặp giá trị X và Y:</p>', unsafe_allow_html=True)

        # Khởi tạo session state để lưu dữ liệu
        if 'custom_data' not in st.session_state:
            st.session_state.custom_data = {'X': [], 'Y': []}

        # Hai cột để nhập X và Y
        col1, col2 = st.columns(2)
        with col1:
            x_input = st.number_input("Giá trị X:", value=0.0, step=0.1, key="x_input")
        with col2:
            y_input = st.number_input("Giá trị Y:", value=0.0, step=0.1, key="y_input")

        # Nút thêm điểm
        if st.button("➕ Thêm điểm"):
            st.session_state.custom_data['X'].append(x_input)
            st.session_state.custom_data['Y'].append(y_input)

        # Hiển thị dữ liệu đã nhập dưới dạng bảng
        if st.session_state.custom_data['X']:
            df = pd.DataFrame(st.session_state.custom_data)
            st.markdown('<p style="font-size:18px;">Dữ liệu đã nhập:</p>', unsafe_allow_html=True)
            st.dataframe(df)

            # Tùy chọn xóa điểm
            delete_index = st.selectbox("Chọn điểm để xóa (nếu cần):", options=range(len(st.session_state.custom_data['X'])), format_func=lambda i: f"Điểm {i}: X={st.session_state.custom_data['X'][i]}, Y={st.session_state.custom_data['Y'][i]}")
            if st.button("🗑️ Xóa điểm"):
                st.session_state.custom_data['X'].pop(delete_index)
                st.session_state.custom_data['Y'].pop(delete_index)
                st.rerun()

            # Chuyển dữ liệu thành numpy array
            X = np.array(st.session_state.custom_data['X']).reshape(-1, 1)
            y = np.array(st.session_state.custom_data['Y']).reshape(-1, 1)

            if len(X) < 2:
                st.error("Vui lòng nhập ít nhất 2 cặp dữ liệu để mô hình hóa!")
                return
        else:
            st.warning("Chưa có dữ liệu nào được thêm. Hãy nhập ít nhất 2 cặp X, Y để tiếp tục!")
            return

    # Phần 1: Giới thiệu
    st.markdown('<h2 style="font-size:32px; color:#32CD32;">📚 1. Linear Regression là gì?</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:22px;">
    ❓ Linear Regression là một phương pháp thống kê dùng để mô hình hóa mối quan hệ tuyến tính giữa biến độc lập (X) và biến phụ thuộc (Y).<br>
    🚀 Mục tiêu: Tìm đường thẳng tốt nhất để dự đoán Y từ X, dựa trên dữ liệu đã cho.<br>
    📐 Công thức cơ bản:
    </p>
    <div style="text-align: center;">
        <p style="font-size:28px; color:#FF69B4;"><b>y = w₀ + w₁x</b></p>
    </div>
    <p style="font-size:22px;">
    - <b>y</b>:  Giá trị cần dự đoán (biến phụ thuộc).<br>
    - <b>x</b>: Biến độc lập (input).<br>
    - <b>w₀</b>: Hệ số chặn (intercept).<br>
    - <b>w₁</b>: Độ dốc của đường thẳng (slope).<br>
    💡 Phương pháp này tối ưu hóa bằng cách giảm thiểu tổng bình phương sai số (Least Squares).
    </p>
    """, unsafe_allow_html=True)

    # Phần 2: Trực quan hóa ý tưởng
    st.markdown('<h2 style="font-size:32px; color:#FFD700;">📈 2. Cơ chế hoạt động của Linear Regression</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">👀 Hãy quan sát cách Linear Regression tìm đường thẳng phù hợp với dữ liệu:</p>', unsafe_allow_html=True)

    # Biểu đồ dữ liệu gốc
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='red', label='Dữ liệu thực tế', s=80)
    ax.set_xlabel('X (Biến độc lập)')
    ax.set_ylabel('Y (Biến phụ thuộc)')
    ax.set_title('Dữ liệu tuyến tính')
    ax.legend()
    st.pyplot(fig)

    st.markdown('<p style="font-size:22px;">🔍 Linear Regression sẽ tìm một đường thẳng tối ưu để biểu diễn mối quan hệ này.</p>', unsafe_allow_html=True)

    # Phần 3: Thực hành với Linear Regression
    st.markdown('<h2 style="font-size:32px; color:#00CED1;">🎮 3. Thực hành với Linear Regression</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">🛠️ Xây dựng mô hình Linear Regression và dự đoán giá trị cụ thể:</p>', unsafe_allow_html=True)

    # Huấn luyện mô hình Linear Regression
    with st.spinner("🔄 Đang xây dựng mô hình..."):
        model = LinearRegression()
        model.fit(X, y)

        # Dự đoán
        X_smooth = np.linspace(min(X), max(X), 100).reshape(-1, 1)
        y_pred = model.predict(X_smooth)

        # Tính MSE
        y_pred_full = model.predict(X)
        mse = mean_squared_error(y, y_pred_full)

        # Tạo biểu đồ
        fig, ax = plt.subplots()
        ax.scatter(X, y, color='red', label='Dữ liệu thực tế', s=80)
        ax.plot(X_smooth, y_pred, 'b-', label='Linear Regression', linewidth=2)
        ax.set_xlabel('X (Biến độc lập)')
        ax.set_ylabel('Y (Biến phụ thuộc)')
        ax.set_title('Mô hình Linear Regression')
        ax.legend()
        st.pyplot(fig)

    st.markdown(f'<p style="font-size:24px; color:#FF6347;">✅ Kết quả: Mô hình Linear Regression đã được xây dựng.</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:20px;">📌 <b>Hệ số chặn (w₀):</b> {model.intercept_[0]:.2f}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:20px;">📌 <b>Độ dốc (w₁):</b> {model.coef_[0][0]:.2f}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:20px; color:#4682B4;">📊 <b>Mean Squared Error (MSE):</b> {mse:.2f}</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:18px;">💡 MSE đo lường sai số trung bình bình phương giữa giá trị thực tế và dự đoán. MSE nhỏ hơn thường cho thấy mô hình tốt hơn.</p>', unsafe_allow_html=True)
    st.markdown(
    r"""
    ### 📌 Công thức tính MSE:
    """)
    st.latex(r"""
    MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
    """)
    st.markdown(
    """
    Trong đó:
    - $$y_i$$ là giá trị thực tế.
    - $$\hat{y}_i$$ là giá trị dự đoán.
    - $$n$$ là số lượng mẫu dữ liệu.
    
    Ý nghĩa:
    - MSE càng nhỏ → Mô hình dự đoán càng chính xác.
    - MSE = 0 → Mô hình hoàn hảo (hiếm khi xảy ra).
    - MSE lớn có thể cho thấy sự chênh lệch lớn giữa dự đoán và thực tế.
    """)
    # Phần dự đoán giá trị cụ thể
    st.markdown('<h3 style="font-size:26px; color:#4682B4;">🔍 Dự đoán giá trị Y từ X</h3>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:20px;">Nhập giá trị X (trong khoảng {min(X)[0]:.2f} đến {max(X)[0]:.2f}) để xem dự đoán và sai số:</p>', unsafe_allow_html=True)
    input_x = st.number_input("Nhập giá trị X:", min_value=float(min(X)), max_value=float(max(X)), value=float(X[len(X)//2]), step=0.1)

    # Dự đoán Y từ X nhập vào
    X_input = np.array([[input_x]])
    y_pred_input = model.predict(X_input)[0][0]

    # Tìm giá trị thực tế gần nhất trong dữ liệu mẫu để tính sai số
    idx_closest = np.argmin(np.abs(X - input_x))
    y_true_closest = y[idx_closest][0]
    error = abs(y_pred_input - y_true_closest)

    # Vẽ điểm dự đoán lên biểu đồ
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='red', label='Dữ liệu thực tế', s=80)
    ax.plot(X_smooth, y_pred, 'b-', label='Linear Regression', linewidth=2)
    ax.scatter([input_x], [y_pred_input], color='green', s=150, marker='*', label=f'Dự đoán (X={input_x:.2f}, Y={y_pred_input:.2f})')
    ax.set_xlabel('X (Biến độc lập)')
    ax.set_ylabel('Y (Biến phụ thuộc)')
    ax.set_title('Mô hình với điểm dự đoán')
    ax.legend()
    st.pyplot(fig)

    st.markdown(f'<p style="font-size:20px;">📈 <b>Giá trị dự đoán Y:</b> {y_pred_input:.2f}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:20px;">🔎 <b>Giá trị thực tế gần nhất (X={X[idx_closest][0]:.2f}):</b> {y_true_closest:.2f}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:20px; color:#FF4500;">⚠️ <b>Sai số tuyệt đối:</b> {error:.2f}</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:18px;">💡 Sai số cho thấy mức độ chênh lệch giữa dự đoán và thực tế.</p>', unsafe_allow_html=True)

    # Phần 4: Liên hệ với hạn chế
    st.markdown('<h2 style="font-size:32px; color:#FFA500;">⚠️ 4. Linear Regression và các hạn chế</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:22px;">
    👍 <b>Ưu điểm:</b><br>
    - 🌟 Đơn giản, dễ triển khai và giải thích.<br>
    - 📈 Hiệu quả với dữ liệu có mối quan hệ tuyến tính.<br>
    👎 <b>Nhược điểm:</b><br>
    - 🚨 Không phù hợp với dữ liệu phi tuyến (non-linear).<br>
    - ⚙️ Dễ bị ảnh hưởng bởi nhiễu (noise) hoặc ngoại lai (outliers).<br>
    💡 <b>Gợi ý:</b> Nếu dữ liệu có xu hướng cong, hãy cân nhắc Polynomial Regression.
    </p>
    """, unsafe_allow_html=True)

    # Phần 5: Tài liệu tham khảo
    st.markdown('<h2 style="font-size:32px; color:#1E90FF;">🔗 5. Tài liệu tham khảo</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">📖 Xem chi tiết về Linear Regression tại <a href="https://machinelearningcoban.com/2016/12/28/linearregression/">Machine Learning Cơ Bản</a>.</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px;">🙏 Cảm ơn bạn đã tham gia khám phá Linear Regression!</p>', unsafe_allow_html=True)

def mlflow_input():
    st.title("🚀 MLflow DAGsHub Tracking với Streamlit")
    
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/PTToan250303/Linear_replication.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url']=DAGSHUB_MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "PTToan250303"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "5ca8caf353d564c358852da97c7487e64fc30a73"

    mlflow.set_experiment("Linear_replication")


def drop(df):
    st.subheader("🗑️ Xóa cột dữ liệu")
    
    if "df" not in st.session_state:
        st.session_state.df = df  # Lưu vào session_state nếu chưa có

    df = st.session_state.df
    columns_to_drop = st.multiselect("📌 Chọn cột muốn xóa:", df.columns.tolist())

    if st.button("🚀 Xóa cột đã chọn"):
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)  # Tạo bản sao thay vì inplace=True
            st.session_state.df = df  # Cập nhật session_state
            st.success(f"✅ Đã xóa cột: {', '.join(columns_to_drop)}")
            st.dataframe(df.head())
        else:
            st.warning("⚠️ Vui lòng chọn ít nhất một cột để xóa!")

    return df

def choose_label(df):
    st.subheader("🎯 Chọn cột dự đoán (label)")

    if "target_column" not in st.session_state:
        st.session_state.target_column = None
    
    selected_label = st.selectbox("📌 Chọn cột dự đoán", df.columns, 
                                  index=df.columns.get_loc(st.session_state.target_column) if st.session_state.target_column else 0)

    X, y = df.drop(columns=[selected_label]), df[selected_label]  # Mặc định
    
    if st.button("✅ Xác nhận Label"):
        st.session_state.target_column = selected_label
        X, y = df.drop(columns=[selected_label]), df[selected_label]
        st.success(f"✅ Đã chọn cột: **{selected_label}**")
    
    return X, y

       
def xu_ly_gia_tri_thieu(df):
    if "df" not in st.session_state:
        st.session_state.df = df.copy()
    df = st.session_state.df

    # Tìm các cột có giá trị thiếu
    missing_cols = df.columns[df.isnull().any()].tolist()

    if not missing_cols:
        st.success("✅ Dữ liệu không có giá trị thiếu!")
        return df

    st.write("### 📌 Khi nào nên chọn phương pháp xử lý?")
    st.info("- **Xóa giá trị thiếu**: Nếu số lượng giá trị thiếu ít hoặc quá nhiều so với tổng dữ liệu.\n"
            "- **Thay thế bằng Mean (Trung bình)**: Nếu dữ liệu có phân phối chuẩn và không có quá nhiều outliers.\n"
            "- **Thay thế bằng Median (Trung vị)**: Nếu dữ liệu có nhiều phân phối lệch.\n"
            "- **Thay thế bằng Mode (Giá trị xuất hiện nhiều nhất)**: Nếu dữ liệu thuộc dạng phân loại (category).")

    selected_cols = st.multiselect("📌 Chọn cột chứa giá trị thiếu:", missing_cols)
    method = st.radio("🔧 Chọn phương pháp xử lý:", ["Xóa giá trị thiếu", "Thay thế bằng Mean", "Thay thế bằng Median", "Thay thế bằng Mode"])

    if st.button("🚀 Xử lý giá trị thiếu"):
        for col in selected_cols:
            if method == "Xóa giá trị thiếu":
                df = df.dropna(subset=[col])
            elif method == "Thay thế bằng Mean":
                df[col] = df[col].fillna(df[col].mean())
            elif method == "Thay thế bằng Median":
                df[col] = df[col].fillna(df[col].median())
            elif method == "Thay thế bằng Mode":
                df[col] = df[col].fillna(df[col].mode()[0])
        
        st.session_state.df = df
        st.success(f"✅ Đã xử lý giá trị thiếu cho các cột đã chọn")
    
    st.dataframe(df.head())
    return df

def chuyen_doi_kieu_du_lieu(df):

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if not categorical_cols:
        st.success("✅ Không có cột dạng chuỗi cần chuyển đổi!")
        return df
    st.write("Chuyển về kiểu dữ liệu số nguyên từ 1-n")
    selected_col = st.selectbox("📌 Cột cần chuyển đổi:", categorical_cols)
    unique_values = df[selected_col].unique()

    if "text_inputs" not in st.session_state:
        st.session_state.text_inputs = {}

    if "mapping_dicts" not in st.session_state:
        st.session_state.mapping_dicts = []

    mapping_dict = {}
    input_values = []
    has_duplicate = False
    has_empty = False  # Kiểm tra nếu có ô trống

    st.write("### Các giá trị cần chuyển đổi:")
    for val in unique_values:
        st.write(f"- `{val}`")  # Hiển thị từng giá trị trên một dòng

    if len(unique_values) < 10:
        for val in unique_values:
            key = f"{selected_col}_{val}"
            if key not in st.session_state.text_inputs:
                st.session_state.text_inputs[key] = ""

            new_val = st.text_input(f"🔄 Nhập giá trị thay thế cho `{val}`:", 
                                    key=key, 
                                    value=st.session_state.text_inputs[key])

            st.session_state.text_inputs[key] = new_val
            input_values.append(new_val)
            mapping_dict[val] = new_val

        # Kiểm tra ô trống
        if "" in input_values:
            has_empty = True

        # Kiểm tra trùng lặp
        duplicate_values = [val for val in input_values if input_values.count(val) > 1 and val != ""]
        if duplicate_values:
            has_duplicate = True
            st.warning(f"⚠ Giá trị `{', '.join(set(duplicate_values))}` đã được sử dụng nhiều lần. Vui lòng chọn số khác!")

        # Nút bị mờ nếu có trùng hoặc chưa nhập đủ giá trị
        btn_disabled = has_duplicate or has_empty

        if st.button("🚀 Chuyển đổi dữ liệu", disabled=btn_disabled):
            column_info = {"column_name": selected_col, "mapping_dict": mapping_dict}
            st.session_state.mapping_dicts.append(column_info)

            df[selected_col] = df[selected_col].map(lambda x: mapping_dict.get(x, x))
            df[selected_col] = pd.to_numeric(df[selected_col], errors='coerce')

            st.session_state.text_inputs.clear()
            st.session_state.df = df
            st.success(f"✅ Đã chuyển đổi cột `{selected_col}`")

    st.dataframe(df.head())
    return df


def chuan_hoa_du_lieu(df):
    # st.subheader("📊 Chuẩn hóa dữ liệu với SMinMaxScaler")

    # Lọc tất cả các cột số
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Tìm các cột nhị phân (chỉ chứa 0 và 1)
    binary_cols = [col for col in numerical_cols if df[col].dropna().isin([0, 1]).all()]

    # Loại bỏ cột nhị phân khỏi danh sách cần chuẩn hóa
    cols_to_scale = list(set(numerical_cols) - set(binary_cols))

    if not cols_to_scale:
        st.success("✅ Không có thuộc tính dạng số cần chuẩn hóa!")
        return df

    if st.button("🚀 Thực hiện Chuẩn hóa"):
        scaler = MinMaxScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

        # Lưu vào session_state
        st.session_state.df = df

        st.success(f"✅ Đã chuẩn hóa xong")
        st.dataframe(df.head())

    return df

def hien_thi_ly_thuyet(df):

                # Kiểm tra lỗi dữ liệu
    st.subheader("🚨 Kiểm tra dữ liệu")
                # Kiểm tra giá trị thiếu
    missing_values = df.isnull().sum()

                # Kiểm tra dữ liệu trùng lặp
    duplicate_count = df.duplicated().sum()

                # Tạo báo cáo lỗi
    error_report = pd.DataFrame({
        'Giá trị thiếu': missing_values,
        'Dữ liệu trùng lặp': duplicate_count,
        'Tỉ lệ trùng lặp (%)': round((duplicate_count / df.shape[0]) * 100,2),
        'Kiểu dữ liệu': df.dtypes.astype(str)
    })

                # Hiển thị báo cáo lỗi
    st.table(error_report)          
   
    
    st.title("🔍 Tiền xử lý dữ liệu")

    # Hiển thị dữ liệu gốc
    
    st.header("⚙️ Các bước chính trong tiền xử lý dữ liệu")
    st.subheader("1️⃣ Loại bỏ các cột không cần thiết")


    df=drop(df)
    
    st.subheader("2️⃣ Xử lý giá trị thiếu")
    df=xu_ly_gia_tri_thieu(df)

    st.subheader("3️⃣ Chuyển đổi kiểu dữ liệu")

    df=chuyen_doi_kieu_du_lieu(df)
    
    st.subheader("4️⃣ Chuẩn hóa dữ liệu số")
 
    df=chuan_hoa_du_lieu(df)
def train_test_size():
    if "df" not in st.session_state:
        st.error("❌ Dữ liệu chưa được tải lên!")
        st.stop()
    
    df = st.session_state.df  # Lấy dữ liệu từ session_stat
    X, y = choose_label(df)
    
    st.subheader("📊 Chia dữ liệu Train - Validation - Test")   
    
    test_size = st.slider("📌 Chọn % dữ liệu Test", 10, 50, 20)
    remaining_size = 100 - test_size
    val_size = st.slider("📌 Chọn % dữ liệu Validation (trong phần Train)", 0, 50, 15)

    st.write(f"📌 **Tỷ lệ phân chia:** Test={test_size}%, Validation={val_size}%, Train={remaining_size - val_size}%")

    

    if st.button("✅ Xác nhận Chia"):
        # st.write("⏳ Đang chia dữ liệu...")

        stratify_option = y if y.nunique() > 1 else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size/100, stratify=stratify_option, random_state=42
        )

        stratify_option = y_train_full if y_train_full.nunique() > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size / (100 - test_size),
            stratify=stratify_option, random_state=42
        )

        # st.write(f"📊 Kích thước tập Train: {X_train.shape[0]} mẫu")
        # st.write(f"📊 Kích thước tập Validation: {X_val.shape[0]} mẫu")
        # st.write(f"📊 Kích thước tập Test: {X_test.shape[0]} mẫu")

        # Lưu vào session_state
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.y = y
        st.session_state.X_train_shape = X_train.shape[0]
        st.session_state.X_val_shape = X_val.shape[0]
        st.session_state.X_test_shape = X_test.shape[0]
        summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        })
        st.table(summary_df)

        # **Log dữ liệu vào MLflow**    
def chia():
    st.subheader("Chia dữ liệu thành tập Train, Validation, và Test")
    st.write("""
    ### 📌 Chia tập dữ liệu
    Dữ liệu được chia thành ba phần để đảm bảo mô hình tổng quát tốt:
    - **Trian(%)**: để train mô hình.
    - **Val (%)**: để validation, dùng để điều chỉnh tham số.
    - **Test(%)**: để test, đánh giá hiệu suất thực tế.
    """)
    train_test_size()

from sklearn.pipeline import make_pipeline   
from sklearn.model_selection import train_test_split, cross_val_score

def train_multiple_linear_regression(X_train, y_train, learning_rate=0.001, n_iterations=200):
    """Huấn luyện hồi quy tuyến tính bội bằng Gradient Descent."""
    
    # Chuyển đổi X_train, y_train sang NumPy array để tránh lỗi
    X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.to_numpy().reshape(-1, 1) if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train.reshape(-1, 1)

    # Kiểm tra NaN hoặc Inf
    if np.isnan(X_train).any() or np.isnan(y_train).any():
        raise ValueError("Dữ liệu đầu vào chứa giá trị NaN!")
    if np.isinf(X_train).any() or np.isinf(y_train).any():
        raise ValueError("Dữ liệu đầu vào chứa giá trị vô cùng (Inf)!")

    # Chuẩn hóa dữ liệu để tránh tràn số
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    # Lấy số lượng mẫu (m) và số lượng đặc trưng (n)
    m, n = X_train.shape
    #st.write(f"Số lượng mẫu (m): {m}, Số lượng đặc trưng (n): {n}")

    # Thêm cột bias (x0 = 1) vào X_train
    X_b = np.c_[np.ones((m, 1)), X_train]
    #st.write(f"Kích thước ma trận X_b: {X_b.shape}")

    # Khởi tạo trọng số ngẫu nhiên nhỏ
    w = np.random.randn(X_b.shape[1], 1) * 0.01  
    #st.write(f"Trọng số ban đầu: {w.flatten()}")

    # Gradient Descent
    for iteration in range(n_iterations):
        gradients = (2/m) * X_b.T.dot(X_b.dot(w) - y_train)

        # Kiểm tra xem gradients có NaN không
        # st.write(gradients)
        if np.isnan(gradients).any():
            raise ValueError("Gradient chứa giá trị NaN! Hãy kiểm tra lại dữ liệu hoặc learning rate.")

        w -= learning_rate * gradients

    #st.success("✅ Huấn luyện hoàn tất!")
    #st.write(f"Trọng số cuối cùng: {w.flatten()}")
    return w
def train_polynomial_regression(X_train, y_train, degree=2, learning_rate=0.001, n_iterations=500):
    """Huấn luyện hồi quy đa thức **không có tương tác** bằng Gradient Descent."""

    # Chuyển dữ liệu sang NumPy array nếu là pandas DataFrame/Series
    X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.to_numpy().reshape(-1, 1) if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train.reshape(-1, 1)

    # Tạo đặc trưng đa thức **chỉ thêm bậc cao, không có tương tác**
    X_poly = np.hstack([X_train] + [X_train**d for d in range(2, degree + 1)])
    # Chuẩn hóa dữ liệu để tránh tràn số
    scaler = MinMaxScaler()
    X_poly = scaler.fit_transform(X_poly)

    # Lấy số lượng mẫu (m) và số lượng đặc trưng (n)
    m, n = X_poly.shape
    print(f"Số lượng mẫu (m): {m}, Số lượng đặc trưng (n): {n}")

    # Thêm cột bias (x0 = 1)
    X_b = np.c_[np.ones((m, 1)), X_poly]
    print(f"Kích thước ma trận X_b: {X_b.shape}")

    # Khởi tạo trọng số ngẫu nhiên nhỏ
    w = np.random.randn(X_b.shape[1], 1) * 0.01  
    print(f"Trọng số ban đầu: {w.flatten()}")

    # Gradient Descent
    for iteration in range(n_iterations):
        gradients = (2/m) * X_b.T.dot(X_b.dot(w) - y_train)

        # Kiểm tra nếu gradient có giá trị NaN
        if np.isnan(gradients).any():
            raise ValueError("Gradient chứa giá trị NaN! Hãy kiểm tra lại dữ liệu hoặc learning rate.")

        w -= learning_rate * gradients

    print("✅ Huấn luyện hoàn tất!")
    print(f"Trọng số cuối cùng: {w.flatten()}")
    
    return w




def chon_mo_hinh():
    st.subheader("🔍 Chọn mô hình hồi quy")

    model_type_V = st.radio("Chọn loại mô hình:", ["Multiple Linear Regression", "Polynomial Regression"], key="model_type")
    model_type = "linear" if model_type_V == "Multiple Linear Regression" else "polynomial"

    n_folds = st.slider("Chọn số folds (KFold Cross-Validation):", min_value=2, max_value=10, value=5, key="n_folds")

    learning_rate = st.slider("Chọn tốc độ học (learning rate):", 
                              min_value=1e-6, max_value=0.1, value=0.01, step=1e-6, format="%.6f", key="learning_rate")

    degree = 2
    if model_type == "polynomial":
        degree = st.slider("Chọn bậc đa thức:", min_value=2, max_value=5, value=2, key="degree")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    if "X_train" not in st.session_state or st.session_state.X_train is None:
        st.warning("⚠️ Vui lòng chia dữ liệu trước khi huấn luyện mô hình!")
        return None, None, None

    X_train, X_test = st.session_state.X_train, st.session_state.X_test
    y_train, y_test = st.session_state.y_train, st.session_state.y_test
    st.session_state["X_train"] = X_train
    st.session_state["y_train"] = y_train
    df = st.session_state.df
    # Chỉ nhập tên Experiment (Không có phần nhập tên Run)
    if "experiment_name" not in st.session_state:
        st.session_state["experiment_name"] = "My_Experiment"

    experiment_name = st.text_input("🔹 Nhập tên Experiment:", st.session_state["experiment_name"], key="experiment_name_input")    

    if experiment_name:
        st.session_state["experiment_name"] = experiment_name

    mlflow.set_experiment(experiment_name)
    st.write(f"✅ Experiment Name: {experiment_name}")

    if st.button("Huấn luyện mô hình", key="train_button"):
        with mlflow.start_run():
            mlflow.log_param("dataset_shape", df.shape)
            mlflow.log_param("target_column", st.session_state.y.name)
            mlflow.log_param("train_size", X_train.shape)
            mlflow.log_param("test_size", X_test.shape)
            
            dataset_path = "dataset.csv"
            df.to_csv(dataset_path, index=False)
            mlflow.log_artifact(dataset_path)

            mlflow.log_param("model_type", model_type)
            mlflow.log_param("n_folds", n_folds)
            mlflow.log_param("learning_rate", learning_rate)
            if model_type == "polynomial":
                mlflow.log_param("degree", degree)

            fold_mse = []
            model = None

            for train_idx, valid_idx in kf.split(X_train):
                X_train_fold, X_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
                y_train_fold, y_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]

                if model_type == "linear":
                    model = LinearRegression()
                    model.fit(X_train_fold, y_train_fold)
                    y_valid_pred = model.predict(X_valid)
                else:  
                    poly_features = PolynomialFeatures(degree=degree)
                    X_train_poly = poly_features.fit_transform(X_train_fold)
                    X_valid_poly = poly_features.transform(X_valid)

                    model = LinearRegression()
                    model.fit(X_train_poly, y_train_fold)
                    y_valid_pred = model.predict(X_valid_poly)

                mse = mean_squared_error(y_valid, y_valid_pred)
                fold_mse.append(mse)

            avg_mse = np.mean(fold_mse)
            mlflow.log_metric("avg_mse", avg_mse)

            st.success(f"MSE trung bình qua các folds: {avg_mse:.4f}")
            # Lưu mô hình vào session_state
            if model_type == "linear":
                st.session_state["linear_model"] = model
            elif model_type == "polynomial":
                st.session_state["polynomial_model"] = model
                st.session_state["poly_features"] = poly_features  # Lưu bộ biến đổi PolynomialFeatures

        return model, avg_mse, None

    return None, None, None




import numpy as np
import streamlit as st

def test():
    st.subheader("📌 Dự đoán với mô hình đã huấn luyện")

    # Chọn mô hình
    model_type = st.selectbox("Chọn mô hình:", ["linear", "polynomial"])
    
    if model_type == "linear" and "linear_model" in st.session_state:
        model = st.session_state["linear_model"]
    elif model_type == "polynomial" and "polynomial_model" in st.session_state:
        model = st.session_state["polynomial_model"]
        poly_features = st.session_state.get("poly_features", None)
        if poly_features is None:
            st.error("⚠ Không tìm thấy poly_features trong session_state. Hãy huấn luyện mô hình lại!")
            return
    else:
        st.warning("⚠ Mô hình chưa được huấn luyện. Vui lòng huấn luyện trước khi dự đoán!")
        return
    
    # Kiểm tra xem dữ liệu test có tồn tại không
    if "X_test" not in st.session_state or st.session_state.X_test is None:
        st.error("⚠ Dữ liệu kiểm tra không tồn tại! Hãy đảm bảo mô hình đã được huấn luyện.")
        return

    X_test = st.session_state.X_test
    column_names = X_test.columns.tolist()
    categorical_columns = X_test.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = X_test.select_dtypes(include=['number']).columns.tolist()
    
    # Kiểm tra nếu có dữ liệu mapping_dicts trong session_state
    if "mapping_dicts" not in st.session_state:
        st.session_state.mapping_dicts = []
    
    # Nhập dữ liệu thực tế từ người dùng
    user_input = {}
    for column_name in column_names:
        mapping_dict = next((d["mapping_dict"] for d in st.session_state.mapping_dicts if d["column_name"] == column_name), None)
        
        if column_name in categorical_columns and mapping_dict:
            value = st.selectbox(f"Chọn giá trị cho {column_name}:", options=list(mapping_dict.keys()), key=f"category_{column_name}")
            user_input[column_name] = mapping_dict[value]
        else:
            user_input[column_name] = st.number_input(f"Nhập giá trị thực tế cho {column_name}:", key=f"column_{column_name}")
    
    # Chuyển đổi thành DataFrame
    X_input_df = pd.DataFrame([user_input])
    
    # Chuẩn hóa dữ liệu số về khoảng [0,1]
    scaler = st.session_state.get("scaler", None)
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(st.session_state.X_train[numerical_columns])  # Fit trên dữ liệu train
        st.session_state["scaler"] = scaler  # Lưu lại để không bị mất sau khi chạy lại app

    X_input_df[numerical_columns] = scaler.transform(X_input_df[numerical_columns])

    X_input_array = X_input_df.to_numpy()
    
    # Xử lý với Polynomial Regression nếu cần
    if model_type == "polynomial" and poly_features:
        X_input_array = poly_features.transform(X_input_array)
    
    # Dự đoán kết quả
    if st.button("📌 Dự đoán"):
        y_pred = model.predict(X_input_array)
        prediction_label = "Sống 🟢" if y_pred[0] >= 0.5 else "Chết 🔴"
        
        # Tính toán độ tin cậy dựa trên tập kiểm tra
        y_test = st.session_state.get("y_test", None)
        if y_test is not None:
            mean_test_value = np.mean(y_test)
            confidence = max(0, 1 - abs(y_pred[0] - mean_test_value) / mean_test_value) * 100
        else:
            confidence = abs(y_pred[0] - 0.5) * 200  # Chuyển đổi khoảng [0.5, 1] thành [0, 100]
        confidence = min(max(confidence, 0), 100)  # Giới hạn từ 0 đến 100
        
        st.write(f"📊 **Dự đoán:** {prediction_label}")
        st.write(f"📈 **Độ tin cậy:** {confidence:.2f}%")
        
        # Giải thích độ tin cậy
        st.info("🔍 Độ tin cậy được tính dựa trên khoảng cách giữa dự đoán và trung bình của tập kiểm tra, nếu không có thì dùng khoảng cách với ngưỡng 0.5.")
       
            
import streamlit as st
import mlflow
import os

import streamlit as st
import mlflow
import os
import pandas as pd
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
def chon():
    try:
                
        final_w, avg_mse, test_mse = chon_mo_hinh()
    except Exception as e:
        st.error(f"Lỗi xảy ra: {e}")


def data(df):
    """Hiển thị dữ liệu đã tải lên"""
    if df is not None:
        st.success("📂 File đã được tải lên thành công!")
        hien_thi_ly_thuyet(df)
    else:
        st.error("❌ Không có dữ liệu để hiển thị.")
def Classification():
    # Định dạng tiêu đề
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
        <div class="title">Linear Regression</div>
        <hr>
    """, unsafe_allow_html=True)

    # Cho phép người dùng tải một file duy nhất
    uploaded_file = st.file_uploader("📥 Chọn một file dataset", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                st.write("Định dạng tệp CSV hợp lệ.")
            else:
                st.error("❌ Định dạng tệp không hợp lệ. Vui lòng tải lại tệp .csv")
                return  # Dừng chương trình nếu tải sai file
        except Exception as e:
            st.error(f"⚠️ Lỗi khi đọc tệp: {e}")
            return

        st.success(f"✅ Đã tải lên: {uploaded_file.name}")
        st.write(df)  # Hiển thị toàn bộ dataset

        # Chỉ hiển thị thanh điều hướng khi có file hợp lệ
        tab1, tab2, tab3, tab4, tab5,tab6 = st.tabs([
            "📘 LT Linear Regression",
            "📘 LT Polynomial Regression",
            "📊 Xử lý dữ liệu",
            "⚙️ Huấn luyện", 
            "💡 Demo",
            "📝 MLflow"
        ])

        with tab1:
            run_linear_regression_app()
        with tab2:
            run_polynomial_regression_app()

        with tab3:
            data(df)
        with tab4:
            chia()
            chon()
        with tab5:
            test()
        with tab6:
            show_experiment_selector()

if __name__ == "__main__":
    Classification()
