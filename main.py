# main.py


import streamlit as st
from streamlit_option_menu import option_menu
import buoi1.Linear_Regression as Linear_Regression
import buoi2.MNIST as b2
import buoi3.Clustering_Algorithms as b3
import buoi4.PCA_t_SNE as b4
import buoi5.NN as b5
import buoi6.PL_NN as b6
def main():
    with st.sidebar:
        selected = option_menu(
            "Menu",
            ["Trang chủ", "Linear Regression","Assignment - Classification","Clustering Algorithms","PCA & t-SNE","Neural Network","Pseudo Labelling"],
            icons=["house"],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "Trang chủ":
        st.title("Chào mừng đến với ứng dụng của tôi!")
        st.write("Chọn một tùy chọn từ menu để tiếp tục.")

    elif selected == "Linear Regression":
        Linear_Regression.Classification()  # Gọi hàm từ buoi1.py
    elif selected == "Assignment - Classification":
        b2.Classification()  # Gọi hàm từ buoi2.py
    elif selected == "Clustering Algorithms":
        b3.ClusteringAlgorithms()  # Gọi hàm từ buoi3.py
    elif selected == "PCA & t-SNE":
        b4.pca_tsne()# Gọi hàm từ buoi4.py
    elif selected == "Neural Network":
        b5.Classification()
    elif selected == "Pseudo Labelling":
        b6.Classification()
if __name__ == "__main__":
    main()
