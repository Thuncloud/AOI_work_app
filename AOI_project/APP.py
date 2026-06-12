import streamlit as st

st.title("🔬 AOI 演算法平台")

# 在主畫面或側邊欄建立分頁
tab1, tab2, tab3 = st.tabs(["🖼️ 空間域點/鄰域運算", "🔍 幾何與特徵偵測", "🌊 傅立葉頻域分析"])

with tab1:
    st.subheader("影像基礎前處理")
    # 這裡放 灰階、翻轉、高斯模糊的控制拉桿與開關
    blur_on = st.checkbox("啟用高斯模糊")
    if blur_on:
        k_size = st.slider("核大小", 1, 15, 5, step=2)

with tab2:
    st.subheader("邊緣與形狀提取")
    # 這裡放 Canny、Hough 直線、圓圈偵測的控制項
    canny_on = st.checkbox("啟用 Canny 邊緣偵測")

with tab3:
    st.subheader("頻率域遮罩過濾")
    # 這裡放 理想/高斯/巴特沃斯 濾波器
    fft_mode = st.radio("濾波模式", ["低通", "高通", "帶通"])
