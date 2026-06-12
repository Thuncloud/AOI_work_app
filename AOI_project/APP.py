import streamlit as st

st.sidebar.title("🎛️ 模組化演算法控制面板")

# 1. 在 Session State 中初始化一個空列表，用來動態儲存執行步驟
if "pipeline" not in st.session_state:
    st.session_state.pipeline = []

# 2. 用 st.expander 或不同區塊將演算法「分門別類」
with st.sidebar.expander("✨ 步驟 1：影像前處理與降噪", expanded=True):
    preprocess_opt = st.selectbox("選擇前處理功能：", ["無", "轉為灰階", "顏色翻轉", "高斯模糊 (濾波)", "中值濾波 (去噪)"])
    if st.button("➕ 加入前處理步驟") and preprocess_opt != "無":
        st.session_state.pipeline.append(preprocess_opt)

with st.sidebar.expander("📐 步驟 2：空間域與特徵偵測", expanded=True):
    feature_opt = st.selectbox("選擇特徵偵測：", ["無", "Canny 邊緣檢測", "二值化處理", "Hough 直線偵測", "方向性邊緣偵測", "圓圈檢測"])
    if st.button("➕ 加入特徵步驟") and feature_opt != "無":
        st.session_state.pipeline.append(feature_opt)

with st.sidebar.expander("🌌 步驟 3：頻域濾波分析", expanded=True):
    frequency_opt = st.selectbox("選擇頻域濾波：", ["無", "低通濾波", "高通濾波", "帶通/帶拒"])
    if st.button("➕ 加入頻域步驟") and frequency_opt != "無":
        st.session_state.pipeline.append(frequency_opt)

# 3. 管理與顯示目前的流水線
st.sidebar.divider()
st.sidebar.subheader("📋 目前排定的 AOI 流水線")

if st.session_state.pipeline:
    for idx, step in enumerate(st.session_state.pipeline):
        st.sidebar.text(f"{idx + 1}. {step}")
    
    if st.sidebar.button("🗑️ 清空所有步驟"):
        st.session_state.pipeline = []
else:
    st.sidebar.caption("目前尚未選擇任何步驟")

# 主畫面就可以直接用 for step in st.session_state.pipeline: 來跑你的 OpenCV 邏輯了！
