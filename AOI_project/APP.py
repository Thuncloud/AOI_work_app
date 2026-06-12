import streamlit as st

# 用字典把類別與功能綁定
ALGORITHM_CATEGORIES = {
    "灰階與降噪": ["轉為灰階", "顏色翻轉", "高斯模糊 (濾波)", "中值濾波 (去噪)"],
    "特徵提取": ["Canny 邊緣檢測", "二值化處理", "Hough 直線偵測", "方向性邊緣偵測", "圓圈檢測"],
    "頻域處理": ["低通濾波", "高通濾波", "帶通/帶拒"]
}

# 把所有功能攤平成一個大列表供 multiselect 使用
all_options = []
for category, steps in ALGORITHM_CATEGORIES.items():
    all_options.extend(steps)

# 渲染到畫面上
selected_steps = st.multiselect("請依序選擇處理步驟：", all_options)
