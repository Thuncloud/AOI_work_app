import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="影像處理與 AOI 應用程式", layout="wide")

st.title("AOI 影像處理與演算法作業")
st.write("本應用程式包含週期性干擾濾除、Canny 邊緣檢測以及圓形檢測功能。")

# 側邊欄選擇功能
option = st.sidebar.selectbox(
    "選擇作業單元",
    ["1. 週期性干擾濾除", "2. Canny 邊緣檢測", "3. 圓形與線條檢測"]
)

# 圖片上傳區域
uploaded_file = st.sidebar.file_uploader("請上傳一張影像", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 讀取圖片並轉為 OpenCV 格式
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # 轉為灰階（若需要）
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    st.sidebar.image(image, caption="原始影像", use_column_width=True)
    
    if option == "1. 週期性干擾濾除":
        st.header("1. 週期性干擾濾除")
        st.write("透過調整濾波頻率與參數來移除週期性干擾。")
        
        # 參數調整介面
        col1, col2 = st.columns(2)
        with col1:
            d1 = st.slider("d1", min_value=1, max_value=30, value=7, step=1)
            d2 = st.slider("d2", min_value=1, max_value=30, value=14, step=1)
            d3 = st.slider("d3", min_value=1, max_value=30, value=21, step=1)
        with col2:
            sigma_u = st.slider("sigma_u", min_value=0.1, max_value=10.0, value=3.5, step=0.1)
            sigma_v = st.slider("sigma_v", min_value=1.0, max_value=50.0, value=18.0, step=0.5)
            
        # 濾波邏輯
        st.info("頻率參數更新完成，請套用至頻率域濾波函式 (如 Notch Filter)。")

    elif option == "2. Canny 邊緣檢測":
        st.header("2. Canny 邊緣檢測")
        st.write("調整參數以調整靈敏度，進行影像分割與邊緣檢測。")
        
        min_val = st.slider("低閾值 (Threshold 1)", min_value=0, max_value=255, value=50)
        max_val = st.slider("高閾值 (Threshold 2)", min_value=0, max_value=255, value=150)
        aperture = st.selectbox("Sobel 核心大小 (Aperture Size)", [3, 5, 7], index=0)
        
        edges = cv2.Canny(gray_image, min_val, max_val, apertureSize=aperture)
        
        col1, col2 = st.columns(2)
        col1.image(image, caption="原始影像", use_column_width=True)
        col2.image(edges, caption="Canny 邊緣檢測結果", use_column_width=True)

    elif option == "3. 圓形與線條檢測":
        st.header("3. 圓形與線條檢測")
        st.write("調整靈敏度與支援度 (Support) 參數，實現特徵檢測。")
        
        param1 = st.slider("霍氏轉換參數 1 (靈敏度)", min_value=10, max_value=200, value=100)
        param2 = st.slider("霍氏轉換參數 2 (累積閥值)", min_value=10, max_value=100, value=30)
        min_r = st.slider("最小半徑", min_value=5, max_value=100, value=10)
        max_r = st.slider("最大半徑", min_value=10, max_value=200, value=50)
        
        output_image = image.copy()
        circles = cv2.HoughCircles(
            gray_image, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=50, 
            param1=param1, 
            param2=param2, 
            minRadius=min_r, 
            maxRadius=max_r
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(output_image, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output_image, (x - 2, y - 2), (x + 2, y + 2), (0, 0, 255), -1)
                
            st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), caption="圓形檢測結果", use_column_width=True)
        else:
            st.warning("未偵測到符合條件的圓形。")
            st.image(image, caption="未改變之影像", use_column_width=True)
else:
    st.info("請在側邊欄上傳圖片以開始操作。")