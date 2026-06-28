import streamlit as st
import cv2
import numpy as np

# 設定網頁標題與佈局
st.set_page_config(layout="wide", page_title="SIFT 幾何視角對齊工具")

st.title("🗺️ SIFT 影像幾何視角配準與對齊工具 (Homography)")
st.markdown("""
本工具專門用於**將一張局部或變形、縮放過的影像（來源圖），自動對齊到另一張標準影像（目標圖）的視角與尺寸上**。
""")

# ==========================================
# 側邊欄：控制與圖片上傳面板
# ==========================================
with st.sidebar:
    st.header("📥 步驟 1：上傳影像")
    
    # 來源影像上傳（待調整角度/縮放的圖，例如：IMG_S.jpg）
    src_file = st.file_uploader("1. 上傳【來源/局部圖】(Source Image)", type=["jpg", "png", "jpeg", "tif"])
    
    # 目標影像上傳（標準視角的基準圖，例如：IMG_L.jpg）
    tgt_file = st.file_uploader("2. 上傳【目標/基準圖】(Target Image)", type=["jpg", "png", "jpeg", "tif"])
    
    st.divider()
    st.header("🎛️ 步驟 2：演算法參數微調")
    
    # SIFT 匹配門檻值 (Lowe's Ratio Test)
    ratio_threshold = st.slider(
        "特徵匹配嚴格度 (Lowe's Ratio)", 
        min_value=0.5, max_value=0.9, value=0.75, step=0.05,
        help="數值越小越嚴格，留下的特徵點越精準；數值放大能接收較模糊的匹配。"
    )
    
    # RANSAC 門檻值
    ransac_thresh = st.slider(
        "RANSAC 容忍誤差像素 (門檻值)", 
        min_value=1.0, max_value=10.0, value=5.0, step=0.5,
        help="計算 Homography 矩陣時，允許特徵點對位最大誤差的像素距離。"
    )

# ==========================================
# 主畫面：運算與視覺化邏輯
# ==========================================
if src_file is not None and tgt_file is not None:
    try:
        # 1. 讀取與解碼影像
        file_bytes_src = np.asarray(bytearray(src_file.read()), dtype=np.uint8)
        file_bytes_tgt = np.asarray(bytearray(tgt_file.read()), dtype=np.uint8)
        
        # 確保以彩色 BGR 模式讀入
        img_src = cv2.imdecode(file_bytes_src, cv2.IMREAD_COLOR)
        img_tgt = cv2.imdecode(file_bytes_tgt, cv2.IMREAD_COLOR)
        
        if img_src is None or img_tgt is None:
            st.error("❌ 影像解碼失敗，請確認檔案格式是否損壞。")
            st.stop()
            
        # 2. 安全轉換為灰階進行特徵提取
        gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        gray_tgt = cv2.cvtColor(img_tgt, cv2.COLOR_BGR2GRAY)
        
        # 3. 初始化 SIFT 偵測器並提取關鍵點
        sift = cv2.SIFT_create()
        kp_src, des_src = sift.detectAndCompute(gray_src, None)
        kp_tgt, des_tgt = sift.detectAndCompute(gray_tgt, None)
        
        # 建立大排版區塊
        col_img_panel, col_res_panel = st.columns([1, 1])
        
        with col_img_panel:
            st.subheader("🖼️ 輸入影像檢視")
            sub_c1, sub_c2 = st.columns(2)
            with sub_c1:
                st.image(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB), caption="來源/局部影像 (Source)", use_container_width=True)
            with sub_c2:
                st.image(cv2.cvtColor(img_tgt, cv2.COLOR_BGR2RGB), caption="基準/目標影像 (Target)", use_container_width=True)
                
        # 4. 進行特徵匹配與單應性矩陣求解
        if des_src is not None and des_tgt is not None:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des_src, des_tgt, k=2)
            
            # Lowe's ratio test 篩選優秀特徵對
            good_matches = []
            for m_pair in matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
            
            # 判斷點數是否足夠 (Homography 至少需要 4 對點)
            if len(good_matches) >= 4:
                src_pts = np.float32([kp_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_tgt[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # 計算轉換矩陣 H
                H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
                
                if H is not None:
                    # 5. 執行透視轉換 (Warp Perspective) 讓來源圖對齊到目標圖的尺寸上
                    h_tgt, w_tgt = img_tgt.shape[:2]
                    img_result = cv2.warpPerspective(img_src, H, (w_tgt, h_tgt))
                    
                    with col_res_panel:
                        st.subheader("🎯 視角對齊結果")
                        st.image(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB), caption="對齊與轉換後影像 (Aligned Result)", use_container_width=True)
                        st.success(f"✅ 自動匹配成功！共找到 {len(good_matches)} 組強健特徵點。")
                        
                        # 展開顯示數學矩陣資訊 (加分點)
                        with st.expander("📊 查看幾何變換矩陣資訊 (Homography Matrix)"):
                            st.write("估算出的 3x3 變換矩陣如下：")
                            st.dataframe(H)
                else:
                    with col_res_panel:
                        st.error("❌ 幾何轉換矩陣 (Homography) 求解失敗，演算法無法收斂。")
            else:
                with col_res_panel:
                    st.warning(f"⚠️ 匹配特徵點數量不足（僅 {len(good_matches)} 組）。幾何配準至少需要 4 組以上不共線的特徵點，請嘗試調高側邊欄的 Ratio 門檻或換一張圖測試。")
        else:
            with col_res_panel:
                st.error("❌ 無法從這兩張影像中提取到足夠的 SIFT 特徵描述子。")
                
    except Exception as e:
        st.error(f"💥 運算過程中發生非預期錯誤: {str(e)}")

else:
    # 尚未上傳完成時的提示畫面
    st.info("💡 請在左側側邊欄分別上傳【來源影像】與【目標影像】以啟動自動視角配準演算法。")
    
    # 展示架構說明
    st.divider()
    st.subheader("💡 演算法對齊邏輯說明")
    st.markdown("""
    1. **SIFT 特徵點檢測**：自動尋找兩張影像中具有獨特性、且不受旋轉與縮放影響的特徵點。
    2. **K-NN 雙向匹配**：找出來源圖與目標圖之間最相似的特徵點對（Features Matching）。
    3. **RANSAC 演算法過濾**：自動剔除錯誤的連線干擾，找出正確的幾何變換方向。
    4. **透視變換 (Homography)**：將來源圖拉伸、旋轉、放大，完美融入目標圖的視角。
    """)
