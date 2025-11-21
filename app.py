import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import glob
from datetime import datetime
import os
import gdown

# Page config
st.set_page_config(page_title="Buôn Bán Xe Máy", page_icon="🏍️", layout="wide")


# ==============================
# 🔄 SCROLL TO TOP FUNCTION
# ==============================
def scroll_to_top():
    """JavaScript để cuộn lên đầu trang"""
    st.components.v1.html(
        """
        <script>
            window.parent.document.querySelector('section.main').scrollTo(0, 0);
        </script>
        """,
        height=0,
    )


# ==============================
# 🖼️ BANNER TIÊU ĐỀ Ở ĐẦU TRANG
# ==============================
if os.path.exists("unnamed.jpg"):
    st.image("unnamed.jpg", use_column_width=True)
else:
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px;'>
        <h1 style='color: white; margin: 0;'>🏍️ HỆ THỐNG BUÔN BÁN XE MÁY</h1>
        <p style='color: white; margin: 10px 0 0 0;'>Tìm kiếm và gợi ý xe máy thông minh</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# ==============================
# 📥 DOWNLOAD MODEL FROM GOOGLE DRIVE
# ==============================
def download_from_gdrive(file_id, output_path):
    """Download file từ Google Drive với error handling tốt hơn"""
    if os.path.exists(output_path):
        return True
    
    try:
        # URL format cho gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        
        # Download với fuzzy=True để xử lý file lớn
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        
        # Kiểm tra file đã download thành công chưa
        if os.path.exists(output_path):
            return True
        else:
            st.error(f"❌ Không thể download file. Vui lòng kiểm tra lại File ID và quyền truy cập.")
            return False
            
    except Exception as e:
        st.error(f"❌ Lỗi khi download: {str(e)}")
        st.info("""
        **Hướng dẫn khắc phục:**
        1. Đảm bảo file trên Google Drive được share với quyền "Anyone with the link can view"
        2. Kiểm tra File ID có đúng không
        3. Link Google Drive: https://drive.google.com/file/d/FILE_ID/view
        """)
        return False


@st.cache_resource
def load_model():
    """Load model và dataframe"""
    
    # Tạo thư mục nếu chưa có
    os.makedirs("recommendation_model", exist_ok=True)
    
    # ⚠️ THAY ĐỔI FILE IDs CỦA BẠN Ở ĐÂY
    # Lấy từ link: https://drive.google.com/file/d/FILE_ID_HERE/view
    MODEL_FILE_ID = "1que7me49U47W0JjV6Es8t1p-d5LLpBg7"  # ← Thay bằng ID của bạn
    DF_FILE_ID = "14sM9VEkJB65DYdB9W4AtemesmjXlV20o"     # ← Thay bằng ID của bạn
    
    model_path = "recommendation_model/model_v4.joblib"
    df_path = "recommendation_model/df_items.joblib"
    
    # Download files nếu chưa có
    if not os.path.exists(model_path) or not os.path.exists(df_path):
        st.info("🔄 Đang tải model lần đầu tiên... Quá trình này có thể mất vài phút.")
        
        # Download model file
        if not os.path.exists(model_path):
            with st.spinner("📥 Đang tải model file..."):
                success = download_from_gdrive(MODEL_FILE_ID, model_path)
                if not success:
                    st.stop()
        
        # Download dataframe file
        if not os.path.exists(df_path):
            with st.spinner("📥 Đang tải data file..."):
                success = download_from_gdrive(DF_FILE_ID, df_path)
                if not success:
                    st.stop()
        
        st.success("✅ Tải model thành công!")
    
    # Load model
    try:
        with st.spinner("⚙️ Đang load model..."):
            model = joblib.load(model_path)
            df = joblib.load(df_path)
            df = df.reset_index(drop=True)
            
            current_year = datetime.now().year
            df["registration_year"] = current_year - df["age"]
            
            return model, df
    except Exception as e:
        st.error(f"❌ Lỗi khi load model: {str(e)}")
        st.info("💡 Thử xóa cache và reload lại trang")
        st.stop()


def handle_multiselect_with_all(selected):
    """Xử lý logic 'Tất cả' trong multiselect"""
    if not selected:
        return ["Tất cả"]

    if "Tất cả" in selected and len(selected) > 1:
        if selected[-1] == "Tất cả":
            return ["Tất cả"]
        else:
            return [x for x in selected if x != "Tất cả"]

    return selected


def search_items(query, df, top_k=10):
    """Tìm kiếm xe theo query"""
    if len(df) == 0:
        return pd.DataFrame()

    if not query.strip():
        results = df.head(top_k).copy()
        results["position"] = results.index
        return results

    df["search_text"] = (
        df["brand"].fillna("")
        + " "
        + df["model"].fillna("")
        + " "
        + df["vehicle_type"].fillna("")
        + " "
        + df["description_norm"].fillna("")
    )

    try:
        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(df["search_text"])
        query_vec = vectorizer.transform([query])

        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]

        results = df.iloc[top_indices].copy()
        results["search_score"] = scores[top_indices]
        results["position"] = top_indices

        return results
    except:
        return pd.DataFrame()


def apply_filters(
    df, brands, models, price_range, vehicle_types, locations, engine_capacities
):
    """Áp dụng bộ lọc"""
    filtered = df.copy()

    if brands and "Tất cả" not in brands:
        filtered = filtered[filtered["brand"].isin(brands)]

    if models and "Tất cả" not in models:
        filtered = filtered[filtered["model"].isin(models)]

    if vehicle_types and "Tất cả" not in vehicle_types:
        filtered = filtered[filtered["vehicle_type"].isin(vehicle_types)]

    if locations and "Tất cả" not in locations:
        filtered = filtered[filtered["location"].isin(locations)]

    if engine_capacities and "Tất cả" not in engine_capacities:
        filtered = filtered[filtered["engine_capacity"].isin(engine_capacities)]

    if price_range[0] is not None and price_range[1] is not None:
        filtered = filtered[
            (filtered["price"] >= price_range[0])
            & (filtered["price"] <= price_range[1])
        ]

    return filtered


def get_recommendations(item_position, model, df, top_k=3):
    """Lấy xe tương tự"""
    sim_scores = model["similarity"][item_position].copy()
    sim_scores[item_position] = -10.0
    top_indices = sim_scores.argsort()[::-1][:top_k]

    recs = df.iloc[top_indices].copy()
    recs["similarity"] = sim_scores[top_indices]
    recs["position"] = top_indices

    return recs


def show_about_page():
    """Trang giới thiệu"""
    st.title("📖 Giới Thiệu Về Hệ Thống")

    st.markdown("---")

    # Mục đích
    st.markdown("## 🎯 Mục Đích")
    st.markdown("""
    Hệ thống **Buôn Bán Xe Máy** được xây dựng nhằm:
    
    - 🔍 **Tìm kiếm thông minh**: Giúp người dùng dễ dàng tìm kiếm xe máy phù hợp với nhu cầu
    - 🎯 **Gợi ý cá nhân hóa**: Đề xuất các xe tương tự dựa trên sở thích và lựa chọn của người dùng
    - 📊 **Lọc đa tiêu chí**: Hỗ trợ lọc theo nhiều tiêu chí như hãng xe, giá, khu vực, dung tích động cơ...
    - 💡 **Trải nghiệm tốt nhất**: Cung cấp giao diện thân thiện, dễ sử dụng cho mọi đối tượng người dùng
    """)

    st.markdown("---")

    # Tính năng chính
    st.markdown("## ✨ Tính Năng Chính")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🔎 Tìm Kiếm & Lọc
        - Tìm kiếm theo từ khóa tự do
        - Lọc theo hãng xe, model
        - Lọc theo loại xe, khu vực
        - Lọc theo dung tích động cơ
        - Lọc theo khoảng giá
        """)

        st.markdown("""
        ### 📋 Hiển Thị Thông Tin
        - Thông tin chi tiết từng xe
        - Giá cả, số km đã đi
        - Năm đăng ký, xuất xứ
        - Mô tả chi tiết sản phẩm
        """)

    with col2:
        st.markdown("""
        ### 🎯 Hệ Thống Gợi Ý
        - Gợi ý xe tương tự
        - Tính toán độ tương đồng
        - Đề xuất dựa trên đặc điểm xe
        - Cá nhân hóa trải nghiệm
        """)

        st.markdown("""
        ### 💻 Giao Diện Người Dùng
        - Thiết kế responsive
        - Dễ dàng điều hướng
        - Hiển thị trực quan
        - Tương tác mượt mà
        """)

    st.markdown("---")

    # Công nghệ
    st.markdown("## 🛠️ Công Nghệ Sử Dụng")

    st.markdown("""
    ### 📚 Thư Viện & Framework
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Frontend & UI**
        - 🎨 **Streamlit**: Framework web app
        - 📊 **Pandas**: Xử lý dữ liệu
        - 🔢 **NumPy**: Tính toán số học
        """)

    with col2:
        st.markdown("""
        **Machine Learning**
        - 🤖 **Scikit-learn**: Thuật toán ML
        - 📝 **TF-IDF**: Vector hóa văn bản
        - 📏 **Cosine Similarity**: Tính độ tương đồng
        """)

    with col3:
        st.markdown("""
        **Lưu Trữ & Xử Lý**
        - 💾 **Joblib**: Lưu/load model
        - 🗂️ **Glob**: Quản lý file
        - ⏰ **Datetime**: Xử lý thời gian
        """)

    st.markdown("---")

    # Thuật toán
    st.markdown("## 🧠 Thuật Toán Gợi Ý")

    st.markdown("""
    Hệ thống sử dụng **Content-Based Filtering** với các bước:
    
    1. **Vector hóa đặc điểm**: Chuyển đổi thông tin xe thành vector số
    2. **TF-IDF**: Trích xuất đặc điểm quan trọng từ mô tả và thông tin xe
    3. **Cosine Similarity**: Tính toán độ tương đồng giữa các xe
    4. **Ranking**: Sắp xếp và đề xuất xe có độ tương đồng cao nhất
    """)

    # Visualization of similarity
    st.info("""
    💡 **Ví dụ**: Khi bạn xem một chiếc Honda Wave Alpha, hệ thống sẽ tìm các xe có:
    - Cùng hãng hoặc phân khúc tương tự
    - Giá cả gần nhau
    - Dung tích động cơ tương đương
    - Đặc điểm kỹ thuật giống nhau
    """)

    st.markdown("---")

    # Thống kê
    st.markdown("## 📊 Thống Kê Hệ Thống")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🏍️ Tổng số xe", f"{len(df):,}")

    with col2:
        st.metric("🏢 Số hãng xe", f"{df['brand'].nunique()}")

    with col3:
        st.metric("🏷️ Số loại xe", f"{df['vehicle_type'].nunique()}")

    with col4:
        st.metric("📍 Số khu vực", f"{df['location'].nunique()}")

    st.markdown("---")

    # Hướng dẫn sử dụng
    st.markdown("## 📖 Hướng Dẫn Sử Dụng")

    with st.expander("🔍 Cách tìm kiếm xe"):
        st.markdown("""
        1. Nhập từ khóa vào ô tìm kiếm (tên xe, hãng, loại xe...)
        2. Sử dụng bộ lọc để thu hẹp kết quả
        3. Nhấn nút "Tìm kiếm" hoặc Enter
        4. Xem danh sách kết quả phù hợp
        """)

    with st.expander("🎯 Cách sử dụng bộ lọc"):
        st.markdown("""
        1. Mở rộng phần "Bộ Lọc Tìm Kiếm"
        2. Chọn các tiêu chí: Hãng xe, Model, Loại xe, Khu vực, Dung tích
        3. Điều chỉnh khoảng giá mong muốn
        4. Kết quả sẽ tự động cập nhật
        """)

    with st.expander("👁️ Cách xem chi tiết và xe tương tự"):
        st.markdown("""
        1. Nhấn nút "Xem chi tiết" trên xe bạn quan tâm
        2. Xem đầy đủ thông tin chi tiết của xe
        3. Cuộn xuống phần "Xe Tương Tự" để xem gợi ý
        4. Nhấn "Xem chi tiết" trên xe gợi ý để khám phá thêm
        """)

    st.markdown("---")

    # Call to action
    st.markdown("## 🚀 Bắt Đầu Ngay")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("🔍 Đi đến Trang Tìm Kiếm", use_container_width=True, type="primary"):
            st.session_state["page"] = "search"
            st.session_state["scroll_to_top"] = True
            st.rerun()

    st.markdown("---")

    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>💡 Được phát triển bởi Hoàng Phúc & Bích Thủy</p>
        <p>📧 Liên hệ hỗ trợ: phucthuy@buonbanxemay.vn</p>
    </div>
    """, unsafe_allow_html=True)


def show_search_page():
    """Trang tìm kiếm"""
    st.title("🏍️ Tìm Kiếm Xe Máy")

    # Filters section
    with st.expander("🔧 Bộ Lọc Tìm Kiếm", expanded=False):
        # Row 1: 4 main filters
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            all_brands = ["Tất cả"] + sorted(df["brand"].unique().tolist())
            selected_brands_raw = st.multiselect(
                "🏢 Hãng xe",
                options=all_brands,
                default=["Tất cả"],
                key="filter_brands",
            )
            selected_brands = handle_multiselect_with_all(selected_brands_raw)

        with col2:
            if selected_brands and "Tất cả" not in selected_brands:
                available_models = (
                    df[df["brand"].isin(selected_brands)]["model"].unique().tolist()
                )
            else:
                available_models = df["model"].unique().tolist()

            all_models = ["Tất cả"] + sorted(available_models)
            selected_models_raw = st.multiselect(
                "🏍️ Model xe",
                options=all_models,
                default=["Tất cả"],
                key="filter_models",
            )
            selected_models = handle_multiselect_with_all(selected_models_raw)

        with col3:
            all_vehicle_types = ["Tất cả"] + sorted(
                df["vehicle_type"].unique().tolist()
            )
            selected_vehicle_types_raw = st.multiselect(
                "🏷️ Loại xe",
                options=all_vehicle_types,
                default=["Tất cả"],
                key="filter_vehicle_types",
            )
            selected_vehicle_types = handle_multiselect_with_all(
                selected_vehicle_types_raw
            )

        with col4:
            all_locations = ["Tất cả"] + sorted(df["location"].unique().tolist())
            selected_locations_raw = st.multiselect(
                "📍 Khu vực",
                options=all_locations,
                default=["Tất cả"],
                key="filter_locations",
            )
            selected_locations = handle_multiselect_with_all(selected_locations_raw)

        st.markdown("---")

        # Row 2: Engine capacity and price range
        col5, col6, col7 = st.columns([2, 3, 1])

        with col5:
            all_engine_capacities = ["Tất cả"] + sorted(
                df["engine_capacity"].unique().tolist()
            )
            selected_engine_capacities_raw = st.multiselect(
                "⚙️ Dung tích",
                options=all_engine_capacities,
                default=["Tất cả"],
                key="filter_engine_capacities",
            )
            selected_engine_capacities = handle_multiselect_with_all(
                selected_engine_capacities_raw
            )

        with col6:
            col_price1, col_price2 = st.columns(2)
            with col_price1:
                min_price_input = st.number_input(
                    "💰 Giá từ (triệu)",
                    min_value=0.0,
                    max_value=float(df["price"].max()),
                    value=float(df["price"].min()),
                    step=1.0,
                    key="filter_min_price",
                    label_visibility="visible",
                )
            with col_price2:
                max_price_input = st.number_input(
                    "💰 Giá đến (triệu)",
                    min_value=0.0,
                    max_value=float(df["price"].max()),
                    value=float(df["price"].max()),
                    step=1.0,
                    key="filter_max_price",
                    label_visibility="visible",
                )

    st.markdown("---")

    # Search bar
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "🔍 Tìm kiếm xe",
            value="",
            placeholder="Nhập tên xe, hãng, loại xe...",
            key="search_query",
        )
    with col2:
        st.write("")
        st.write("")
        search_btn = st.button("Tìm kiếm", use_container_width=True, type="primary")

    # Xác định query để sử dụng
    if query:
        current_query = query
    elif st.session_state.get("last_query", ""):
        current_query = st.session_state["last_query"]
    else:
        current_query = ""

    # Áp dụng bộ lọc
    price_range = (min_price_input, max_price_input)
    filtered_df = apply_filters(
        df,
        selected_brands,
        selected_models,
        price_range,
        selected_vehicle_types,
        selected_locations,
        selected_engine_capacities,
    )

    # Tìm kiếm trong filtered_df
    if current_query:
        results = search_items(current_query, filtered_df, top_k=10)
    else:
        results = filtered_df.head(10).copy()
        results["position"] = results.index

    # Cập nhật last_query khi có query mới
    if query:
        st.session_state["last_query"] = query

    # Hiển thị query hiện tại đang được tìm kiếm
    if current_query:
        st.info(f"🔍 Đang tìm kiếm: **{current_query}**")

    # Kiểm tra nếu không có xe nào
    if len(results) == 0:
        st.warning(
            "⚠️ Không tìm thấy xe phù hợp. Vui lòng thử điều chỉnh bộ lọc hoặc từ khóa."
        )
        return

    st.session_state["search_results"] = results

    st.markdown("---")
    st.subheader(f"📋 Kết quả ({len(results)} xe)")

    for idx, row in results.iterrows():
        with st.container():
            col_a, col_b = st.columns([4, 1])

            with col_a:
                st.markdown(f"### {row['brand']} {row['model']}")

                st.markdown(
                    f"**💰 Giá:** {row['price']:.1f} triệu VNĐ | **📏 Số km đã đi:** {row['km_driven']:,} km | **📅 Năm đăng ký:** {int(row['registration_year'])}"
                )
                st.markdown(
                    f"**🏢 Thương hiệu:** {row['brand']} | **🏷️ Loại xe:** {row['vehicle_type']} | **⚙️ Dung tích:** {row['engine_capacity']}"
                )
                st.markdown(
                    f"**🌍 Xuất xứ:** {row['origin']} | **📍 Địa điểm:** {row['location']}"
                )

                if pd.notna(row["description_norm"]) and row["description_norm"]:
                    desc_short = (
                        row["description_norm"][:150] + "..."
                        if len(row["description_norm"]) > 150
                        else row["description_norm"]
                    )
                    st.markdown(f"**📝 Mô tả:** {desc_short}")

            with col_b:
                st.write("")
                st.write("")
                if st.button("Xem chi tiết", key=f"view_{int(row['position'])}_{idx}"):
                    st.session_state["page"] = "detail"
                    st.session_state["selected_position"] = int(row["position"])
                    st.session_state["scroll_to_top"] = True
                    st.rerun()

            st.markdown("---")


def show_detail_page():
    """Trang chi tiết xe"""
    item_position = st.session_state["selected_position"]

    if item_position < 0 or item_position >= len(df):
        st.error("Xe không tồn tại!")
        if st.button("← Quay lại"):
            st.session_state["page"] = "search"
            st.session_state["scroll_to_top"] = True
            st.rerun()
        return

    item = df.iloc[item_position]

    # Back button
    if st.button("← Quay lại tìm kiếm"):
        st.session_state["page"] = "search"
        st.session_state["scroll_to_top"] = True
        st.rerun()

    st.markdown("---")

    # Title
    st.title(f"{item['brand']} {item['model']}")

    # Main info card
    st.markdown("### 💳 Thông Tin Chính")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Giá bán", f"{item['price']:.1f} triệu VNĐ")
    col2.metric("📏 Số km đã đi", f"{item['km_driven']:,} km")
    col3.metric("📅 Năm đăng ký", f"{int(item['registration_year'])}")
    col4.metric("🏷️ Loại xe", item["vehicle_type"])

    st.markdown("---")

    # Detailed info
    st.markdown("### 📋 Thông Tin Chi Tiết")

    col_x, col_y = st.columns(2)

    with col_x:
        st.markdown(f"""
        - **🏢 Thương hiệu:** {item['brand']}
        - **🏍️ Model:** {item['model']}
        - **⚙️ Dung tích động cơ:** {item['engine_capacity']}
        """)

    with col_y:
        st.markdown(f"""
        - **🌍 Xuất xứ:** {item['origin']}
        - **📍 Địa điểm:** {item['location']}
        - **🏷️ Phân loại:** {item['vehicle_type']}
        """)

    st.markdown("---")

    # Description
    st.markdown("### 📝 Mô Tả Chi Tiết")
    if pd.notna(item["description_norm"]) and item["description_norm"]:
        st.write(item["description_norm"])
    else:
        st.info("Không có mô tả chi tiết")

    st.markdown("---")
    st.markdown("---")

    # Recommendations section
    st.markdown("## 🎯 Xe Tương Tự Bạn Có Thể Quan Tâm")
    st.markdown("")

    recs = get_recommendations(item_position, model, df, top_k=3)

    # Display as cards
    cols = st.columns(3)

    for i, (idx, row) in enumerate(recs.iterrows()):
        with cols[i]:
            with st.container():
                st.markdown(f"""
                <div style="
                    border: 2px solid #e0e0e0;
                    border-radius: 10px;
                    padding: 20px;
                    background-color: #f9f9f9;
                    height: 100%;
                ">
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"### {row['brand']} {row['model']}")

                st.markdown(f"**💰 Giá:** {row['price']:.1f} triệu VNĐ")
                st.markdown(f"**📏 Số km:** {row['km_driven']:,} km")
                st.markdown(f"**📅 Năm đăng ký:** {int(row['registration_year'])}")
                st.markdown(f"**🏢 Thương hiệu:** {row['brand']}")
                st.markdown(f"**⚙️ Dung tích:** {row['engine_capacity']}")
                st.markdown(f"**🌍 Xuất xứ:** {row['origin']}")
                st.markdown(f"**📍 Địa điểm:** {row['location']}")

                similarity_pct = row["similarity"] * 100
                st.markdown(f"""
                <div style="
                    background-color: #4CAF50;
                    color: white;
                    padding: 5px 10px;
                    border-radius: 5px;
                    text-align: center;
                    margin: 10px 0;
                ">
                    🎯 Độ tương đồng: {similarity_pct:.1f}%
                </div>
                """, unsafe_allow_html=True)

                if st.button(
                    "👁️ Xem chi tiết",
                    key=f"rec_{int(row['position'])}_{i}",
                    use_container_width=True,
                ):
                    st.session_state["selected_position"] = int(row["position"])
                    st.session_state["scroll_to_top"] = True
                    st.rerun()


# Load model
model, df = load_model()

# Initialize session state
if "page" not in st.session_state:
    st.session_state["page"] = "about"
if "selected_position" not in st.session_state:
    st.session_state["selected_position"] = None
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""
if "search_results" not in st.session_state:
    st.session_state["search_results"] = None
if "scroll_to_top" not in st.session_state:
    st.session_state["scroll_to_top"] = False

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🧭 Điều Hướng")

    if st.button(
        "📖 Giới Thiệu",
        use_container_width=True,
        type="primary" if st.session_state["page"] == "about" else "secondary",
    ):
        st.session_state["page"] = "about"
        st.session_state["scroll_to_top"] = True
        st.rerun()

    if st.button(
        "🔍 Tìm Kiếm",
        use_container_width=True,
        type="primary" if st.session_state["page"] == "search" else "secondary",
    ):
        st.session_state["page"] = "search"
        st.session_state["scroll_to_top"] = True
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Thống Kê Nhanh")
    st.metric("Tổng số xe", f"{len(df):,}")
    st.metric("Số hãng", f"{df['brand'].nunique()}")
    st.metric("Số loại xe", f"{df['vehicle_type'].nunique()}")

# Check if need to scroll to top
if st.session_state.get("scroll_to_top", False):
    scroll_to_top()
    st.session_state["scroll_to_top"] = False

# Route pages
if st.session_state["page"] == "about":
    show_about_page()
elif st.session_state["page"] == "search":
    show_search_page()
elif st.session_state["page"] == "detail":
    show_detail_page()

# Footer
st.markdown("---")
st.markdown(f"*Hệ thống gợi ý xe máy - Tổng số xe: {len(df):,}*")
