import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os
from huggingface_hub import hf_hub_download

# ==============================
# 🏗️ FEATURE BUILDER CLASS - KHỚP VỚI MODEL TRAIN
# ==============================
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class FeatureBuilder:
    """Class để xây dựng feature matrix - KHỚP 100% VỚI MODEL TRAIN"""
    
    def __init__(self):
        self.mm_scaler = MinMaxScaler()
        self.fitted = False
        self.expected_n_features = 8  # ✅ THÊM DÒNG NÀY
        
    def preprocess_df(self, df):
        """Tiền xử lý dataframe - KHỚP VỚI CODE TRAIN"""
        df_proc = df.copy()
        
        # ============================================
        # 1) BASIC CLEANING
        # ============================================
        df_proc['price'] = df_proc['price'].clip(1, 500)
        df_proc['km_driven'] = df_proc['km_driven'].clip(0, 200000)
        df_proc['age'] = df_proc['age'].clip(0, 30)
        
        # Fill missing values
        df_proc['price'] = df_proc['price'].fillna(df_proc['price'].median())
        df_proc['km_driven'] = df_proc['km_driven'].fillna(df_proc['km_driven'].median())
        df_proc['age'] = df_proc['age'].fillna(df_proc['age'].median())
        
        # ============================================
        # 2) ENGINE CC - VIETNAM SPECIFIC
        # ============================================
        VN_CC_DICT = {
            "exciter":150, "r15":155, "r3":321, "r25":250,
            "sirius":110, "jupiter":110, "nouvo":135, "janus":125,
            "latte":125, "grande":125, "mio":110,
            "wave":110, "future":125, "dream":100, "cub":50,
            "winner":150, "winner x":150,
            "vision":110, "lead":125, "sh mode":125,
            "air blade":125, "airblade":125,
            "sh":125, "vario":160, "pcx":125, "click":125,
            "vespa":125, "primavera":125, "sprint":125,
            "raider":150, "satria":150, "gsx":150,
            "attila":125, "shark":125, "hayate":125,
            "cb150":150, "cbr150":150, "cbr250":250, "cbr300":300,
            "rebel":300, "shadow":750,
            "mt15":155, "mt03":321, "mt07":689,
            "z300":300, "z650":650, "ninja":300,
            "duke":200, "rc":200,
        }
        
        def extract_engine_cc(row):
            """Extract engine CC từ model name"""
            model = str(row.get('model', '')).lower()
            is_pkl = bool(row.get('xe_pkl', 0))
            
            # Check dictionary first
            for key, cc in VN_CC_DICT.items():
                if key in model:
                    return cc
            
            # Regex fallback
            import re
            patterns = [
                r'\b(50|70|90|100|110|125|150|155|200|250|300|350|400|500|650|750|1000)\b',
                r'(?:cb|cbr|mt|gsx|ninja|duke|rc)[\s-]?(\d{2,4})'
            ]
            
            for pat in patterns:
                matches = re.findall(pat, model)
                if matches:
                    vals = [int(v) for v in matches if str(v).isdigit()]
                    if vals:
                        cc = max(vals)
                        if 50 <= cc <= 1200:
                            return cc
            
            # Default values
            return 300 if is_pkl else 125
        
        df_proc['engine_cc'] = df_proc.apply(extract_engine_cc, axis=1)
        
        # ============================================
        # 3) VEHICLE TYPE MAPPING
        # ============================================
        mapping_vehicle = {
            "Xe số": 0,
            "Tay ga": 1,
            "Tay côn/Moto": 2,
            "PKL": 2
        }
        df_proc['vehicle_type_num'] = df_proc['vehicle_type'].map(mapping_vehicle).fillna(1)
        
        # ============================================
        # 4) BOOLEAN FEATURES
        # ============================================
        for bcol in ['xe_pkl', 'xe_zin', 'xe_co', 'xe_da_thay_doi', 'xe_chinh_chu', 'xe_nang_cap']:
            if bcol not in df_proc.columns:
                df_proc[bcol] = 0
            df_proc[bcol] = df_proc[bcol].astype(int)
        
        # ============================================
        # 5) LOG TRANSFORMS
        # ============================================
        df_proc['log_km'] = np.log1p(df_proc['km_driven'])  # ← TÊN ĐÚNG: log_km
        df_proc['log_price'] = np.log1p(df_proc['price'])
        
        # ============================================
        # 6) DERIVED FEATURES
        # ============================================
        df_proc['km_per_year'] = df_proc['km_driven'] / (df_proc['age'] + 1)
        df_proc['log_km_per_year'] = np.log1p(df_proc['km_per_year'])
        
        # power_ratio = engine_cc / (price + 1)
        df_proc['power_ratio'] = df_proc['engine_cc'] / (df_proc['price'] + 1)
        
        # price_per_cc = price / (engine_cc + 1)
        df_proc['price_per_cc'] = df_proc['price'] / (df_proc['engine_cc'] + 1)
        
        # ============================================
        # 7) ENGINE CLASS (CATEGORICAL → NUMERIC)
        # ============================================
        bins = [0, 150, 300, 650, 2000]
        labels = [1, 2, 3, 4]
        df_proc['engine_class'] = pd.cut(
            df_proc['engine_cc'], 
            bins=bins, 
            labels=labels, 
            include_lowest=True
        )
        df_proc['engine_class'] = df_proc['engine_class'].fillna(1).astype(int)
        
        # ============================================
        # 8) PRICE MINMAX NORMALIZATION
        # ============================================
        # Fit scaler on first call
        if not self.fitted:
            self.mm_scaler.fit(df_proc[['price']])
            self.fitted = True
        
        df_proc['price_minmax'] = self.mm_scaler.transform(df_proc[['price']]).ravel()
        
        return df_proc
    
    def fit(self, df):
        """Fit (không cần thiết cho rule-based, nhưng giữ để tương thích)"""
        return self
    
    def transform(self, df):
        """Transform (đã làm trong preprocess_df)"""
        return df
    
    def fit_transform(self, df):
        """Fit và transform"""
        return self.preprocess_df(df)


    def build_feature_matrix(self, df):
        """
        Build feature matrix - AUTO-DETECT số features
        """
        
        # Thử các feature sets theo thứ tự
        feature_sets = {
            "BASE_6": [
                "log_price", "log_km", "age", 
                "log_km_per_year", "engine_cc", "vehicle_type_num"
            ],
            "V4_MINMAX_6": [
                "price_minmax", "log_km", "engine_cc",
                "engine_class", "vehicle_type_num", "power_ratio"
            ],
            "V4_BOOL_7": [
                "price_minmax", "log_km", "engine_cc", "engine_class",
                "vehicle_type_num", "power_ratio", "xe_pkl"
            ],
            "V4_BOOL_8": [
                "price_minmax", "log_km", "engine_cc", "engine_class",
                "vehicle_type_num", "power_ratio", "xe_pkl", "xe_zin"
            ]
        }
        
        # Lấy expected features từ model nếu có
        expected_n_features = getattr(self, 'expected_n_features', 8)
        
        # Chọn feature set phù hợp
        for name, feats in feature_sets.items():
            if len(feats) == expected_n_features:
                feature_names = feats
                # st.info(f"✅ Using feature set: {name} ({len(feats)} features)")
                break
        else:
            # Default fallback
            feature_names = feature_sets["V4_BOOL_7"]
            st.warning(f"⚠️ Using default: V4_BOOL_7")
        
        # Build features
        features = []
        for col in feature_names:
            if col in df.columns:
                values = df[col].values.reshape(-1, 1)
                values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                features.append(values)
            else:
                st.error(f"❌ Missing column: {col}")
                features.append(np.zeros((len(df), 1)))
        
        X = np.hstack(features)
        
        # st.info(f"📊 Built feature matrix: {X.shape}")
        
        return X

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
if os.path.exists("banner.jpg"):
    st.image("banner.jpg", use_column_width=True)
else:
    st.markdown(
        """
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px;'>
        <h1 style='color: white; margin: 0;'>🏍️ HỆ THỐNG BUÔN BÁN XE MÁY</h1>
        <p style='color: white; margin: 10px 0 0 0;'>Tìm kiếm và gợi ý xe máy thông minh</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)


# ==============================
# 📥 DOWNLOAD FROM HUGGING FACE
# ==============================
def download_from_huggingface(repo_id, filename, cache_dir="./model_cache"):
    """
    Download file từ Hugging Face Hub

    Args:
        repo_id: ID của repository trên Hugging Face (vd: "username/repo-name")
        filename: Tên file cần download
        cache_dir: Thư mục lưu cache

    Returns:
        str: Đường dẫn đến file đã download
    """
    try:
        # Tạo thư mục cache nếu chưa có
        os.makedirs(cache_dir, exist_ok=True)

        # Download file
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            resume_download=True,
        )

        return file_path

    except Exception as e:
        st.error(f"❌ Lỗi khi tải {filename}: {str(e)}")
        return None


@st.cache_resource
# Thêm đoạn này TRƯỚC hàm load_model() để test
def check_files_exist():
    """Kiểm tra các file có tồn tại không"""
    from huggingface_hub import list_repo_files
    
    REPO_ID = "Mayer1226/Recommendation"
    
    try:
        files = list_repo_files(repo_id=REPO_ID)
        st.write("📁 **Các file trong repository:**")
        for f in files:
            st.write(f"- {f}")
        return files
    except Exception as e:
        st.error(f"Lỗi: {e}")
        return []

# Gọi hàm này để kiểm tra
# if st.button("🔍 Kiểm tra files trên Hugging Face"):
#     check_files_exist()
@st.cache_resource(show_spinner=False)
def load_model():
    """Load model và dataframe từ Hugging Face - SỬ DỤNG CLUSTERING ML"""
    
    REPO_ID = "Mayer1226/Recommendation"
    MODEL_FILENAME = "model_v4_20251121_202731.joblib"
    DF_FILENAME = "df_items_20251121_202731.joblib"
    CLUSTER_FILENAME = "motorbike_cluster_model.joblib"
    
    try:
        with st.spinner("🔄 Đang tải dữ liệu từ Hugging Face..."):
            # Download files
            model_path = download_from_huggingface(REPO_ID, MODEL_FILENAME)
            df_path = download_from_huggingface(REPO_ID, DF_FILENAME)
            cluster_model_path = download_from_huggingface(REPO_ID, CLUSTER_FILENAME)
            
            if not all([model_path, df_path, cluster_model_path]):
                st.error("❌ Không thể tải đầy đủ files")
                st.stop()
            
            # Load models
            model = joblib.load(model_path)
            df = joblib.load(df_path)
            df = df.reset_index(drop=True)
            
            cluster_package = joblib.load(cluster_model_path)
            
            # st.success(f"✅ Loaded {len(df):,} xe và clustering model!")
            
            # ============================================
            # APPLY CLUSTERING WITH NEW FEATUREBUILDER
            # ============================================
            
            try:
                # st.info("🔄 Đang phân loại xe bằng ML clustering...")
                
                # Extract components
                cluster_scaler = cluster_package.get("scaler")
                cluster_kmeans = cluster_package.get("kmeans")
                cluster_labels = cluster_package.get("cluster_labels")
                
                # Create NEW FeatureBuilder (khớp với model train)
                cluster_feature_builder = FeatureBuilder()
                
                # Step 1: Preprocess
                df_proc = cluster_feature_builder.preprocess_df(df)
                
                # Step 2: Build features
                Xc = cluster_feature_builder.build_feature_matrix(df_proc)
                
                # Step 3: Validate
                expected_features = cluster_scaler.n_features_in_
                actual_features = Xc.shape[1]
                
                # st.info(f"📊 Features: {actual_features} (expected: {expected_features})")
                
                if actual_features != expected_features:
                    st.error(f"❌ Feature mismatch: {actual_features} vs {expected_features}")
                    
                    # Show details
                    with st.expander("🔍 Chi tiết features"):
                        st.write(f"**Actual shape:** {Xc.shape}")
                        st.write(f"**Expected:** {expected_features}")
                        st.write(f"**Sample values (first row):**")
                        st.code(Xc[0])
                    
                    # Fallback to rule-based
                    st.warning("⚠️ Sử dụng phân loại rule-based")
                    df = apply_rule_based_clustering(df)
                    
                else:
                    # Step 4: Transform và predict
                    Xc_scaled = cluster_scaler.transform(Xc)
                    df["cluster_id"] = cluster_kmeans.predict(Xc_scaled)
                    df["cluster_name"] = df["cluster_id"].map(cluster_labels)
                    
                    # Validate results
                    n_clusters = df["cluster_id"].nunique()
                    cluster_dist = df["cluster_name"].value_counts().to_dict()
                    
                    st.success(f"✅ ML Clustering thành công: {n_clusters} phân khúc!")
                    st.info(f"📊 Phân bố: {cluster_dist}")
                
            except Exception as cluster_error:
                # st.error(f"❌ Lỗi clustering: {str(cluster_error)}")
                
                # with st.expander("🔍 Chi tiết lỗi"):
                #     import traceback
                #     st.code(traceback.format_exc())
                
                # Fallback
                # st.warning("⚠️ Sử dụng phân loại rule-based")
                df = apply_rule_based_clustering(df)
            
            # ============================================
            # ADD METADATA
            # ============================================
            
            cluster_colors = {
                0: "#f94144",
                1: "#f3722c",
                2: "#f9c74f",
                3: "#90be6d",
                4: "#577590",
            }
            df["cluster_color"] = df["cluster_id"].map(cluster_colors).fillna("#667eea")
            
            current_year = datetime.now().year
            df["registration_year"] = current_year - df["age"]
            
            cluster_package["cluster_colors"] = cluster_colors
            cluster_package["feature_builder"] = cluster_feature_builder
            
            return model, df, cluster_package
            
    except Exception as e:
        st.error(f"❌ Lỗi khi load model: {str(e)}")
        
        # with st.expander("🔍 Chi tiết lỗi đầy đủ"):
        #     import traceback
        #     st.code(traceback.format_exc())
        
        st.stop()


def apply_rule_based_clustering(df):
    """Fallback: Rule-based clustering nếu ML fail"""
    
    def classify_motorbike(row):
        price = row['price']
        age = row['age']
        km = row['km_driven']
        vehicle_type = str(row.get('vehicle_type', ''))
        
        if price > 80 or 'PKL' in vehicle_type or 'Moto' in vehicle_type:
            return 4
        elif 25 <= price <= 80 and age <= 10:
            return 0
        elif km < 5000 and age >= 5:
            return 2
        elif age > 15 or (vehicle_type == 'Xe số' and price < 15):
            return 1
        else:
            return 3
    
    df['cluster_id'] = df.apply(classify_motorbike, axis=1)
    
    cluster_labels = {
        0: "Xe Phổ Thông Cao Cấp",
        1: "Xe Số Cũ – Kinh Tế",
        2: "Xe Ít Sử Dụng – Còn Mới",
        3: "Xe Phổ Thông – Đã Qua Sử Dụng",
        4: "Xe Cao Cấp & PKL"
    }
    
    df['cluster_name'] = df['cluster_id'].map(cluster_labels)
    
    return df


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
    st.markdown(
        """
    Hệ thống **Buôn Bán Xe Máy** được xây dựng nhằm:
    
    - 🔍 **Tìm kiếm thông minh**: Giúp người dùng dễ dàng tìm kiếm xe máy phù hợp với nhu cầu
    - 🎯 **Gợi ý cá nhân hóa**: Đề xuất các xe tương tự dựa trên sở thích và lựa chọn của người dùng
    - 🚀 **Phân cụm thông minh**: Tự động phân loại xe theo 5 phân khúc xe có đặc trưng dựa trên máy học.
    - 📊 **Lọc đa tiêu chí**: Hỗ trợ lọc theo nhiều tiêu chí như hãng xe, giá, khu vực, dung tích động cơ...
    - 💡 **Trải nghiệm tốt nhất**: Cung cấp giao diện thân thiện, dễ sử dụng cho mọi đối tượng người dùng
    """
    )

    st.markdown("---")

    # Tính năng chính
    st.markdown("## ✨ Tính Năng Chính")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        ### 🔎 Tìm Kiếm & Lọc
        - Tìm kiếm theo từ khóa tự do
        - Lọc theo hãng xe, model
        - Lọc theo loại xe, khu vực
        - Lọc theo dung tích động cơ
        - Lọc theo khoảng giá
        """
        )

        st.markdown(
            """
        ### 📋 Hiển Thị Thông Tin
        - Thông tin chi tiết từng xe
        - Giá cả, số km đã đi
        - Năm đăng ký, xuất xứ
        - Mô tả chi tiết sản phẩm
        - **Badge phân cụm màu sắc**
        """
        )

    with col2:
        st.markdown(
            """
        ### 🎯 Hệ Thống Gợi Ý
        - Gợi ý xe tương tự
        - Tính toán độ tương đồng
        - Đề xuất dựa trên đặc điểm xe
        - Cá nhân hóa trải nghiệm
        """
        )

        st.markdown(
            """
        ### 💻 Giao Diện Người Dùng
        - Thiết kế responsive
        - Dễ dàng điều hướng
        - Hiển thị trực quan
        - Tương tác mượt mà
        """
        )

    st.markdown("---")

    # ==============================
    # 🚀 PHẦN MỚI: PHÂN CỤM XE MÁY
    # ==============================
    st.markdown("## 🚀 Tính Năng Phân Cụm Xe Máy Thông Minh")
    
    st.markdown(
        """
        Hệ thống sử dụng **Machine Learning (K-Means Clustering)** để tự động phân loại 
        xe máy thành **5 phân khúc** dựa trên nhiều đặc điểm:
        """
    )

    # Hiển thị 5 cụm với màu sắc
    cluster_info = {
        0: {
            "name": "Xe Phổ Thông Cao Cấp",
            "color": "#f94144",
            "icon": "🏆",
            "description": "Xe phổ thông nhưng giá cao, chất lượng tốt, ít km đã đi",
            "examples": "Honda SH Mode, Yamaha Grande, Vespa Primavera"
        },
        1: {
            "name": "Xe Số Cũ – Kinh Tế",
            "color": "#f3722c",
            "icon": "💰",
            "description": "Xe số đã qua sử dụng lâu, giá rẻ, phù hợp sinh viên",
            "examples": "Honda Wave, Future cũ, Dream cũ"
        },
        2: {
            "name": "Xe Ít Sử Dụng – Còn Mới",
            "color": "#f9c74f",
            "icon": "✨",
            "description": "Xe đã qua sử dụng nhưng số km rất thấp, gần như mới",
            "examples": "Xe zin, chính chủ, ít đi"
        },
        3: {
            "name": "Xe Phổ Thông – Đã Qua Sử Dụng",
            "color": "#90be6d",
            "icon": "🛵",
            "description": "Xe phổ thông, giá trung bình, đã qua sử dụng vừa phải",
            "examples": "Air Blade, Vision, Lead đã qua sử dụng"
        },
        4: {
            "name": "Xe Cao Cấp & PKL",
            "color": "#577590",
            "icon": "🏍️",
            "description": "Xe phân khối lớn, moto cao cấp, giá trị cao",
            "examples": "Honda CBR, Yamaha R15, Kawasaki Ninja"
        }
    }

    # Hiển thị từng cụm
    for cluster_id, info in cluster_info.items():
        with st.expander(f"{info['icon']} **Cụm {cluster_id}: {info['name']}**", expanded=False):
            col_a, col_b = st.columns([1, 3])
            
            with col_a:
                st.markdown(
                    f"""
                    <div style="
                        background-color: {info['color']};
                        color: white;
                        padding: 30px;
                        border-radius: 10px;
                        text-align: center;
                        font-size: 40px;
                    ">
                        {info['icon']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col_b:
                st.markdown(f"**📝 Mô tả:** {info['description']}")
                st.markdown(f"**🏍️ Ví dụ:** {info['examples']}")
                
                # Thống kê số lượng xe trong cụm
                cluster_count = len(df[df['cluster_id'] == cluster_id])
                cluster_pct = (cluster_count / len(df)) * 100
                st.markdown(f"**📊 Số lượng:** {cluster_count:,} xe ({cluster_pct:.1f}%)")

    st.markdown("---")

    # Lợi ích của phân cụm
    st.markdown("### 💡 Lợi Ích Của Phân Cụm")
    
    col_benefit1, col_benefit2 = st.columns(2)
    
    with col_benefit1:
        st.markdown(
            """
            #### 👤 Cho Người Dùng
            
            - ✅ **Dễ dàng nhận biết**: Badge màu sắc giúp phân biệt nhanh phân khúc xe
            - ✅ **Tìm kiếm nhanh hơn**: Lọc theo nhóm xe phù hợp với nhu cầu
            - ✅ **So sánh dễ dàng**: Xe cùng cụm có đặc điểm tương đồng
            - ✅ **Gợi ý chính xác**: Hệ thống đề xuất xe trong cùng phân khúc
            - ✅ **Hiểu rõ giá trị**: Biết xe thuộc phân khúc nào để đánh giá giá
            """
        )
    
    with col_benefit2:
        st.markdown(
            """
            #### 🏢 Cho Quản Trị Viên
            
            - ✅ **Phân tích thị trường**: Hiểu rõ cơ cấu xe trên sàn
            - ✅ **Quản lý**: Theo dõi số lượng xe theo từng phân khúc
            - ✅ **Chiến lược giá**: Định giá dựa trên phân cụm tự động
            - ✅ **Marketing hiệu quả**: Nhắm đúng đối tượng khách hàng
            - ✅ **Báo cáo nhanh**: Thống kê theo nhóm xe dễ dàng
            """
        )

    st.markdown("---")

    # Công nghệ phân cụm
    st.markdown("### 🧠 Công Nghệ Phân Cụm")
    
    st.markdown(
        """
        #### 📊 Thuật Toán: K-Means Clustering
        
        Hệ thống sử dụng thuật toán **K-Means** với các bước:
        
        1. **Chuẩn hóa dữ liệu**: Sử dụng MinMaxScaler để đưa các đặc điểm về cùng thang đo
        2. **Trích xuất đặc điểm**: 8 features quan trọng:
           - `price_minmax`: Giá xe (đã chuẩn hóa)
           - `log_km`: Số km đã đi (log transform)
           - `engine_cc`: Dung tích động cơ
           - `engine_class`: Phân loại động cơ (1-4)
           - `vehicle_type_num`: Loại xe (số, tay ga, PKL)
           - `power_ratio`: Tỷ lệ công suất/giá
           - `xe_pkl`: Xe phân khối lớn (0/1)
           - `xe_zin`: Xe zin (0/1)
        
        3. **Phân cụm**: K-Means với k=5 tự động gom nhóm xe tương đồng
        4. **Gán nhãn**: Mỗi cụm được gán tên có ý nghĩa dựa trên đặc điểm trung bình
        5. **Màu sắc**: Mỗi cụm có màu riêng để dễ nhận biết
        """
    )

    # Visualization của phân cụm
    st.info(
        """
        💡 **Ví dụ thực tế**: 
        
        - Một chiếc **Honda SH 2020, giá 70 triệu, 5000km** → Cụm 0 (Xe Phổ Thông Cao Cấp) 🏆
        - Một chiếc **Wave Alpha 2010, giá 8 triệu, 50000km** → Cụm 1 (Xe Số Cũ – Kinh Tế) 💰
        - Một chiếc **Yamaha R15 2022, giá 90 triệu, 2000km** → Cụm 4 (Xe Cao Cấp & PKL) 🏍️
        """
    )

    st.markdown("---")

    # ==============================
    # KẾT THÚC PHẦN PHÂN CỤM
    # ==============================

    # Công nghệ
    st.markdown("## 🛠️ Công Nghệ Sử Dụng")

    st.markdown(
        """
    ### 📚 Thư Viện & Framework
    """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        **Frontend & UI**
        - 🎨 **Streamlit**: Framework web app
        - 📊 **Pandas**: Xử lý dữ liệu
        - 🔢 **NumPy**: Tính toán số học
        """
        )

    with col2:
        st.markdown(
            """
        **Machine Learning**
        - 🤖 **Scikit-learn**: Thuật toán ML
        - 🎯 **K-Means**: Phân cụm xe máy
        - 📝 **TF-IDF**: Vector hóa văn bản
        - 📏 **Cosine Similarity**: Tính độ tương đồng
        """
        )

    with col3:
        st.markdown(
            """
        **Lưu Trữ & Xử Lý**
        - 💾 **Joblib**: Lưu/load model
        - 🤗 **Hugging Face**: Cloud storage
        - ⏰ **Datetime**: Xử lý thời gian
        """
        )

    st.markdown("---")

    # Thuật toán
    st.markdown("## 🧠 Thuật Toán Gợi Ý")

    st.markdown(
        """
    Hệ thống sử dụng **Content-Based Filtering** kết hợp **Clustering** với các bước:
    
    1. **Phân cụm trước**: Gom nhóm xe theo 5 phân khúc bằng K-Means
    2. **Vector hóa đặc điểm**: Chuyển đổi thông tin xe thành vector số
    3. **TF-IDF**: Trích xuất đặc điểm quan trọng từ mô tả và thông tin xe
    4. **Cosine Similarity**: Tính toán độ tương đồng giữa các xe
    5. **Ranking**: Sắp xếp và đề xuất xe có độ tương đồng cao nhất (ưu tiên cùng cụm)
    """
    )

    # Visualization of similarity
    st.info(
        """
    💡 **Ví dụ**: Khi bạn xem một chiếc Honda Wave Alpha (Cụm 1 - Xe Số Cũ), hệ thống sẽ:
    
    1. **Ưu tiên** gợi ý xe trong cùng Cụm 1 (Future, Dream cũ...)
    2. Tìm xe có **đặc điểm tương tự**:
       - Cùng hãng hoặc phân khúc
       - Giá cả gần nhau
       - Dung tích động cơ tương đương
       - Đặc điểm kỹ thuật giống nhau
    3. Hiển thị **độ tương đồng** (%) để bạn dễ so sánh
    """
    )

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
        st.metric("🚀 Số phân cụm", "5")

    # Thống kê phân cụm
    st.markdown("### 📈 Phân Bố Theo Cụm")
    
    cluster_stats = df['cluster_name'].value_counts().sort_index()
    
    cols_stats = st.columns(5)
    for i, (cluster_name, count) in enumerate(cluster_stats.items()):
        with cols_stats[i]:
            cluster_id = df[df['cluster_name'] == cluster_name]['cluster_id'].iloc[0]
            color = cluster_info[cluster_id]['color']
            pct = (count / len(df)) * 100
            
            st.markdown(
                f"""
                <div style="
                    background-color: {color};
                    color: white;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                ">
                    <div style="font-size: 24px; font-weight: bold;">{count:,}</div>
                    <div style="font-size: 12px; margin-top: 5px;">{pct:.1f}%</div>
                    <div style="font-size: 10px; margin-top: 5px; opacity: 0.9;">{cluster_info[cluster_id]['icon']} Cụm {cluster_id}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # Hướng dẫn sử dụng
    st.markdown("## 📖 Hướng Dẫn Sử Dụng")

    with st.expander("🔍 Cách tìm kiếm xe"):
        st.markdown(
            """
        1. Nhập từ khóa vào ô tìm kiếm (tên xe, hãng, loại xe...)
        2. Sử dụng bộ lọc để thu hẹp kết quả
        3. **Chú ý badge màu sắc** để biết xe thuộc phân khúc nào
        4. Nhấn nút "Tìm kiếm" hoặc Enter
        5. Xem danh sách kết quả phù hợp
        """
        )

    with st.expander("🎯 Cách sử dụng bộ lọc"):
        st.markdown(
            """
        1. Mở rộng phần "Bộ Lọc Tìm Kiếm"
        2. Chọn các tiêu chí: Hãng xe, Model, Loại xe, Khu vực, Dung tích
        3. Điều chỉnh khoảng giá mong muốn
        4. Kết quả sẽ tự động cập nhật
        5. **Lưu ý**: Xe cùng màu badge thuộc cùng phân khúc
        """
        )

    with st.expander("👁️ Cách xem chi tiết và xe tương tự"):
        st.markdown(
            """
        1. Nhấn nút "Xem chi tiết" trên xe bạn quan tâm
        2. Xem **badge phân cụm** ở đầu trang để biết xe thuộc nhóm nào
        3. Xem đầy đủ thông tin chi tiết của xe
        4. Cuộn xuống phần "Xe Tương Tự" để xem gợi ý
        5. **Xe gợi ý ưu tiên cùng phân cụm** để đảm bảo phù hợp
        6. Nhấn "Xem chi tiết" trên xe gợi ý để khám phá thêm
        """
        )
    
    with st.expander("🚀 Hiểu về phân cụm xe"):
        st.markdown(
            """
        **Badge màu sắc** trên mỗi xe cho biết:
        
        - 🏆 **Đỏ đậm** (#f94144): Xe Phổ Thông Cao Cấp - Chất lượng tốt, giá cao
        - 💰 **Cam đậm** (#f3722c): Xe Số Cũ – Kinh Tế - Giá rẻ, đã qua sử dụng lâu
        - ✨ **Vàng** (#f9c74f): Xe Ít Sử Dụng – Còn Mới - Số km thấp, gần như mới
        - 🛵 **Xanh lá** (#90be6d): Xe Phổ Thông – Đã Qua Sử Dụng - Giá trung bình
        - 🏍️ **Xanh dương** (#577590): Xe Cao Cấp & PKL - Phân khối lớn, giá trị cao
        
        **Lợi ích**:
        - Nhận biết nhanh phân khúc xe
        - So sánh xe cùng nhóm dễ dàng
        - Đánh giá giá trị hợp lý
        - Tìm xe phù hợp với ngân sách
        """
        )

    st.markdown("---")

    # Call to action
    st.markdown("## 🚀 Bắt Đầu Ngay")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button(
            "🔍 Đi đến Trang Tìm Kiếm", use_container_width=True, type="primary"
        ):
            st.session_state["page"] = "search"
            st.session_state["scroll_to_top"] = True
            st.rerun()

    st.markdown("---")

    # Footer
    st.markdown(
        """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>💡 Được phát triển bởi Hoàng Phúc & Bích Thủy</p>
        <p>🚀 Tích hợp Machine Learning Clustering cho phân loại thông minh</p>
        <p>📧 Liên hệ hỗ trợ: phucthuy@buonbanxemay.vn</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

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
                
                # Cluster badge - CHỈ HIỂN THỊ MỘT LẦN
                st.markdown(
                    f"""
                    <span style="
                        background-color:{row['cluster_color']};
                        color:white;
                        padding:5px 10px;
                        border-radius:5px;
                        display:inline-block;
                        margin-bottom:10px;">
                        🚀 {row['cluster_name']}
                    </span>
                    """,
                    unsafe_allow_html=True,
                )

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
    
    # Cluster badge
    st.markdown(
        f"""
        <div style="
            background-color:{item['cluster_color']};
            display:inline-block;
            color:white;
            padding:8px 15px;
            border-radius:6px;
            font-weight:bold;
            margin-top:5px;
            margin-bottom:15px;">
            🚀 Thuộc cụm: {item['cluster_name']}
        </div>
        """,
        unsafe_allow_html=True,
    )

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
        st.markdown(
            f"""
        - **🏢 Thương hiệu:** {item['brand']}
        - **🏍️ Model:** {item['model']}
        - **⚙️ Dung tích động cơ:** {item['engine_capacity']}
        """
        )

    with col_y:
        st.markdown(
            f"""
        - **🌍 Xuất xứ:** {item['origin']}
        - **📍 Địa điểm:** {item['location']}
        - **🏷️ Phân loại:** {item['vehicle_type']}
        """
        )

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
                st.markdown(
                    f"""
                <div style="
                    border: 2px solid #e0e0e0;
                    border-radius: 10px;
                    padding: 20px;
                    background-color: #f9f9f9;
                    height: 100%;
                ">
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.markdown(f"### {row['brand']} {row['model']}")

                st.markdown(f"**💰 Giá:** {row['price']:.1f} triệu VNĐ")
                st.markdown(f"**📏 Số km:** {row['km_driven']:,} km")
                st.markdown(f"**📅 Năm đăng ký:** {int(row['registration_year'])}")
                st.markdown(f"**🏢 Thương hiệu:** {row['brand']}")
                st.markdown(f"**⚙️ Dung tích:** {row['engine_capacity']}")
                st.markdown(f"**🌍 Xuất xứ:** {row['origin']}")
                st.markdown(f"**📍 Địa điểm:** {row['location']}")

                similarity_pct = row["similarity"] * 100
                st.markdown(
                    f"""
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
                """,
                    unsafe_allow_html=True,
                )

                if st.button(
                    "👁️ Xem chi tiết",
                    key=f"rec_{int(row['position'])}_{i}",
                    use_container_width=True,
                ):
                    st.session_state["selected_position"] = int(row["position"])
                    st.session_state["scroll_to_top"] = True
                    st.rerun()

# ==============================
# 📊 TRANG QUẢN TRỊ VIÊN - SỬA LỖI
# ==============================
def show_admin_page():
    """Trang quản trị viên - Phân tích và quản lý"""
    
    # Header với gradient
    st.markdown(
        """
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;'>
            <h1 style='color: white; margin: 0;'>👨‍💼 BẢNG ĐIỀU KHIỂN QUẢN TRỊ</h1>
            <p style='color: white; margin: 10px 0 0 0;'>Phân tích dữ liệu và quản lý hệ thống</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # ==============================
    # 📊 SECTION 1: TỔNG QUAN HỆ THỐNG
    # ==============================
    st.markdown("## 📊 Tổng Quan Hệ Thống")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🏍️ Tổng số xe",
            f"{len(df):,}",
            delta=None
        )
    
    with col2:
        st.metric(
            "🏢 Số hãng xe",
            f"{df['brand'].nunique()}",
            delta=None
        )
    
    with col3:
        avg_price = df['price'].mean()
        st.metric(
            "💰 Giá TB",
            f"{avg_price:.1f}M",
            delta=None
        )
    
    with col4:
        avg_km = df['km_driven'].mean()
        st.metric(
            "📏 Km TB",
            f"{avg_km:,.0f}",
            delta=None
        )
    
    with col5:
        st.metric(
            "🚀 Phân cụm",
            "5",
            delta=None
        )
    
    st.markdown("---")
    
    # ==============================
    # 📈 SECTION 2: PHÂN TÍCH PHÂN CỤM
    # ==============================
    st.markdown("## 🚀 Phân Tích Phân Cụm")
    
    # ✅ KIỂM TRA CÁC CỘT TỒN TẠI
    agg_dict = {
        'price': ['mean', 'min', 'max', 'count'],
        'km_driven': 'mean',
        'age': 'mean'
    }
    
    # Thêm engine_capacity nếu có (thay vì engine_cc)
    if 'engine_capacity' in df.columns:
        # Chuyển đổi engine_capacity sang số nếu cần
        df['engine_capacity_num'] = df['engine_capacity'].str.extract('(\d+)').astype(float)
        agg_dict['engine_capacity_num'] = 'mean'
    
    # Thống kê theo cụm
    cluster_stats = df.groupby('cluster_id').agg(agg_dict).round(2)
    
    # Đặt tên cột
    if 'engine_capacity_num' in agg_dict:
        cluster_stats.columns = ['Giá TB', 'Giá Min', 'Giá Max', 'Số lượng', 'Km TB', 'Tuổi TB', 'CC TB']
    else:
        cluster_stats.columns = ['Giá TB', 'Giá Min', 'Giá Max', 'Số lượng', 'Km TB', 'Tuổi TB']
    
    cluster_stats = cluster_stats.reset_index()
    
    # Thêm tên cụm và màu
    cluster_labels = {
        0: "Xe Phổ Thông Cao Cấp",
        1: "Xe Số Cũ – Kinh Tế",
        2: "Xe Ít Sử Dụng – Còn Mới",
        3: "Xe Phổ Thông – Đã Qua Sử Dụng",
        4: "Xe Cao Cấp & PKL"
    }
    
    cluster_colors = {
        0: "#f94144",
        1: "#f3722c",
        2: "#f9c74f",
        3: "#90be6d",
        4: "#577590",
    }
    
    cluster_stats['Tên cụm'] = cluster_stats['cluster_id'].map(cluster_labels)
    cluster_stats['Màu'] = cluster_stats['cluster_id'].map(cluster_colors)
    
    # Hiển thị bảng với màu sắc
    st.markdown("### 📋 Bảng Thống Kê Chi Tiết")
    
    for idx, row in cluster_stats.iterrows():
        with st.expander(f"🚀 Cụm {row['cluster_id']}: {row['Tên cụm']} ({row['Số lượng']:.0f} xe)", expanded=False):
            col_a, col_b = st.columns([1, 3])
            
            with col_a:
                st.markdown(
                    f"""
                    <div style="
                        background-color: {row['Màu']};
                        color: white;
                        padding: 40px;
                        border-radius: 10px;
                        text-align: center;
                        font-size: 50px;
                        font-weight: bold;
                    ">
                        {row['cluster_id']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col_b:
                # ✅ KIỂM TRA CỘT CC TB
                if 'CC TB' in row:
                    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                else:
                    col_b1, col_b2, col_b3 = st.columns(3)
                
                with col_b1:
                    st.metric("💰 Giá TB", f"{row['Giá TB']:.1f}M")
                    st.metric("📉 Giá Min", f"{row['Giá Min']:.1f}M")
                
                with col_b2:
                    st.metric("📈 Giá Max", f"{row['Giá Max']:.1f}M")
                    st.metric("🏍️ Số lượng", f"{row['Số lượng']:.0f}")
                
                with col_b3:
                    st.metric("📏 Km TB", f"{row['Km TB']:,.0f}")
                    st.metric("📅 Tuổi TB", f"{row['Tuổi TB']:.1f} năm")
                
                if 'CC TB' in row:
                    with col_b4:
                        st.metric("⚙️ CC TB", f"{row['CC TB']:.0f}cc")
                        pct = (row['Số lượng'] / len(df)) * 100
                        st.metric("📊 Tỷ lệ", f"{pct:.1f}%")
                else:
                    # Hiển thị tỷ lệ ở cột 3
                    with col_b3:
                        pct = (row['Số lượng'] / len(df)) * 100
                        st.metric("📊 Tỷ lệ", f"{pct:.1f}%")
    
    st.markdown("---")
    
    # ==============================
    # 📊 SECTION 3: BIỂU ĐỒ PHÂN TÍCH
    # ==============================
    st.markdown("## 📊 Biểu Đồ Phân Tích")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Phân bố cụm", "💰 Phân tích giá", "🏢 Thương hiệu", "📍 Khu vực"])
    
    with tab1:
        st.markdown("### 📈 Phân Bố Xe Theo Cụm")
        
        # Tính toán phân bố
        cluster_distribution = df['cluster_name'].value_counts()
        
        # Hiển thị dạng bar chart bằng HTML/CSS
        for cluster_name, count in cluster_distribution.items():
            cluster_id = df[df['cluster_name'] == cluster_name]['cluster_id'].iloc[0]
            color = cluster_colors[cluster_id]
            pct = (count / len(df)) * 100
            
            st.markdown(
                f"""
                <div style="margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span><strong>{cluster_name}</strong></span>
                        <span><strong>{count:,} xe ({pct:.1f}%)</strong></span>
                    </div>
                    <div style="
                        width: 100%;
                        background-color: #e0e0e0;
                        border-radius: 5px;
                        overflow: hidden;
                    ">
                        <div style="
                            width: {pct}%;
                            background-color: {color};
                            padding: 10px;
                            color: white;
                            text-align: center;
                            font-weight: bold;
                        ">
                            {pct:.1f}%
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Insights
        st.info(
            f"""
            💡 **Insights:**
            - Cụm có nhiều xe nhất: **{cluster_distribution.index[0]}** ({cluster_distribution.values[0]:,} xe)
            - Cụm có ít xe nhất: **{cluster_distribution.index[-1]}** ({cluster_distribution.values[-1]:,} xe)
            - Phân bố tương đối {'đều' if cluster_distribution.std() < 500 else 'không đều'}
            """
        )
    
    with tab2:
        st.markdown("### 💰 Phân Tích Giá Theo Cụm")
        
        # Tạo bảng so sánh giá
        price_comparison = df.groupby('cluster_name')['price'].agg(['mean', 'min', 'max', 'median']).round(2)
        price_comparison.columns = ['Giá TB', 'Giá Min', 'Giá Max', 'Giá Median']
        price_comparison = price_comparison.sort_values('Giá TB', ascending=False)
        
        st.dataframe(
            price_comparison.style.background_gradient(cmap='RdYlGn_r', subset=['Giá TB']),
            use_container_width=True
        )
        
        # Phân tích khoảng giá
        st.markdown("#### 📊 Phân Bố Theo Khoảng Giá")
        
        price_ranges = {
            "< 10M": len(df[df['price'] < 10]),
            "10-20M": len(df[(df['price'] >= 10) & (df['price'] < 20)]),
            "20-40M": len(df[(df['price'] >= 20) & (df['price'] < 40)]),
            "40-80M": len(df[(df['price'] >= 40) & (df['price'] < 80)]),
            "> 80M": len(df[df['price'] >= 80])
        }
        
        col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
        
        for col, (range_name, count) in zip([col_p1, col_p2, col_p3, col_p4, col_p5], price_ranges.items()):
            with col:
                pct = (count / len(df)) * 100
                st.metric(range_name, f"{count:,}", f"{pct:.1f}%")
        
        # Insights
        max_range = max(price_ranges, key=price_ranges.get)
        st.info(
            f"""
            💡 **Insights:**
            - Khoảng giá phổ biến nhất: **{max_range}** ({price_ranges[max_range]:,} xe)
            - Giá trung bình toàn hệ thống: **{df['price'].mean():.1f}M VNĐ**
            - Giá cao nhất: **{df['price'].max():.1f}M VNĐ**
            - Giá thấp nhất: **{df['price'].min():.1f}M VNĐ**
            """
        )
    
    with tab3:
        st.markdown("### 🏢 Phân Tích Thương Hiệu")
        
        # Top 10 thương hiệu
        top_brands = df['brand'].value_counts().head(10)
        
        st.markdown("#### 🏆 Top 10 Thương Hiệu")
        
        for idx, (brand, count) in enumerate(top_brands.items(), 1):
            pct = (count / len(df)) * 100
            avg_price = df[df['brand'] == brand]['price'].mean()
            
            st.markdown(
                f"""
                <div style="
                    background-color: #f0f0f0;
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 10px;
                    border-left: 5px solid #667eea;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="font-size: 18px;">#{idx}. {brand}</strong>
                            <div style="color: #666; margin-top: 5px;">
                                {count:,} xe ({pct:.1f}%) | Giá TB: {avg_price:.1f}M VNĐ
                            </div>
                        </div>
                        <div style="
                            background-color: #667eea;
                            color: white;
                            padding: 10px 20px;
                            border-radius: 5px;
                            font-weight: bold;
                        ">
                            {count:,}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Phân tích theo cụm
        st.markdown("#### 📊 Thương Hiệu Theo Cụm")
        
        brand_cluster = pd.crosstab(df['brand'], df['cluster_name'])
        top_brand_cluster = brand_cluster.loc[top_brands.index[:5]]
        
        st.dataframe(
            top_brand_cluster.style.background_gradient(cmap='Blues'),
            use_container_width=True
        )
        
        st.info(
            f"""
            💡 **Insights:**
            - Thương hiệu phổ biến nhất: **{top_brands.index[0]}** ({top_brands.values[0]:,} xe)
            - Tổng số thương hiệu: **{df['brand'].nunique()}**
            - Thương hiệu có giá TB cao nhất: **{df.groupby('brand')['price'].mean().idxmax()}**
            """
        )
    
    with tab4:
        st.markdown("### 📍 Phân Tích Khu Vực")
        
        # Top 10 khu vực
        top_locations = df['location'].value_counts().head(10)
        
        st.markdown("#### 🗺️ Top 10 Khu Vực")
        
        col_l1, col_l2 = st.columns(2)
        
        for idx, (location, count) in enumerate(top_locations.items(), 1):
            pct = (count / len(df)) * 100
            avg_price = df[df['location'] == location]['price'].mean()
            
            col = col_l1 if idx % 2 == 1 else col_l2
            
            with col:
                st.markdown(
                    f"""
                    <div style="
                        background-color: #f9f9f9;
                        padding: 12px;
                        border-radius: 6px;
                        margin-bottom: 10px;
                        border: 1px solid #e0e0e0;
                    ">
                        <strong>#{idx}. {location}</strong><br>
                        <span style="color: #666;">
                            {count:,} xe ({pct:.1f}%)<br>
                            Giá TB: {avg_price:.1f}M VNĐ
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Phân tích theo cụm
        st.markdown("#### 📊 Khu Vực Theo Cụm")
        
        location_cluster = df.groupby(['location', 'cluster_name']).size().unstack(fill_value=0)
        top_location_cluster = location_cluster.loc[top_locations.index[:5]]
        
        st.dataframe(
            top_location_cluster.style.background_gradient(cmap='Greens'),
            use_container_width=True
        )
        
        st.info(
            f"""
            💡 **Insights:**
            - Khu vực có nhiều xe nhất: **{top_locations.index[0]}** ({top_locations.values[0]:,} xe)
            - Tổng số khu vực: **{df['location'].nunique()}**
            - Khu vực có giá TB cao nhất: **{df.groupby('location')['price'].mean().idxmax()}**
            """
        )
    
    st.markdown("---")
    
    # ==============================
    # 🔍 SECTION 4: TÌM KIẾM & LỌC NÂNG CAO
    # ==============================
    st.markdown("## 🔍 Tìm Kiếm & Lọc Nâng Cao")
    
    with st.expander("🔧 Bộ Lọc Quản Trị", expanded=False):
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            filter_cluster = st.multiselect(
                "🚀 Chọn cụm",
                options=["Tất cả"] + list(cluster_labels.values()),
                default=["Tất cả"],
                key="admin_filter_cluster"
            )
        
        with col_f2:
            filter_brand = st.multiselect(
                "🏢 Chọn thương hiệu",
                options=["Tất cả"] + sorted(df['brand'].unique().tolist()),
                default=["Tất cả"],
                key="admin_filter_brand"
            )
        
        with col_f3:
            filter_location = st.multiselect(
                "📍 Chọn khu vực",
                options=["Tất cả"] + sorted(df['location'].unique().tolist()),
                default=["Tất cả"],
                key="admin_filter_location"
            )
        
        col_f4, col_f5 = st.columns(2)
        
        with col_f4:
            filter_price_min = st.number_input(
                "💰 Giá từ (triệu)",
                min_value=0.0,
                max_value=float(df['price'].max()),
                value=0.0,
                key="admin_filter_price_min"
            )
        
        with col_f5:
            filter_price_max = st.number_input(
                "💰 Giá đến (triệu)",
                min_value=0.0,
                max_value=float(df['price'].max()),
                value=float(df['price'].max()),
                key="admin_filter_price_max"
            )
    
    # Áp dụng filter
    filtered_admin_df = df.copy()
    
    if "Tất cả" not in filter_cluster:
        filtered_admin_df = filtered_admin_df[filtered_admin_df['cluster_name'].isin(filter_cluster)]
    
    if "Tất cả" not in filter_brand:
        filtered_admin_df = filtered_admin_df[filtered_admin_df['brand'].isin(filter_brand)]
    
    if "Tất cả" not in filter_location:
        filtered_admin_df = filtered_admin_df[filtered_admin_df['location'].isin(filter_location)]
    
    filtered_admin_df = filtered_admin_df[
        (filtered_admin_df['price'] >= filter_price_min) &
        (filtered_admin_df['price'] <= filter_price_max)
    ]
    
    # Hiển thị kết quả
    st.markdown(f"### 📋 Kết Quả Lọc: {len(filtered_admin_df):,} xe")
    
    if len(filtered_admin_df) > 0:
        # Tùy chọn hiển thị
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        
        with col_opt1:
            sort_by = st.selectbox(
                "Sắp xếp theo",
                ["price", "km_driven", "age", "registration_year"],
                format_func=lambda x: {
                    "price": "Giá",
                    "km_driven": "Số km",
                    "age": "Tuổi xe",
                    "registration_year": "Năm đăng ký"
                }[x],
                key="admin_sort_by"
            )
        
        with col_opt2:
            sort_order = st.selectbox(
                "Thứ tự",
                ["Giảm dần", "Tăng dần"],
                key="admin_sort_order"
            )
        
        with col_opt3:
            show_limit = st.number_input(
                "Hiển thị",
                min_value=10,
                max_value=100,
                value=20,
                step=10,
                key="admin_show_limit"
            )
        
        # Sắp xếp
        ascending = sort_order == "Tăng dần"
        display_df = filtered_admin_df.sort_values(by=sort_by, ascending=ascending).head(show_limit)
        
        # ✅ KIỂM TRA CÁC CỘT TỒN TẠI TRƯỚC KHI HIỂN THỊ
        display_columns = ['brand', 'model', 'price', 'km_driven', 'age', 'vehicle_type', 'location', 'cluster_name']
        
        # Lọc chỉ các cột tồn tại
        available_columns = [col for col in display_columns if col in display_df.columns]
        
        # Hiển thị bảng
        st.dataframe(
            display_df[available_columns].rename(columns={
                'brand': 'Hãng',
                'model': 'Model',
                'price': 'Giá (M)',
                'km_driven': 'Km',
                'age': 'Tuổi',
                'vehicle_type': 'Loại',
                'location': 'Khu vực',
                'cluster_name': 'Cụm'
            }),
            use_container_width=True,
            height=400
        )
        
        # Export data
        st.markdown("#### 💾 Xuất Dữ Liệu")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            csv = display_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Tải xuống CSV",
                data=csv,
                file_name=f"motorbike_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_export2:
            # Summary stats
            if st.button("📊 Xem thống kê tóm tắt", use_container_width=True):
                st.write("**Thống kê dữ liệu đã lọc:**")
                numeric_cols = ['price', 'km_driven', 'age']
                available_numeric = [col for col in numeric_cols if col in display_df.columns]
                st.write(display_df[available_numeric].describe())
    
    else:
        st.warning("⚠️ Không có dữ liệu phù hợp với bộ lọc")
    
    st.markdown("---")
    
    # ==============================
    # ⚙️ SECTION 5: CÀI ĐẶT HỆ THỐNG
    # ==============================
    # st.markdown("## ⚙️ Cài Đặt Hệ Thống")
    
    # col_set1, col_set2 = st.columns(2)
    
    # with col_set1:
    #     st.markdown("### 🔄 Cập Nhật Dữ Liệu")
        
    #     if st.button("🔄 Làm mới dữ liệu", use_container_width=True):
    #         st.cache_resource.clear()
    #         st.success("✅ Đã làm mới dữ liệu!")
    #         st.rerun()
        
    #     st.markdown("### 📊 Thông Tin Model")
    #     st.info(
    #         f"""
    #         - **Số features:** 8
    #         - **Thuật toán:** K-Means Clustering
    #         - **Số cụm:** 5
    #         - **Similarity:** Cosine Similarity
    #         """
    #     )
    
    # with col_set2:
    #     st.markdown("### 📈 Hiệu Suất Hệ Thống")
        
    #     col_perf1, col_perf2 = st.columns(2)
        
    #     with col_perf1:
    #         st.metric("Số xe", f"{len(df):,}")
    #         st.metric("Số cụm", "5")
        
    #     with col_perf2:
    #         st.metric("Thương hiệu", f"{df['brand'].nunique()}")
    #         st.metric("Khu vực", f"{df['location'].nunique()}")
        
    #     st.markdown("### 🕐 Thời Gian")
    #     st.info(f"**Cập nhật lần cuối:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    st.markdown("---")
    
    # Footer
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>👨‍💼 Bảng điều khiển quản trị viên</p>
            <p>🔒 Chỉ dành cho người quản trị hệ thống</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Load model
model, df, cluster_model = load_model()

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
# ==============================
# SIDEBAR NAVIGATION - CẬP NHẬT
# ==============================
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
    
    # ✅ THÊM NÚT QUẢN TRỊ
    if st.button(
        "👨‍💼 Quản Trị",
        use_container_width=True,
        type="primary" if st.session_state["page"] == "admin" else "secondary",
    ):
        st.session_state["page"] = "admin"
        st.session_state["scroll_to_top"] = True
        st.rerun()

    # Phần thống kê và thông tin tác giả giữ nguyên...

    # st.markdown("---")
    # st.markdown("### 📊 Thống Kê Nhanh")
    # st.metric("Tổng số xe", f"{len(df):,}")
    # st.metric("Số hãng", f"{df['brand'].nunique()}")
    # st.metric("Số dòng xe", f"{df['model'].nunique()}")
    
    # ==============================
    # 👥 THÔNG TIN TÁC GIẢ & PHÁT HÀNH
    # ==============================
    st.markdown("---")
    st.markdown(
        """
        <div style='
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px;
            border-radius: 10px;
            color: white;
            text-align: center;
        '>
            <h4 style='margin: 0 0 10px 0; color: white;'>👥 Tác Giả</h4>
            <p style='margin: 5px 0; font-size: 14px;'>
                <strong>Hoàng Phúc & Bích Thủy</strong>
            </p>
            <hr style='border: 1px solid rgba(255,255,255,0.3); margin: 10px 0;'>
            <p style='margin: 5px 0; font-size: 13px;'>
                📅 <strong>Ngày phát hành:</strong><br>22/11/2025
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Check if need to scroll to top
if st.session_state.get("scroll_to_top", False):
    scroll_to_top()
    st.session_state["scroll_to_top"] = False


# ==============================
# 🔧 DEBUG HELPER
# ==============================
# if st.sidebar.checkbox("🔧 Debug Mode"):
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("### 🔍 Debug Info")
    
#     st.sidebar.write(f"**DF Shape:** {df.shape}")
#     st.sidebar.write(f"**Clusters:** {df['cluster_id'].nunique()}")
    
#     # Test feature building
#     if st.sidebar.button("Test Feature Matrix"):
#         try:
#             fb = FeatureBuilder()
#             test_df = df.head(5)
#             df_proc = fb.preprocess_df(test_df)
#             X_test = fb.build_feature_matrix(df_proc)
            
#             st.sidebar.success(f"✅ Shape: {X_test.shape}")
#             st.sidebar.write("**Feature names:**")
#             st.sidebar.code([
#                 "price_minmax", "log_km", "engine_cc", "engine_class",
#                 "vehicle_type_num", "power_ratio", "xe_pkl", "xe_zin"
#             ])
#             st.sidebar.write("**Sample row:**")
#             st.sidebar.code(X_test[0])
            
#         except Exception as e:
#             st.sidebar.error(f"❌ Error: {e}")


# ==============================
# ROUTE PAGES - CẬP NHẬT
# ==============================
if st.session_state["page"] == "about":
    show_about_page()
elif st.session_state["page"] == "search":
    show_search_page()
elif st.session_state["page"] == "detail":
    show_detail_page()
elif st.session_state["page"] == "admin":  # ✅ THÊM ROUTE MỚI
    show_admin_page()

# Footer
st.markdown("---")
st.markdown(f"*Hệ thống gợi ý xe máy - Tổng số xe: {len(df):,}*")

