import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import base64
# Set background image
# -------------------------
def set_background(image_file):
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("mobilephonebg.png")

# ---------------------------
# Load model and dataset
# ---------------------------
model = joblib.load("el_model.pkl")  # Your trained Elastic Net model
train_data = pd.read_csv("Mobile Phone Pricing.csv")  # Used to fit scaler

# ---------------------------
# Feature Engineering Function
# ---------------------------
def feature_engineering(df):
    # Outlier capping for fc and px_height
    for col in ['fc', 'px_height']:
        UE = df[col].quantile(0.99)
        LE = df[col].quantile(0.01)
        df[col] = np.where(df[col] > UE, UE, df[col])
        df[col] = np.where(df[col] < LE, LE, df[col])

    # Derived features
    df["pixel_area"] = df["px_height"] * df["px_width"]
    df["screen_ratio"] = df["sc_w"] / (df["sc_h"] + 1e-5)
    df["total_camera_mp"] = df["fc"] + df["pc"]

    def combine_3g_4g(row):
        if row["three_g"] == 1 and row["four_g"] == 1:
            return 1
        elif row["three_g"] == 1 and row["four_g"] == 0:
            return 2
        else:
            return 0
    df["network_type"] = df.apply(combine_3g_4g, axis=1)

    # Drop raw columns no longer needed
    df.drop(columns=["three_g", "four_g", "sc_w", "sc_h", "fc", "pc", "px_height", "px_width"], inplace=True)

    return df

# ---------------------------
# Fit scaler on training data
# ---------------------------
X_train = train_data.drop(columns="price_range")
X_train_processed = feature_engineering(X_train.copy())

# ---------------------------
# Streamlit UI
# ---------------------------
st.markdown(
    "<h1 style='color: #1E90FF; text-align: left;'>📱 Mobile Phone Price Range Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='color: #FFFFFF; text-align: left;'>Enter mobile specifications to predict the price range</h3>",
    unsafe_allow_html=True
)


st.sidebar.header("📥 Input Mobile Specifications")

def user_input_features():
    battery_power = st.sidebar.number_input("Battery Power (mAh)", 500, 2000, 1520)
    blue = int(st.sidebar.selectbox("Bluetooth", ("1", "0")))
    clock_speed = st.sidebar.slider("Clock Speed (GHz)", 0.5, 3.0, 2.2)
    dual_sim = int(st.sidebar.selectbox("Dual SIM", ("0", "1")))
    fc = st.sidebar.number_input("Front Camera (MP)", 0, 20, 5)
    four_g = int(st.sidebar.selectbox("4G", ("1", "0")))
    int_memory = st.sidebar.slider("Internal Memory (GB)", 1, 64, 33)
    m_dep = st.sidebar.slider("Mobile Depth (cm)", 0.0, 1.0, 0.5)
    mobile_wt = st.sidebar.number_input("Mobile Weight (grams)", 80, 200, 177)
    n_cores = int(st.sidebar.selectbox("Number of Cores", tuple(range(1, 9))))
    pc = st.sidebar.number_input("Primary Camera (MP)", 0, 20, 18)
    px_height = st.sidebar.number_input("Pixel Height", 0, 1960, 151)
    px_width = st.sidebar.number_input("Pixel Width", 300, 2000, 1005)
    ram = st.sidebar.number_input("RAM (MB)", 256, 4000, 3826)
    sc_h = st.sidebar.number_input("Screen Height (cm)", 1, 20, 14)
    sc_w = st.sidebar.number_input("Screen Width (cm)", 1, 20, 9)
    talk_time = st.sidebar.number_input("Talk Time (hours)", 2, 20, 13)
    three_g = int(st.sidebar.selectbox("3G", ("1", "0")))
    touch_screen = int(st.sidebar.selectbox("Touch Screen", ("1", "0")))
    wifi = int(st.sidebar.selectbox("WiFi", ("1", "0")))

    data = {
        "battery_power": battery_power,
        "blue": blue,
        "clock_speed": clock_speed,
        "dual_sim": dual_sim,
        "fc": fc,
        "four_g": four_g,
        "int_memory": int_memory,
        "m_dep": m_dep,
        "mobile_wt": mobile_wt,
        "n_cores": n_cores,
        "pc": pc,
        "px_height": px_height,
        "px_width": px_width,
        "ram": ram,
        "sc_h": sc_h,
        "sc_w": sc_w,
        "talk_time": talk_time,
        "three_g": three_g,
        "touch_screen": touch_screen,
        "wifi": wifi
    }
    return pd.DataFrame([data])

# ---------------------------
# Main App
# ---------------------------
input_df = user_input_features()
st.subheader("🔍 User Input")
st.write(input_df)

# Feature engineering + scaling
processed_df = feature_engineering(input_df.copy())

# Predict
if st.button("Predict Price Range"):
    prediction = model.predict(processed_df)[0]
    prediction_proba = model.predict_proba(processed_df)[0]

    label_map = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
    st.success(f"💡 Predicted Price Range: **{label_map[prediction]}**")
        # Prepare engineered training dataset for comparisons
    engineered_train = feature_engineering(train_data.drop(columns="price_range").copy())
    engineered_train["price_range"] = train_data["price_range"]

    # --- Dashboard Style: 2 Rows × 2 Columns ---
    st.subheader("📊 Feature Dashboard")

    dashboard_features = ["ram", "battery_power", "talk_time", "total_camera_mp"]

    # Average values for predicted class
    avg_pred_class = engineered_train[engineered_train["price_range"] == prediction][dashboard_features].mean()

    # Your phone's values
    your_values = processed_df.iloc[0][dashboard_features]

    for row in range(0, len(dashboard_features), 2):
        cols = st.columns(2)
        for col_idx, feature in enumerate(dashboard_features[row:row+2]):
            with cols[col_idx]:
                fig_feat, ax_feat = plt.subplots(figsize=(3.5, 2.5))
                ax_feat.bar(["Avg in Class", "Your Phone"], 
                            [avg_pred_class[feature], your_values[feature]],
                            color=["#1E90FF", "#4CAF50"], alpha=0.8)
                ax_feat.set_title(feature)
                ax_feat.set_ylabel("Value")
                st.pyplot(fig_feat) 
