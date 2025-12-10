# timeline-app
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Timeline + RandomForest Threat Predictor", layout="wide")

st.title("🛡️ Threat-Level Predictor + Interactive Timeline")
st.write(
    "هذا التطبيق يطبّق خوارزمية **Random Forest** للتنبؤ بمستوى التهديد، "
    "يستخدم **NumPy** لتنظيم المعطيات، و**Matplotlib** لرسم الخط الزمني والرسوم البيانية."
)

# Sidebar - data upload or generate sample
st.sidebar.header("بيانات")
data_source = st.sidebar.radio("اختر مصدر البيانات:", ("Upload CSV", "Generate Sample"))

if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("ارفع ملف CSV يحتوي صفوف بيانات (features) وعمود الهدف 'threat_level' أو اختر العمود بنفسك", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = None
else:
    # Generate sample dataset
    st.sidebar.info("تم إنشاء مجموعة بيانات تجريبية تحتوي على تواريخ، بعض الميزات، ومستوى التهديد (0=low,1=medium,2=high).")
    n = st.sidebar.slider("عدد الصفوف (عينة)", 100, 5000, 800)
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n).to_series().sample(frac=1, random_state=42).reset_index(drop=True)
    feature_1 = rng.normal(loc=0.0, scale=1.0, size=n) + (np.linspace(-1,1,n))
    feature_2 = rng.integers(0, 100, size=n)
    feature_3 = rng.normal(loc=5.0, scale=2.0, size=n)
    # create correlated target
    scores = 0.3*feature_1 + 0.02*feature_2 + 0.1*feature_3 + rng.normal(0,0.5,n)
    thresholds = np.quantile(scores, [0.33, 0.66])
    threat_level = np.digitize(scores, thresholds)
    df = pd.DataFrame({
        "date": dates,
        "feature_1": feature_1,
        "feature_2": feature_2,
        "feature_3": feature_3,
        "threat_level": threat_level
    })
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)

if df is None:
    st.info("انتظري رفع ملف CSV أو اختاري توليد مجموعة بيانات تجريبية من الشريط الجانبي.")
    st.stop()

st.subheader("معاينة البيانات")
st.dataframe(df.head())

# Select columns
all_columns = df.columns.tolist()
st.sidebar.subheader("إعدادات النموذج")
target_col = st.sidebar.selectbox("اختر عمود الهدف (target):", options=all_columns, index=all_columns.index("threat_level") if "threat_level" in all_columns else 0)
date_col = st.sidebar.selectbox("اختر عمود التاريخ (اختياري):", options=[None] + all_columns, index=1 if "date" in all_columns else 0)
# Features selected automatically as numeric columns except target and date
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
default_features = [c for c in numeric_cols if c != target_col]
features = st.sidebar.multiselect("اختر ميزات (features) لاستخدامها في التدريب:", options=default_features, default=default_features)

if len(features) == 0:
    st.error("مهم: اختاري على الأقل ميزة واحدة لتدريب النموذج.")
    st.stop()

# Preprocessing
st.subheader("معالجة البيانات وتحويلها باستخدام NumPy")
st.write("نقوم بتنظيف القيم الفارغة وتحويل إلى مصفوفات NumPy للحوسبة السريعة.")

# Drop rows where target is missing
before_rows = df.shape[0]
df_clean = df.dropna(subset=[target_col])
after_rows = df_clean.shape[0]
st.write(f"عدد الصفوف قبل التنظيف: {before_rows} → بعد حذف القيم المفقودة في عمود الهدف: {after_rows}")

# For selected features, fill missing with median
for col in features:
    if df_clean[col].isnull().any():
        median = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median)

# Convert feature matrix and target vector to NumPy arrays
X = df_clean[features].to_numpy(dtype=float)
y = df_clean[target_col].to_numpy(dtype=int)

st.write(f"شكل مصفوفة الميزات X: {X.shape} — شكل المتجه الهدف y: {y.shape}")

# Train / Test split
test_size = st.sidebar.slider("نسبة الاختبار (test size)", 5, 50, 20) / 100.0
random_state = st.sidebar.number_input("Random state (لإعادة التكرار)", min_value=1, max_value=9999, value=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state), stratify=y if len(np.unique(y))>1 else None)

st.subheader("تدريب نموذج Random Forest")
n_estimators = st.sidebar.slider("عدد الأشجار (n_estimators)", 10, 500, 100)
max_depth = st.sidebar.slider("الحد الأقصى لعمق الشجرة (max_depth) — 0 يعني بدون حد", 0, 50, 0)
max_depth_val = None if max_depth == 0 else int(max_depth)

clf = RandomForestClassifier(n_estimators=int(n_estimators), max_depth=max_depth_val, random_state=int(random_state))
with st.spinner("جارٍ تدريب النموذج..."):
    clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.success(f"تم التدريب! دقة النموذج على مجموعة الاختبار: {acc:.4f}")

st.subheader("تقرير الأداء")
st.text("Classification report:")
st.text(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix:")
st.write(pd.DataFrame(cm, index=np.unique(y), columns=np.unique(y)))

# Feature importance
st.subheader("أهمية الميزات (Feature Importance)")
importances = clf.feature_importances_
fi_df = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False)
st.table(fi_df)

# Matplotlib visualizations - must use plt (no seaborn)
st.subheader("الرسوم البيانية والـ Timeline باستخدام Matplotlib")

# 1) Feature importance bar chart
fig1, ax1 = plt.subplots(figsize=(6,3))
ax1.bar(fi_df["feature"], fi_df["importance"])
ax1.set_title("Feature Importances")
ax1.set_ylabel("Importance")
ax1.set_xlabel("Feature")
plt.xticks(rotation=30)
st.pyplot(fig1)

# 2) Confusion matrix heatmap (matplotlib)
fig2, ax2 = plt.subplots(figsize=(5,4))
cax = ax2.matshow(cm, cmap='viridis')
fig2.colorbar(cax)
ax2.set_title("Confusion Matrix")
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
ax2.set_xticks(range(len(np.unique(y))))
ax2.set_yticks(range(len(np.unique(y))))
st.pyplot(fig2)

# 3) Timeline plot: requires a date column selected
if date_col and date_col in df_clean.columns:
    st.subheader("الخط الزمني — متوسط مستوى التهديد عبر الزمن")
    # ensure date column is datetime
    dates = pd.to_datetime(df_clean[date_col])
    timeline_df = pd.DataFrame({
        "date": dates,
        "threat": df_clean[target_col]
    })
    timeline_df = timeline_df.sort_values("date")
    timeline_df.set_index("date", inplace=True)
    daily_mean = timeline_df["threat"].resample("D").mean().fillna(method="ffill").fillna(0)
    fig3, ax3 = plt.subplots(figsize=(10,3))
    ax3.plot(daily_mean.index, daily_mean.values, marker='o', linewidth=1)
    ax3.set_title("متوسط مستوى التهديد — يومياً")
    ax3.set_ylabel("متوسط تهديد")
    ax3.set_xlabel("التاريخ")
    plt.xticks(rotation=25)
    st.pyplot(fig3)
else:
    st.info("لم تحددي عمود التاريخ، لذلك لا يمكن رسم الخط الزمني. يمكنك رفع ملف يحتوي عمود تاريخ واختياره من الشريط الجانبي.")

# Allow user to run predictions on new rows
st.subheader("تجربة التنبؤ على بيانات جديدة")
st.write("أدخلي قيم ميزات جديدة لاختبار النموذج (سيتم التنبؤ بمستوى التهديد).")
input_vals = {}
cols = st.columns(len(features))
for i, f in enumerate(features):
    with cols[i]:
        input_vals[f] = st.number_input(f"قيمة {f}", value=float(df_clean[f].median()))
input_arr = np.array([input_vals[f] for f in features], dtype=float).reshape(1, -1)
pred = clf.predict(input_arr)[0]
st.write(f"التنبؤ بمستوى التهديد للمدخلات: **{pred}**")

# Offer download of model results (predictions on whole dataset)
st.subheader("تنزيل التنبؤات على مجموعة البيانات")
if st.button("حسّن التنبوءات وأحفظ CSV"):
    preds_all = clf.predict(df_clean[features].to_numpy())
    out_df = df_clean.copy()
    out_df["predicted_threat"] = preds_all
    csv = out_df.to_csv(index=False).encode('utf-8')
    st.download_button("تحميل CSV مع التنبؤات", data=csv, file_name="predictions.csv", mime="text/csv")

st.markdown("---")
st.caption("تم التطوير باستخدام RandomForest (scikit-learn)، NumPy للمعالجة، وMatplotlib للرسوم. يمكنك رفع ملفك الخاص أو تجربة العينة المولدة.")
