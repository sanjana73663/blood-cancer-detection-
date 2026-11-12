# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import time
import sqlite3
from fpdf import FPDF
import hashlib
import pandas as pd
from datetime import datetime
import io

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Blood Cancer Detection", page_icon="🩸", layout="centered")

DB_PATH = "patients.db"
MODEL_PATH = "cnn_blood_cancer_model.h5"

# ---------------------------
# UTIL: DB
# ---------------------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS patients(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        phone TEXT,
        email TEXT,
        age TEXT,
        blood_group TEXT,
        result TEXT,
        confidence REAL,
        timestamp TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password_hash TEXT,
        role TEXT
    )
    """)
    conn.commit()

    # Create default accounts if not exist
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        # default passwords: admin/admin123, doctor/doctor123 (change in production)
        add_user("admin", "admin123", "admin")
        add_user("doctor", "doctor123", "doctor")
    conn.close()

def add_user(username, password, role):
    conn = get_conn()
    cur = conn.cursor()
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    try:
        cur.execute("INSERT INTO users(username, password_hash, role) VALUES (?, ?, ?)",
                    (username, password_hash, role))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()

def verify_user(username, password):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT password_hash, role FROM users WHERE username = ?", (username,))
    r = cur.fetchone()
    conn.close()
    if not r:
        return False, None
    password_hash, role = r
    return (hashlib.sha256(password.encode()).hexdigest() == password_hash), role

def save_patient(name, phone, email, age, blood_group, result, confidence):
    conn = get_conn()
    cur = conn.cursor()
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("""
        INSERT INTO patients(name, phone, email, age, blood_group, result, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (name, phone, email, age, blood_group, result, float(confidence), ts))
    conn.commit()
    conn.close()

def fetch_patients(query=None):
    conn = get_conn()
    cur = conn.cursor()
    if query:
        cur.execute(query)
    else:
        cur.execute("SELECT * FROM patients ORDER BY timestamp DESC")
    rows = cur.fetchall()
    conn.close()
    cols = ["id","name","phone","email","age","blood_group","result","confidence","timestamp"]
    return pd.DataFrame(rows, columns=cols)

def delete_patient(record_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM patients WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

init_db()

# ---------------------------
# MODEL LOAD
# ---------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error("Model load error: " + str(e))
        model = None
    return model

model = load_model()

# ---------------------------
# AUTH (session)
# ---------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None

def do_login(username, password):
    ok, role = verify_user(username, password)
    if ok:
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.role = role
        st.success(f"Logged in as {username} ({role})")
    else:
        st.error("Invalid credentials")

def do_logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None
    st.experimental_rerun()

# ---------------------------
# PDF generator
# ---------------------------
def generate_pdf(name, phone, email, blood_group, result_text, confidence_score):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)

    pdf.cell(200, 10, txt="Blood Cancer Detection Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Phone: {phone}", ln=True)
    pdf.cell(200, 10, txt=f"Email: {email}", ln=True)
    pdf.cell(200, 10, txt=f"Blood Group: {blood_group}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Result: {result_text}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence_score}%", ln=True)
    pdf.ln(5)
    pdf.cell(200, 8, txt=f"Generated at (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf

# ---------------------------
# PAGE: AUTH / HEADER
# ---------------------------
st.markdown("<h1 style='text-align:center;color:#003366;'>🩸 Blood Cancer Detection System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;color:#990000;'>AI-powered microscopic blood cell analysis</h4>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([3,1])
with col2:
    if st.session_state.authenticated:
        st.write(f"**{st.session_state.username}** ({st.session_state.role})")
        if st.button("Logout"):
            do_logout()
    else:
        with st.expander("Login (admin / doctor)"):
            usr = st.text_input("Username", key="login_user")
            pwd = st.text_input("Password", type="password", key="login_pwd")
            if st.button("Login", key="login_btn"):
                do_login(usr, pwd)

# ---------------------------
# MAIN LAYOUT: If not logged in, show public UI (but require login for admin/doctor pages)
# ---------------------------
# Patient form (public)
st.header("📋 Patient Information & Detection")
with st.form("patient_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    name = col1.text_input("Full Name", key="name")
    phone = col2.text_input("Phone Number", key="phone")
    email = col1.text_input("Email", key="email")
    age = col2.text_input("Age", key="age")
    blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], key="bg")

    submitted = st.form_submit_button("Save Details (you can still upload image below)")

if submitted:
    st.success("Patient details saved in form — upload image to run detection.")

# Image upload / prediction
st.markdown("### 📤 Upload Blood Smear Image")
uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

result_text = ""
confidence_score = 0.0

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Sample", use_container_width=True)

        with st.spinner("Analyzing the blood smear image..."):
            time.sleep(1)

            img_resized = img.resize((128,128))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            if model is None:
                st.error("Model not loaded — prediction unavailable.")
            else:
                prediction = model.predict(img_array)[0][0]
                confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
                confidence_score = round(confidence, 2)
                if prediction > 0.5:
                    st.error("⚠️ Leukemia Detected")
                    result_text = "Leukemia Detected"
                else:
                    st.success("✅ Healthy Sample")
                    result_text = "Healthy Sample"

                st.progress(int(min(confidence_score, 100)))
                st.write(f"**Confidence:** {confidence_score:.2f}%")

                # Save to DB if at least name or phone is provided
                if name or phone:
                    save_patient(name, phone, email, age, blood_group, result_text, confidence_score)
                    st.success("🗂️ Patient record saved to database!")

                # Recommendation
                st.markdown("---")
                if result_text == "Leukemia Detected":
                    st.warning("🩺 **Recommendation:** Please consult a hematologist or oncologist immediately.")
                else:
                    st.info("😄 **Recommendation:** No signs of leukemia detected. Stay healthy!")

                # PDF Download
                if name:
                    pdf_buf = generate_pdf(name, phone, email, blood_group, result_text, confidence_score)
                    st.download_button("⬇️ Download Report as PDF", data=pdf_buf, file_name="Blood_Cancer_Report.pdf", mime="application/pdf")
                # Share text box
                share_text = f"Blood Cancer Detection Result:\nPatient: {name}\nResult: {result_text}\nConfidence: {confidence_score}%"
                st.text_area("Share Result", share_text, height=120)

    except Exception as e:
        st.error("Error processing image: " + str(e))

st.markdown("---")

# ---------------------------
# ADMIN / DOCTOR PAGES (require login)
# ---------------------------
if st.session_state.authenticated and st.session_state.role in ("admin", "doctor"):
    st.header("🔒 Protected: Admin / Doctor Panel")

    tabs = st.tabs(["Patient Records", "Analytics", "Manage Users (admin only)"])
    # -------- Patient Records Tab --------
    with tabs[0]:
        st.subheader("Patient Records")
        df = fetch_patients()
        st.write(f"Total records: {len(df)}")

        # filters
        colA, colB, colC = st.columns([1,1,2])
        with colA:
            filter_result = st.selectbox("Filter by result", options=["All","Leukemia Detected","Healthy Sample"])
        with colB:
            date_from = st.date_input("From date", value=None)
            date_to = st.date_input("To date", value=None)
        with colC:
            search_q = st.text_input("Search name/phone/email")

        qdf = df.copy()
        if filter_result != "All":
            qdf = qdf[qdf["result"] == filter_result]
        # date filter (if not default)
        try:
            if date_from:
                qdf = qdf[qdf["timestamp"] >= pd.to_datetime(str(date_from)).strftime("%Y-%m-%d")]
            if date_to:
                qdf = qdf[qdf["timestamp"] <= pd.to_datetime(str(date_to)).strftime("%Y-%m-%d 23:59:59")]
        except Exception:
            pass
        if search_q:
            qdf = qdf[qdf["name"].str.contains(search_q, case=False, na=False) |
                      qdf["phone"].astype(str).str.contains(search_q, na=False) |
                      qdf["email"].str.contains(search_q, case=False, na=False)]

        st.dataframe(qdf.reset_index(drop=True))

        # Delete records (admin only)
        if st.session_state.role == "admin":
            st.markdown("**Admin actions**")
            del_id = st.number_input("Enter record ID to delete (or 0 to skip)", min_value=0, step=1, value=0)
            if st.button("Delete Record"):
                if del_id > 0:
                    delete_patient(del_id)
                    st.success(f"Deleted record {del_id}")
                    st.rerun()

                else:
                    st.info("No record deleted.")

        # Export CSV
        csv_buf = qdf.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export visible records as CSV", data=csv_buf, file_name="patient_records.csv", mime="text/csv")

    # -------- Analytics Tab --------
    with tabs[1]:
        st.subheader("Progress Analytics")
        df_all = fetch_patients()
        if df_all.empty:
            st.info("No patient records to analyze yet.")
        else:
            # Result distribution
            dist = df_all["result"].value_counts()
            st.write("Result distribution:")
            st.bar_chart(dist)

            # Daily counts (by date, UTC)
            try:
                df_all["date"] = pd.to_datetime(df_all["timestamp"]).dt.date
                daily = df_all.groupby("date").size()
                st.write("Daily submissions:")
                st.line_chart(daily)
            except Exception:
                st.info("Not enough timestamp data for daily chart.")

            # Average confidence by result
            avg_conf = df_all.groupby("result")["confidence"].mean().round(2)
            st.write("Average confidence by result:")
            st.table(avg_conf)

    # -------- Manage Users Tab (admin only) --------
    with tabs[2]:
        st.subheader("User Management")
        if st.session_state.role != "admin":
            st.info("Only admin users can manage accounts.")
        else:
            # show users
            conn = get_conn()
            users_df = pd.read_sql_query("SELECT id, username, role FROM users", conn)
            conn.close()
            st.dataframe(users_df)

            st.markdown("**Create new user**")
            new_user = st.text_input("Username", key="nu_user")
            new_pass = st.text_input("Password", type="password", key="nu_pass")
            new_role = st.selectbox("Role", ["doctor","admin"], key="nu_role")
            if st.button("Create user"):
                if new_user and new_pass:
                    add_user(new_user, new_pass, new_role)
                    st.success(f"User {new_user} added.")
                    st.experimental_rerun()
                else:
                    st.error("Provide username and password.")

else:
    st.info("Login as admin or doctor to access the admin/doctor features. Default accounts: (admin/admin123), (doctor/doctor123).")
    st.caption("You can still use the patient form and upload images without logging in.")

st.markdown("---")
st.info("⚠️ Disclaimer: This is a research/demo tool and not a medical diagnostic system.")
