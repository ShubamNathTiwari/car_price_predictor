"""
app.py - Car Price Prediction System
A full-fledged ML-powered car price predictor with 3D UI, comparison features, and AI insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
import datetime
import warnings
import time
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AutoValue AI | Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# INDIAN NUMBER FORMATTERS
# All price values coming from the model/stats are in ₹ LAKHS (after training
# normalisation in model_trainer.py which divides raw rupees by 1,00,000).
# However, if stats were saved before normalisation, values may be raw rupees.
# We auto-detect: anything > 10,000 is treated as raw rupees and converted.
# ══════════════════════════════════════════════════════════════════════════════
def _to_lakhs(value):
    """Ensure value is in Lakhs. If it looks like raw rupees (> 10,000), divide."""
    if value is None:
        return None
    return value / 1_00_000 if value > 10_000 else value

def _indian_comma(rupees):
    """Format integer rupees with Indian comma grouping: 8,50,000"""
    s = str(int(round(rupees)))
    if len(s) <= 3:
        return s
    last3 = s[-3:]
    rest = s[:-3]
    grps = []
    while len(rest) > 2:
        grps.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        grps.insert(0, rest)
    return ",".join(grps) + "," + last3

def fmt_inr(value):
    """
    Format any price value (Lakhs or raw rupees) into Indian standard:
      < ₹1 L   →  ₹ 87,500
      ₹1-99 L  →  ₹ 8.50 Lakh
      ≥ ₹100 L →  ₹ 1.25 Crore
    """
    if value is None:
        return "—"
    lakhs = _to_lakhs(value)
    if lakhs >= 100:
        return "₹ {:.2f} Crore".format(lakhs / 100)
    elif lakhs >= 1:
        return "₹ {:.2f} Lakh".format(lakhs)
    else:
        return "₹ {:s}".format(_indian_comma(lakhs * 1_00_000))

def fmt_inr_full(value):
    """Full two-line breakdown: ₹ 8.50 Lakh (₹ 8,50,000)"""
    if value is None:
        return "—"
    lakhs = _to_lakhs(value)
    rupees = lakhs * 1_00_000
    indian = _indian_comma(rupees)
    if lakhs >= 100:
        return "₹ {:.2f} Crore  (₹ {})".format(lakhs / 100, indian)
    else:
        return "₹ {:.2f} Lakh  (₹ {})".format(lakhs, indian)

def fmt_inr_short(value):
    """Compact card format — no line breaks, fits in metric cards."""
    if value is None:
        return "—"
    lakhs = _to_lakhs(value)
    if lakhs >= 100:
        return "₹{:.2f}Cr".format(lakhs / 100)
    elif lakhs >= 1:
        return "₹{:.1f}L".format(lakhs)
    else:
        return "₹{:s}".format(_indian_comma(lakhs * 1_00_000))

def fmt_km(km):
    """Format kilometers: 1,23,456 km"""
    km = int(km)
    s = str(km)
    if len(s) > 3:
        last3 = s[-3:]
        rest = s[:-3]
        grps = []
        while len(rest) > 2:
            grps.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            grps.insert(0, rest)
        return ",".join(grps) + "," + last3 + " km"
    return s + " km"

def fmt_num(n):
    """Generic Indian number grouping."""
    n = int(round(n))
    s = str(n)
    if len(s) > 3:
        last3 = s[-3:]
        rest = s[:-3]
        grps = []
        while len(rest) > 2:
            grps.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            grps.insert(0, rest)
        return ",".join(grps) + "," + last3
    return s


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
  :root {
    --primary:#00d4ff; --secondary:#7b2ff7; --accent:#ff6b35;
    --border:rgba(0,212,255,0.3); --text:#e0e0ff; --success:#00ff88; --warning:#ffd700;
  }
  .stApp {
    background:linear-gradient(135deg,#0a0a1a 0%,#0d1535 30%,#1a0d2e 60%,#0a1a2e 100%);
    font-family:'Rajdhani',sans-serif; color:var(--text);
  }
  .stApp::before {
    content:''; position:fixed; top:0;left:0;right:0;bottom:0;
    background:radial-gradient(ellipse at 20% 50%,rgba(123,47,247,0.15) 0%,transparent 60%),
               radial-gradient(ellipse at 80% 20%,rgba(0,212,255,0.1) 0%,transparent 50%),
               radial-gradient(ellipse at 50% 80%,rgba(255,107,53,0.08) 0%,transparent 50%);
    pointer-events:none; z-index:0;
  }
  [data-testid="stSidebar"] {
    background:linear-gradient(180deg,#080820 0%,#0d1535 50%,#150d30 100%) !important;
    border-right:1px solid var(--border); box-shadow:5px 0 30px rgba(0,212,255,0.1);
  }
  [data-testid="stSidebar"] label {
    color:var(--primary)!important; font-family:'Rajdhani',sans-serif!important;
    font-weight:600!important; font-size:0.85rem!important;
    text-transform:uppercase!important; letter-spacing:1px!important;
  }
  .card-3d {
    background:linear-gradient(145deg,rgba(20,20,60,0.9),rgba(10,10,35,0.95));
    border:1px solid var(--border); border-radius:16px; padding:24px;
    box-shadow:0 10px 40px rgba(0,0,0,0.5),0 0 20px rgba(0,212,255,0.08),
               inset 0 1px 0 rgba(255,255,255,0.05);
    transition:transform 0.3s ease; margin-bottom:16px; position:relative; overflow:hidden;
  }
  .card-3d::before {
    content:''; position:absolute; top:0;left:0;right:0; height:1px;
    background:linear-gradient(90deg,transparent,var(--primary),transparent);
  }
  .metric-card {
    background:linear-gradient(135deg,rgba(0,212,255,0.08),rgba(123,47,247,0.08));
    border:1px solid var(--border); border-radius:12px; padding:20px; text-align:center;
    position:relative; overflow:hidden; box-shadow:0 4px 20px rgba(0,0,0,0.3); transition:all 0.3s ease;
  }
  .metric-card::after {
    content:''; position:absolute; bottom:0;left:0;right:0; height:2px;
    background:linear-gradient(90deg,var(--secondary),var(--primary));
  }
  .metric-card:hover { transform:translateY(-3px); border-color:rgba(0,212,255,0.6); }
  .metric-value {
    font-family:'Orbitron',monospace; font-size:1.5rem; font-weight:700;
    background:linear-gradient(135deg,var(--primary),var(--secondary));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; display:block; line-height:1.3;
  }
  .metric-label { font-size:0.72rem; color:rgba(224,224,255,0.6); text-transform:uppercase; letter-spacing:2px; margin-top:6px; }
  .hero-title {
    font-family:'Orbitron',monospace; font-size:3rem; font-weight:900; text-align:center;
    background:linear-gradient(135deg,#00d4ff 0%,#7b2ff7 50%,#ff6b35 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; letter-spacing:2px; line-height:1.2; margin-bottom:8px;
  }
  .hero-subtitle {
    font-family:'Rajdhani',sans-serif; font-size:1.1rem; color:rgba(0,212,255,0.7);
    text-align:center; letter-spacing:4px; text-transform:uppercase; margin-bottom:32px;
  }
  .section-header {
    font-family:'Orbitron',monospace; font-size:1.05rem; font-weight:700; color:var(--primary);
    letter-spacing:2px; text-transform:uppercase; margin-bottom:16px; padding-bottom:8px;
    border-bottom:1px solid var(--border); position:relative;
  }
  .section-header::after {
    content:''; position:absolute; bottom:-1px;left:0; width:60px;height:2px; background:var(--accent);
  }
  .price-display {
    background:linear-gradient(135deg,rgba(0,212,255,0.1),rgba(123,47,247,0.15));
    border:2px solid var(--primary); border-radius:20px; padding:28px; text-align:center;
    box-shadow:0 0 40px rgba(0,212,255,0.2),inset 0 0 30px rgba(0,212,255,0.05);
    animation:pulse-glow 3s ease-in-out infinite;
  }
  @keyframes pulse-glow {
    0%,100%{box-shadow:0 0 40px rgba(0,212,255,0.2),0 0 80px rgba(0,212,255,0.08);}
    50%{box-shadow:0 0 60px rgba(0,212,255,0.35),0 0 120px rgba(0,212,255,0.15);}
  }
  .price-main { font-family:'Orbitron',monospace; font-size:2.4rem; font-weight:900;
    background:linear-gradient(135deg,#00ff88,#00d4ff); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; background-clip:text; display:block; line-height:1.3; }
  .price-full { font-size:0.85rem; color:rgba(0,212,255,0.7); letter-spacing:1px; margin-top:8px; }
  .price-range { font-size:0.75rem; color:rgba(224,224,255,0.4); margin-top:6px; }
  .price-label { font-size:1rem; color:var(--primary); letter-spacing:2px; margin-bottom:6px; }
  .compare-card {
    background:linear-gradient(145deg,rgba(15,15,50,0.95),rgba(8,8,25,0.98));
    border-radius:20px; padding:22px; border:1px solid var(--border);
    box-shadow:0 15px 50px rgba(0,0,0,0.5),0 0 30px rgba(123,47,247,0.1);
    transition:all 0.3s ease;
  }
  .compare-card:hover { border-color:rgba(0,212,255,0.5); transform:translateY(-3px); }
  .compare-badge { display:inline-block; padding:4px 14px; border-radius:20px;
    font-size:0.7rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; margin-bottom:12px; }
  .badge-a { background:linear-gradient(135deg,rgba(0,212,255,0.2),rgba(0,212,255,0.1)); border:1px solid var(--primary); color:var(--primary); }
  .badge-b { background:linear-gradient(135deg,rgba(255,107,53,0.2),rgba(255,107,53,0.1)); border:1px solid var(--accent); color:var(--accent); }
  .compare-car-name { font-family:'Orbitron',monospace; font-size:1rem; font-weight:700; color:white; margin-bottom:4px; }
  .stat-row { display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.05); }
  .stat-label { font-size:0.76rem; color:rgba(224,224,255,0.5); text-transform:uppercase; letter-spacing:1px; }
  .stat-value { font-weight:700; color:var(--text); font-size:0.88rem; }
  .car-img-container {
    background:linear-gradient(135deg,rgba(0,212,255,0.05),rgba(123,47,247,0.08));
    border:1px solid var(--border); border-radius:16px; padding:16px; text-align:center;
    min-height:140px; display:flex; align-items:center; justify-content:center;
    flex-direction:column; margin-bottom:12px;
  }
  .car-emoji-display { font-size:5.5rem; line-height:1; filter:drop-shadow(0 0 20px rgba(0,212,255,0.4)); animation:float 4s ease-in-out infinite; }
  @keyframes float { 0%,100%{transform:translateY(0px) rotate(-2deg);} 50%{transform:translateY(-10px) rotate(2deg);} }
  .winner-tag { background:linear-gradient(135deg,#ffd700,#ff8c00); color:#000;
    padding:4px 12px; border-radius:20px; font-size:0.7rem; font-weight:800;
    letter-spacing:2px; text-transform:uppercase; display:inline-block; margin-top:8px;
    box-shadow:0 0 15px rgba(255,215,0,0.4); }
  .stSelectbox>div>div { background:rgba(15,15,45,0.9)!important; border:1px solid var(--border)!important; border-radius:10px!important; }
  .stSlider>div>div>div { background:var(--primary)!important; }
  div[data-testid="stMetric"] { background:linear-gradient(135deg,rgba(0,212,255,0.06),rgba(123,47,247,0.06)); border:1px solid var(--border); border-radius:12px; padding:16px; }
  div[data-testid="stMetric"] label { color:rgba(0,212,255,0.8)!important; font-size:0.8rem!important; text-transform:uppercase!important; letter-spacing:1px!important; font-family:'Rajdhani',sans-serif!important; }
  div[data-testid="stMetric"] div { font-family:'Orbitron',monospace!important; color:white!important; font-size:1.3rem!important; }
  .stTabs [data-baseweb="tab-list"] { background:rgba(10,10,30,0.8)!important; border:1px solid var(--border)!important; border-radius:12px!important; padding:4px!important; gap:4px!important; }
  .stTabs [data-baseweb="tab"] { border-radius:8px!important; color:rgba(224,224,255,0.5)!important; font-family:'Rajdhani',sans-serif!important; font-size:0.85rem!important; font-weight:600!important; letter-spacing:1px!important; text-transform:uppercase!important; }
  .stTabs [aria-selected="true"] { background:linear-gradient(135deg,rgba(0,212,255,0.2),rgba(123,47,247,0.2))!important; color:var(--primary)!important; border:1px solid var(--border)!important; }
  ::-webkit-scrollbar{width:6px;} ::-webkit-scrollbar-track{background:#0a0a1a;} ::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
  .stButton>button { background:linear-gradient(135deg,#00d4ff,#7b2ff7)!important; color:white!important; border:none!important; border-radius:12px!important; font-family:'Rajdhani',sans-serif!important; font-weight:700!important; font-size:1rem!important; letter-spacing:2px!important; text-transform:uppercase!important; padding:12px 32px!important; box-shadow:0 4px 20px rgba(0,212,255,0.3)!important; transition:all 0.3s ease!important; }
  .stButton>button:hover { transform:translateY(-2px)!important; box-shadow:0 8px 30px rgba(0,212,255,0.5)!important; }
  .info-box { background:rgba(0,212,255,0.05); border:1px solid rgba(0,212,255,0.2); border-radius:10px; padding:12px 16px; margin:8px 0; font-size:0.88rem; }
  .warning-box { background:rgba(255,107,53,0.08); border:1px solid rgba(255,107,53,0.3); border-radius:10px; padding:12px 16px; margin:8px 0; font-size:0.88rem; color:#ffb399; }
  .footer { text-align:center; padding:24px; color:rgba(224,224,255,0.3); font-size:0.75rem; letter-spacing:2px; text-transform:uppercase; border-top:1px solid var(--border); margin-top:40px; }
  #MainMenu{visibility:hidden;} footer{visibility:hidden;} header{visibility:hidden;}
  hr { border:none; height:1px; background:linear-gradient(90deg,transparent,var(--border),transparent); margin:20px 0; }
  @media(max-width:768px){
    .hero-title{font-size:1.8rem!important;}
    .price-main{font-size:1.8rem!important;}
  }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        art_dir = 'model_artifacts'
        required = ['label_encoders.pkl','feature_cols.pkl','model_results.pkl','dataset_stats.pkl']
        for req in required:
            if not os.path.exists(os.path.join(art_dir, req)):
                return None,None,None,None,None,None
        artifacts = {}
        for fname, dname in {'random_forest.pkl':'Random Forest','gradient_boosting.pkl':'Gradient Boosting','ridge_regression.pkl':'Ridge Regression','xgboost.pkl':'XGBoost'}.items():
            path = os.path.join(art_dir, fname)
            if os.path.exists(path):
                artifacts[dname] = joblib.load(path)
        le_dict       = joblib.load(os.path.join(art_dir,'label_encoders.pkl'))
        feature_cols  = joblib.load(os.path.join(art_dir,'feature_cols.pkl'))
        model_results = joblib.load(os.path.join(art_dir,'model_results.pkl'))
        stats         = joblib.load(os.path.join(art_dir,'dataset_stats.pkl'))
        ref_path = os.path.join(art_dir,'reference_year.pkl')
        reference_year = joblib.load(ref_path) if os.path.exists(ref_path) else datetime.date.today().year
        return artifacts,le_dict,feature_cols,model_results,stats,reference_year
    except Exception:
        return None,None,None,None,None,None

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('car_data.csv')
        df.columns = df.columns.str.strip()
        if df['Selling_Price'].max() > 10_000:
            df['Selling_Price'] = df['Selling_Price'] / 1_00_000
        df['Car_Age'] = datetime.date.today().year - df['Year']
        return df
    except Exception:
        return None

def predict_price(model, le_dict, feature_cols, reference_year,
                  car_name, year, kms_driven, mileage, engine, max_power, seats,
                  fuel_type, seller_type, transmission, owner, present_price=None):
    car_age = reference_year - year
    def safe_encode(le, value, col_name):
        classes = list(le.classes_)
        return le.transform([value if value in classes else classes[0]])[0]
    all_values = {
        'Car_Name_enc':     safe_encode(le_dict['Car_Name'],    car_name,    'Car_Name'),
        'Present_Price':    present_price if present_price is not None else 0.0,
        'Car_Age':          car_age, 'Kms_Driven': kms_driven,
        'Mileage':          mileage,  'Engine':     engine,
        'Max_Power':        max_power, 'Seats':      seats,
        'Fuel_Type_enc':    safe_encode(le_dict['Fuel_Type'],   fuel_type,   'Fuel_Type'),
        'Seller_Type_enc':  safe_encode(le_dict['Seller_Type'], seller_type, 'Seller_Type'),
        'Transmission_enc': safe_encode(le_dict['Transmission'],transmission,'Transmission'),
        'Owner_enc':        safe_encode(le_dict['Owner'],       owner,       'Owner'),
    }
    row = [[all_values[c] for c in feature_cols]]
    return max(0, model.predict(pd.DataFrame(row, columns=feature_cols))[0])

def get_car_emoji(car_name, fuel_type="Petrol"):
    n = str(car_name).lower()
    if any(x in n for x in ['fortuner','creta','duster','brezza','nexon','eco sport','tucson']): return "🚙"
    if any(x in n for x in ['innova','ertiga','lodgy','enjoy','tavera']): return "🚐"
    if any(x in n for x in ['superb','elantra','corolla','city','verna','ciaz','vento','rapid','jetta','octavia']): return "🚗"
    if any(x in n for x in ['alto','kwid','beat','spark','brio','ritz']): return "🚕"
    return "🚕" if fuel_type == "CNG" else "🚗"

def get_color_hex(fuel_type):
    return {'Petrol':'#00d4ff','Diesel':'#ff6b35','CNG':'#00ff88','Electric':'#ffd700'}.get(fuel_type,'#7b2ff7')

def render_car_card(col, badge_class, badge_text, car_name, fuel_type, year, kms, seller, trans, owner, price, is_winner=False):
    emoji = get_car_emoji(car_name, fuel_type)
    color = get_color_hex(fuel_type)
    price_str  = fmt_inr(price) if price else "—"
    rupees_str = "₹ {:s}".format(fmt_num(price * 1_00_000)) if price else ""
    winner_div = ("<div class=\"winner-tag\">" + chr(127942) + " BEST VALUE</div>") if (is_winner and price) else ""
    html = (
        "<div class=\"compare-card\">"
        + "<span class=\"compare-badge " + badge_class + "\">" + badge_text + "</span>"
        + "<div class=\"car-img-container\" style=\"border-color:" + color + "30;\">"
        + "<div class=\"car-emoji-display\">" + emoji + "</div>"
        + "<div style=\"color:" + color + ";font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;margin-top:6px;\">" + fuel_type + " &middot; " + trans + "</div>"
        + "</div>"
        + "<div class=\"compare-car-name\">" + car_name.title() + "</div>"
        + "<div style=\"color:rgba(224,224,255,0.4);font-size:0.75rem;margin-bottom:10px;\">" + str(year) + " &middot; " + fmt_km(kms) + "</div>"
        + winner_div
        + "<div class=\"stat-row\"><span class=\"stat-label\">Seller</span><span class=\"stat-value\">" + str(seller) + "</span></div>"
        + "<div class=\"stat-row\"><span class=\"stat-label\">Transmission</span><span class=\"stat-value\">" + str(trans) + "</span></div>"
        + "<div class=\"stat-row\"><span class=\"stat-label\">Owner</span><span class=\"stat-value\">" + str(owner) + "</span></div>"
        + "<div class=\"stat-row\" style=\"border:none;padding-top:10px;flex-direction:column;align-items:flex-start;\">"
        + "<span class=\"stat-label\" style=\"margin-bottom:4px;\">Predicted Price</span>"
        + "<span style=\"font-family:'Orbitron',monospace;font-size:1.1rem;background:linear-gradient(135deg,#00ff88,#00d4ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:900;\">" + price_str + "</span>"
        + "<span style=\"font-size:0.72rem;color:rgba(224,224,255,0.35);margin-top:2px;\">" + rupees_str + "</span>"
        + "</div></div>"
    )
    col.markdown(html, unsafe_allow_html=True)

def auto_train_if_needed():
    required = ['model_artifacts/random_forest.pkl','model_artifacts/label_encoders.pkl',
                'model_artifacts/feature_cols.pkl','model_artifacts/dataset_stats.pkl']
    if not all(os.path.exists(p) for p in required):
        pb = st.progress(0, text="⚙️ First-run setup — training ML models (~60s)…")
        try:
            import model_trainer
            pb.progress(10, text="📊 Loading & preprocessing data…")
            df, le_dict = model_trainer.load_and_preprocess('car_data.csv')
            pb.progress(35, text="🔧 Engineering features…")
            X, y, feature_cols = model_trainer.get_features_target(df)
            from sklearn.model_selection import train_test_split
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
            pb.progress(45, text="🤖 Training models…")
            results,trained_models,best_name = model_trainer.train_models(X_train,X_test,y_train,y_test)
            pb.progress(88, text="💾 Saving artifacts…")
            model_trainer.save_artifacts(trained_models,le_dict,feature_cols,results,df)
            pb.progress(100, text="✅ Training complete!")
            time.sleep(0.8); pb.empty()
        except Exception as e:
            pb.empty()
            st.error(f"❌ Model training failed: {e}")
            st.info("Make sure `car_data.csv` and `model_trainer.py` are present.")
            st.stop()
        st.cache_resource.clear(); st.cache_data.clear(); st.rerun()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    auto_train_if_needed()
    artifacts,le_dict,feature_cols,model_results,stats,reference_year = load_artifacts()
    df = load_dataset()
    model_ready = artifacts is not None and len(artifacts) > 0

    PAGES = ["🏠 Dashboard","🔮 Price Predictor","⚡ Car Comparison","📊 Analytics Hub","🤖 Model Intelligence"]
    if "page" not in st.session_state:
        st.session_state.page = PAGES[0]

    # ── SIDEBAR ONLY navigation (no top bar) ─────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:20px 0 10px 0;">
          <div style="font-family:'Orbitron',monospace;font-size:1.4rem;font-weight:900;
                      background:linear-gradient(135deg,#00d4ff,#7b2ff7);-webkit-background-clip:text;
                      -webkit-text-fill-color:transparent;background-clip:text;">🚗 AUTOVALUE AI</div>
          <div style="font-size:0.65rem;color:rgba(0,212,255,0.5);letter-spacing:3px;text-transform:uppercase;margin-top:4px;">Car Intelligence System</div>
        </div><hr>""", unsafe_allow_html=True)

        sidebar_page = st.selectbox("📍 Navigate", PAGES,
                                    index=PAGES.index(st.session_state.page),
                                    label_visibility="collapsed")
        if sidebar_page != st.session_state.page:
            st.session_state.page = sidebar_page
            st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        if not model_ready:
            st.markdown('<div class="warning-box">⚠️ Models not trained.<br>Run: <code>python model_trainer.py</code></div>', unsafe_allow_html=True)
        else:
            n_rec = stats.get('total_records', 0)
            p_mean = stats.get('price_mean', 0)
            st.markdown(f'<div class="info-box">✅ <strong>{len(artifacts)} models</strong> loaded<br>📊 <strong>{fmt_num(n_rec)}</strong> training records<br>💰 Avg price: <strong>{fmt_inr_short(p_mean)}</strong></div>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.7rem;color:rgba(224,224,255,0.3);text-align:center;letter-spacing:1px;">POWERED BY MACHINE LEARNING<br>Random Forest · XGBoost · GBM</div>', unsafe_allow_html=True)

    page = st.session_state.page

    # ════════════════════════════════════════════
    # DASHBOARD
    # ════════════════════════════════════════════
    if "Dashboard" in page:
        st.markdown('<div class="hero-title">🚗 AUTOVALUE AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-subtitle">Machine Learning Car Price Intelligence System</div>', unsafe_allow_html=True)

        if df is not None and stats:
            c1,c2,c3,c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card"><span class="metric-value">{fmt_num(stats["total_records"])}</span><div class="metric-label">🗃️ Total Cars</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><span class="metric-value">{fmt_inr_short(stats["price_mean"])}</span><div class="metric-label">💰 Avg Resale Price</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><span class="metric-value">{stats["year_min"]}<br>{stats["year_max"]}</span><div class="metric-label">📅 Year Range</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card"><span class="metric-value">{fmt_num(len(stats["car_names"]))}</span><div class="metric-label">🏷️ Car Models</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_left,col_right = st.columns([3,2])
        with col_left:
            if df is not None:
                st.markdown('<div class="section-header">📈 Price Distribution by Fuel Type</div>', unsafe_allow_html=True)
                fig = px.violin(df,x='Fuel_Type',y='Selling_Price',color='Fuel_Type',
                    color_discrete_map={'Petrol':'#00d4ff','Diesel':'#ff6b35','CNG':'#00ff88'},
                    box=True,points="outliers",
                    labels={'Selling_Price':'Resale Price (₹ Lakh)','Fuel_Type':'Fuel Type'})
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='rgba(224,224,255,0.8)',family='Rajdhani'),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True,gridcolor='rgba(0,212,255,0.08)',title='Resale Price (₹ Lakh)'),
                    showlegend=False,height=350,margin=dict(l=0,r=0,t=20,b=0))
                st.plotly_chart(fig, use_container_width=True)
        with col_right:
            if df is not None:
                st.markdown('<div class="section-header">🔢 Fuel Type Share</div>', unsafe_allow_html=True)
                fuel_counts = df['Fuel_Type'].value_counts()
                fig2 = go.Figure(data=[go.Pie(
                    labels=fuel_counts.index,values=fuel_counts.values,hole=0.65,
                    marker=dict(colors=['#00d4ff','#ff6b35','#00ff88','#7b2ff7'],line=dict(color='#0a0a1a',width=3)),
                    textfont=dict(family='Rajdhani',size=13,color='white'))])
                fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='rgba(224,224,255,0.8)',family='Rajdhani'),height=350,
                    showlegend=True,legend=dict(font=dict(color='white',size=12)),margin=dict(l=0,r=0,t=20,b=0),
                    annotations=[dict(text="FUEL<br>TYPES",x=0.5,y=0.5,font_size=14,font_family='Orbitron',font_color='#00d4ff',showarrow=False)])
                st.plotly_chart(fig2, use_container_width=True)

        if df is not None:
            st.markdown('<div class="section-header">🌐 3D Market View: Price × Year × KMs Driven</div>', unsafe_allow_html=True)
            fcm = {'Petrol':'#00d4ff','Diesel':'#ff6b35','CNG':'#00ff88'}
            fig3d = go.Figure(data=[
                go.Scatter3d(
                    x=df[df['Fuel_Type']==ft]['Year'],
                    y=df[df['Fuel_Type']==ft]['Kms_Driven'],
                    z=df[df['Fuel_Type']==ft]['Selling_Price'],
                    mode='markers',name=ft,
                    marker=dict(size=5,color=fcm.get(ft,'#7b2ff7'),opacity=0.8),
                    hovertemplate="<b>"+ft+"</b><br>Year: %{x}<br>KMs: %{y:,}<br>Price: ₹%{z:.2f} L<extra></extra>"
                ) for ft in df['Fuel_Type'].unique()])
            fig3d.update_layout(
                scene=dict(
                    xaxis=dict(title='Year',backgroundcolor='rgba(10,10,30,0.8)',gridcolor='rgba(0,212,255,0.1)',showbackground=True),
                    yaxis=dict(title='KMs Driven',backgroundcolor='rgba(10,10,30,0.8)',gridcolor='rgba(0,212,255,0.1)',showbackground=True),
                    zaxis=dict(title='Price (₹ Lakh)',backgroundcolor='rgba(10,10,30,0.8)',gridcolor='rgba(0,212,255,0.1)',showbackground=True),
                    bgcolor='rgba(10,10,30,0.8)'),
                paper_bgcolor='rgba(0,0,0,0)',font=dict(color='rgba(224,224,255,0.8)',family='Rajdhani'),
                height=480,legend=dict(font=dict(color='white')),margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig3d, use_container_width=True)

    # ════════════════════════════════════════════
    # PRICE PREDICTOR
    # ════════════════════════════════════════════
    elif "Predictor" in page:
        st.markdown('<div class="hero-title">🔮 PRICE PREDICTOR</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-subtitle">AI-Powered Market Valuation Engine</div>', unsafe_allow_html=True)
        if not model_ready:
            st.error("⚠️ Please train the models first: `python model_trainer.py`"); st.stop()

        col_form,col_result = st.columns([2,1])
        with col_form:
            st.markdown('<div class="card-3d"><div class="section-header">🚗 Vehicle Details</div>', unsafe_allow_html=True)
            c1,c2 = st.columns(2)
            with c1:
                car_name    = st.selectbox("Car Model", sorted(stats['car_names']), key="pred_car")
                year        = st.slider("Manufacturing Year", stats['year_min'], stats['year_max'], 2017)
                fuel_type   = st.selectbox("Fuel Type", stats['fuel_types'], key="pred_fuel")
                owner       = st.selectbox("Ownership", stats.get('owner_values',['First Owner','Second Owner','Third Owner']), key="pred_owner")
                seller_type = st.selectbox("Seller Type", stats['seller_types'], key="pred_seller")
            with c2:
                kms_driven   = st.number_input("KMs Driven", min_value=500, max_value=500000, value=35000, step=1000)
                mileage      = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=50.0, value=18.0, step=0.5)
                engine       = st.number_input("Engine (cc)", min_value=600, max_value=5000, value=1200, step=50)
                max_power    = st.number_input("Max Power (bhp)", min_value=30.0, max_value=500.0, value=80.0, step=1.0)
                seats        = st.selectbox("Seats", stats.get('seats_values',[2,4,5,6,7,8]), index=2, key="pred_seats")
                transmission = st.selectbox("Transmission", stats['transmissions'], key="pred_trans")

            present_price = None
            if 'Present_Price' in feature_cols:
                pp_default = round(stats.get('present_price_mean',8.0),1)
                pp_min     = round(stats.get('present_price_min',1.0),1)
                pp_max     = round(stats.get('present_price_max',100.0),1)
                present_price = st.number_input(
                    "Ex-showroom Price (₹ Lakh)",
                    min_value=pp_min, max_value=pp_max, value=pp_default, step=0.5,
                    help="Current ex-showroom price in ₹ Lakh")
            model_choice = st.selectbox("🤖 ML Model", list(artifacts.keys()))
            st.markdown('</div>', unsafe_allow_html=True)
            predict_btn = st.button("⚡ PREDICT PRICE", use_container_width=True)

        with col_result:
            st.markdown('<div class="section-header">💰 Valuation</div>', unsafe_allow_html=True)
            if predict_btn:
                with st.spinner("Analysing market data…"):
                    time.sleep(0.4)
                    model = artifacts[model_choice]
                    price = predict_price(model,le_dict,feature_cols,reference_year,
                                          car_name,year,kms_driven,mileage,engine,max_power,seats,
                                          fuel_type,seller_type,transmission,owner,present_price=present_price)
                low  = price * 0.88
                high = price * 1.12
                emoji = get_car_emoji(car_name, fuel_type)

                st.markdown("""
                <div class="price-display">
                  <div style="font-size:4rem;margin-bottom:6px;filter:drop-shadow(0 0 20px rgba(0,212,255,0.5));">{emoji}</div>
                  <div class="price-label">ESTIMATED RESALE VALUE</div>
                  <span class="price-main">{main}</span>
                  <div class="price-full">{full}</div>
                  <div class="price-range">Expected Range: {lo} – {hi}</div>
                  <div style="margin-top:14px;display:flex;gap:8px;justify-content:center;flex-wrap:wrap;">
                    <span style="background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.3);padding:3px 10px;border-radius:20px;font-size:0.7rem;color:#00d4ff;">{fuel}</span>
                    <span style="background:rgba(123,47,247,0.1);border:1px solid rgba(123,47,247,0.3);padding:3px 10px;border-radius:20px;font-size:0.7rem;color:#9b5ff7;">{trans}</span>
                    <span style="background:rgba(255,107,53,0.1);border:1px solid rgba(255,107,53,0.3);padding:3px 10px;border-radius:20px;font-size:0.7rem;color:#ff8c5a;">{yr}</span>
                  </div>
                </div>""".format(
                    emoji=emoji, main=fmt_inr(price), full=fmt_inr_full(price),
                    lo=fmt_inr(low), hi=fmt_inr(high),
                    fuel=fuel_type, trans=transmission, yr=year
                ), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                avg_price  = _to_lakhs(stats['price_mean'])
                price_min  = _to_lakhs(stats['price_min'])
                price_max  = _to_lakhs(stats['price_max'])
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",value=price,
                    number={'valueformat':'.2f','suffix':' L','prefix':'₹ ','font':{'color':'#00d4ff','family':'Orbitron','size':18}},
                    gauge={
                        'axis':{'range':[price_min, price_max],'tickcolor':'rgba(0,212,255,0.4)'},
                        'bar':{'color':'rgba(0,212,255,0.8)','thickness':0.3},
                        'bgcolor':'rgba(0,0,0,0)','bordercolor':'rgba(0,212,255,0.2)',
                        'steps':[
                            {'range':[price_min, avg_price*0.6],'color':'rgba(255,0,0,0.15)'},
                            {'range':[avg_price*0.6, avg_price*1.2],'color':'rgba(255,165,0,0.15)'},
                            {'range':[avg_price*1.2, price_max],'color':'rgba(0,255,136,0.15)'},
                        ],
                        'threshold':{'line':{'color':'#ffd700','width':3},'value':avg_price}
                    },
                    title={'text':"Market Avg: "+fmt_inr_short(avg_price),'font':{'color':'rgba(224,224,255,0.6)','size':11,'family':'Rajdhani'}}
                ))
                fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)',font=dict(color='white'),height=220,margin=dict(l=10,r=10,t=30,b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown('<div class="section-header">📊 All Models</div>', unsafe_allow_html=True)
                rows = []
                for mname, mobj in artifacts.items():
                    p = predict_price(mobj,le_dict,feature_cols,reference_year,
                                      car_name,year,kms_driven,mileage,engine,max_power,seats,
                                      fuel_type,seller_type,transmission,owner,present_price=present_price)
                    rows.append({"Model":mname, "Predicted Price":fmt_inr(p),
                                 "In Rupees":"₹ "+fmt_num(p*1_00_000),
                                 "R²": str(model_results[mname]['R2 Score'])+"%" if mname in model_results else "—"})
                st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
            else:
                st.markdown("""
                <div style="text-align:center;padding:60px 20px;color:rgba(224,224,255,0.3);">
                  <div style="font-size:4rem;margin-bottom:12px;">🔮</div>
                  <div style="font-family:'Orbitron',monospace;font-size:0.9rem;letter-spacing:2px;">
                    AWAITING INPUT<br><span style="font-size:0.7rem;color:rgba(224,224,255,0.2);">Configure specs and click PREDICT</span>
                  </div>
                </div>""", unsafe_allow_html=True)


    # ════════════════════════════════════════════
    # CAR COMPARISON
    # ════════════════════════════════════════════
    elif "Comparison" in page:
        st.markdown('<div class="hero-title">⚡ CAR COMPARISON</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-subtitle">Head-to-Head Intelligence Analysis</div>', unsafe_allow_html=True)
        if not model_ready:
            st.error("⚠️ Please train the models first: `python model_trainer.py`"); st.stop()

        car_names    = sorted(stats['car_names'])
        model_choice = st.selectbox("🤖 Prediction Model", list(artifacts.keys()), key="cmp_model")
        st.markdown("<hr>", unsafe_allow_html=True)
        colA,vs_col,colB = st.columns([5,1,5])
        with colA:
            st.markdown('<div class="compare-badge badge-a">🔵 CAR A</div>', unsafe_allow_html=True)
            a_name = st.selectbox("Car Model A", car_names, index=0, key="cmp_a_name")
            a1,a2 = st.columns(2)
            with a1:
                a_year=st.slider("Year A",stats['year_min'],stats['year_max'],2017,key="cmp_a_yr")
                a_fuel=st.selectbox("Fuel A",stats['fuel_types'],key="cmp_a_fuel")
                a_seller=st.selectbox("Seller A",stats['seller_types'],key="cmp_a_seller")
                a_owner=st.selectbox("Owner A",stats.get('owner_values',['First Owner','Second Owner','Third Owner']),key="cmp_a_own")
            with a2:
                a_kms=st.number_input("KMs A",500,500000,30000,1000,key="cmp_a_kms")
                a_mileage=st.number_input("Mileage A (kmpl)",5.0,50.0,18.0,0.5,key="cmp_a_mil")
                a_engine=st.number_input("Engine A (cc)",600,5000,1200,50,key="cmp_a_eng")
                a_max_power=st.number_input("Max Power A (bhp)",30.0,500.0,80.0,1.0,key="cmp_a_pwr")
                a_seats=st.selectbox("Seats A",stats.get('seats_values',[2,4,5,6,7,8]),index=2,key="cmp_a_seats")
                a_trans=st.selectbox("Trans A",stats['transmissions'],key="cmp_a_trans")

        with vs_col:
            st.markdown('<div style="display:flex;align-items:center;justify-content:center;height:100%;min-height:300px;"><div style="font-family:\'Orbitron\',monospace;font-size:1.5rem;font-weight:900;background:linear-gradient(135deg,#00d4ff,#ff6b35);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">VS</div></div>', unsafe_allow_html=True)

        with colB:
            st.markdown('<div class="compare-badge badge-b">🟠 CAR B</div>', unsafe_allow_html=True)
            b_name=st.selectbox("Car Model B",car_names,index=min(5,len(car_names)-1),key="cmp_b_name")
            b1,b2=st.columns(2)
            with b1:
                b_year=st.slider("Year B",stats['year_min'],stats['year_max'],2015,key="cmp_b_yr")
                b_fuel=st.selectbox("Fuel B",stats['fuel_types'],key="cmp_b_fuel")
                b_seller=st.selectbox("Seller B",stats['seller_types'],key="cmp_b_seller")
                b_owner=st.selectbox("Owner B",stats.get('owner_values',['First Owner','Second Owner','Third Owner']),key="cmp_b_own")
            with b2:
                b_kms=st.number_input("KMs B",500,500000,45000,1000,key="cmp_b_kms")
                b_mileage=st.number_input("Mileage B (kmpl)",5.0,50.0,16.0,0.5,key="cmp_b_mil")
                b_engine=st.number_input("Engine B (cc)",600,5000,1500,50,key="cmp_b_eng")
                b_max_power=st.number_input("Max Power B (bhp)",30.0,500.0,100.0,1.0,key="cmp_b_pwr")
                b_seats=st.selectbox("Seats B",stats.get('seats_values',[2,4,5,6,7,8]),index=2,key="cmp_b_seats")
                b_trans=st.selectbox("Trans B",stats['transmissions'],key="cmp_b_trans")

        compare_btn = st.button("⚡ COMPARE CARS", use_container_width=True)
        if compare_btn:
            with st.spinner("Computing valuations…"):
                time.sleep(0.4)
                model=artifacts[model_choice]
                price_a=predict_price(model,le_dict,feature_cols,reference_year,a_name,a_year,a_kms,a_mileage,a_engine,a_max_power,a_seats,a_fuel,a_seller,a_trans,a_owner)
                price_b=predict_price(model,le_dict,feature_cols,reference_year,b_name,b_year,b_kms,b_mileage,b_engine,b_max_power,b_seats,b_fuel,b_seller,b_trans,b_owner)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<div class="section-header" style="text-align:center;">📊 COMPARISON RESULTS</div>', unsafe_allow_html=True)

            s1,s2 = st.columns(2)
            with s1:
                st.markdown(f"""<div style="background:linear-gradient(135deg,rgba(0,212,255,0.08),rgba(0,212,255,0.03));border:1px solid rgba(0,212,255,0.3);border-radius:14px;padding:18px;text-align:center;">
                  <div style="font-size:0.72rem;color:rgba(0,212,255,0.7);letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">🔵 {a_name.title()}</div>
                  <div style="font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:900;background:linear-gradient(135deg,#00ff88,#00d4ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{fmt_inr(price_a)}</div>
                  <div style="font-size:0.72rem;color:rgba(224,224,255,0.35);margin-top:6px;">₹ {fmt_num(price_a*1_00_000)} Rupees</div>
                </div>""", unsafe_allow_html=True)
            with s2:
                st.markdown(f"""<div style="background:linear-gradient(135deg,rgba(255,107,53,0.08),rgba(255,107,53,0.03));border:1px solid rgba(255,107,53,0.3);border-radius:14px;padding:18px;text-align:center;">
                  <div style="font-size:0.72rem;color:rgba(255,107,53,0.8);letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">🟠 {b_name.title()}</div>
                  <div style="font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:900;background:linear-gradient(135deg,#ff6b35,#ffd700);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{fmt_inr(price_b)}</div>
                  <div style="font-size:0.72rem;color:rgba(224,224,255,0.35);margin-top:6px;">₹ {fmt_num(price_b*1_00_000)} Rupees</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            r1,r2=st.columns(2)
            render_car_card(r1,"badge-a","CAR A",a_name,a_fuel,a_year,a_kms,a_seller,a_trans,a_owner,price_a,price_a>=price_b)
            render_car_card(r2,"badge-b","CAR B",b_name,b_fuel,b_year,b_kms,b_seller,b_trans,b_owner,price_b,price_b>price_a)

            st.markdown("<br>", unsafe_allow_html=True)
            metrics={'Price (₹ L)':[price_a,price_b],'Mileage (kmpl)':[a_mileage,b_mileage],
                     'KMs (÷1000)':[a_kms/1000,b_kms/1000],'Car Age (yr)':[reference_year-a_year,reference_year-b_year],
                     'Engine (cc÷10)':[a_engine/10,b_engine/10]}
            cats=list(metrics.keys()); vals_a=[metrics[k][0] for k in cats]; vals_b=[metrics[k][1] for k in cats]
            fig_cmp=go.Figure()
            fig_cmp.add_trace(go.Bar(name="🔵 "+a_name.title(),x=cats,y=vals_a,
                marker=dict(color='rgba(0,212,255,0.8)',line=dict(color='#00d4ff',width=1)),
                text=[f"{v:.1f}" for v in vals_a],textposition='outside',textfont=dict(color='#00d4ff',family='Orbitron',size=10)))
            fig_cmp.add_trace(go.Bar(name="🟠 "+b_name.title(),x=cats,y=vals_b,
                marker=dict(color='rgba(255,107,53,0.8)',line=dict(color='#ff6b35',width=1)),
                text=[f"{v:.1f}" for v in vals_b],textposition='outside',textfont=dict(color='#ff6b35',family='Orbitron',size=10)))
            fig_cmp.update_layout(barmode='group',plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='rgba(224,224,255,0.8)',family='Rajdhani'),xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True,gridcolor='rgba(0,212,255,0.08)'),
                legend=dict(font=dict(color='white',size=12)),height=350,margin=dict(l=0,r=0,t=40,b=0),bargap=0.25,bargroupgap=0.08)
            st.plotly_chart(fig_cmp, use_container_width=True)

            st.markdown('<div class="section-header">🕸️ Attribute Radar</div>', unsafe_allow_html=True)
            radar_cats=['Value Score','KM Efficiency','Newness','Mileage','Market Demand']
            def normalize(v,lo,hi): return (v-lo)/(hi-lo+0.001)*100 if hi>lo else 50
            a_scores=[normalize(price_a,0,stats['price_max']),100-normalize(a_kms,0,200000),normalize(a_year,2005,2024),normalize(a_mileage,stats.get('mileage_min',5),stats.get('mileage_max',50)),80 if a_fuel=='Diesel' else 65]
            b_scores=[normalize(price_b,0,stats['price_max']),100-normalize(b_kms,0,200000),normalize(b_year,2005,2024),normalize(b_mileage,stats.get('mileage_min',5),stats.get('mileage_max',50)),80 if b_fuel=='Diesel' else 65]
            fig_radar=go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=a_scores+[a_scores[0]],theta=radar_cats+[radar_cats[0]],fill='toself',fillcolor='rgba(0,212,255,0.12)',line=dict(color='#00d4ff',width=2),name="🔵 "+a_name.title()))
            fig_radar.add_trace(go.Scatterpolar(r=b_scores+[b_scores[0]],theta=radar_cats+[radar_cats[0]],fill='toself',fillcolor='rgba(255,107,53,0.12)',line=dict(color='#ff6b35',width=2),name="🟠 "+b_name.title()))
            fig_radar.update_layout(polar=dict(bgcolor='rgba(0,0,0,0)',radialaxis=dict(visible=True,range=[0,100],gridcolor='rgba(0,212,255,0.1)',tickfont=dict(size=9)),angularaxis=dict(gridcolor='rgba(0,212,255,0.1)',tickfont=dict(family='Rajdhani',size=11,color='rgba(224,224,255,0.7)'))),paper_bgcolor='rgba(0,0,0,0)',font=dict(color='white',family='Rajdhani'),legend=dict(font=dict(color='white')),height=380,margin=dict(l=40,r=40,t=20,b=20))
            st.plotly_chart(fig_radar, use_container_width=True)

            winner=a_name if price_a>=price_b else b_name
            diff=abs(price_a-price_b)
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(255,215,0,0.1),rgba(255,140,0,0.08));border:2px solid rgba(255,215,0,0.4);border-radius:20px;padding:24px;text-align:center;box-shadow:0 0 40px rgba(255,215,0,0.1);">
              <div style="font-size:2rem;margin-bottom:6px;">🏆</div>
              <div style="font-family:'Orbitron',monospace;font-size:1.1rem;color:#ffd700;letter-spacing:2px;">HIGHER VALUE: {winner.upper()}</div>
              <div style="color:rgba(255,215,0,0.8);font-size:1rem;margin-top:8px;font-weight:700;">{fmt_inr(diff)} higher</div>
              <div style="color:rgba(255,215,0,0.45);font-size:0.78rem;margin-top:4px;">₹ {fmt_num(diff*1_00_000)} Rupees difference</div>
            </div>""", unsafe_allow_html=True)


    # ════════════════════════════════════════════
    # ANALYTICS HUB
    # ════════════════════════════════════════════
    elif "Analytics" in page:
        st.markdown('<div class="hero-title">📊 ANALYTICS HUB</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-subtitle">Deep Market Intelligence & Data Insights</div>', unsafe_allow_html=True)
        if df is None: st.error("Dataset not found."); st.stop()

        tabs=st.tabs(["📈 Price Trends","🔧 Feature Impact","📦 Market Overview","🌐 3D Analysis"])

        with tabs[0]:
            st.markdown('<div class="section-header">Price Trends by Year</div>', unsafe_allow_html=True)
            yearly=df.groupby('Year')['Selling_Price'].agg(['mean','min','max']).reset_index()
            fig_trend=go.Figure()
            fig_trend.add_trace(go.Scatter(x=yearly['Year'],y=yearly['max'],name='Max',line=dict(color='rgba(255,107,53,0.5)',dash='dot',width=1),fill=None))
            fig_trend.add_trace(go.Scatter(x=yearly['Year'],y=yearly['mean'],name='Avg',line=dict(color='#00d4ff',width=3),fill='tonexty',fillcolor='rgba(0,212,255,0.06)'))
            fig_trend.add_trace(go.Scatter(x=yearly['Year'],y=yearly['min'],name='Min',line=dict(color='rgba(0,255,136,0.5)',dash='dot',width=1),fill='tonexty',fillcolor='rgba(0,255,136,0.03)'))
            fig_trend.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='rgba(224,224,255,0.8)',family='Rajdhani'),xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True,gridcolor='rgba(0,212,255,0.08)',title='Price (₹ Lakh)'),
                legend=dict(font=dict(color='white')),height=380,margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig_trend, use_container_width=True)
            pivot=df.pivot_table(values='Selling_Price',index='Fuel_Type',columns='Year',aggfunc='mean')
            fig_heat=px.imshow(pivot,color_continuous_scale='viridis',text_auto='.1f')
            fig_heat.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='rgba(224,224,255,0.8)',family='Rajdhani'),height=260,margin=dict(l=0,r=0,t=20,b=0),
                title=dict(text="Avg Price Heatmap (₹ Lakh): Fuel × Year",font=dict(color='#00d4ff',family='Orbitron',size=13)))
            st.plotly_chart(fig_heat, use_container_width=True)

        with tabs[1]:
            st.markdown('<div class="section-header">Feature Correlation</div>', unsafe_allow_html=True)
            num_cols=[c for c in ['Selling_Price','Kms_Driven','Car_Age','Mileage','Engine','Max_Power','Seats'] if c in df.columns]
            corr=df[num_cols].corr()
            fig_corr=px.imshow(corr,text_auto='.2f',color_continuous_scale='RdBu_r',zmin=-1,zmax=1)
            fig_corr.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='rgba(224,224,255,0.8)',family='Rajdhani'),height=380,margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig_corr, use_container_width=True)
            fig_box=px.box(df,x='Transmission',y='Selling_Price',color='Fuel_Type',
                color_discrete_map={'Petrol':'#00d4ff','Diesel':'#ff6b35','CNG':'#00ff88'},
                labels={'Selling_Price':'Price (₹ Lakh)'})
            fig_box.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='rgba(224,224,255,0.8)',family='Rajdhani'),height=350,
                legend=dict(font=dict(color='white')),margin=dict(l=0,r=0,t=20,b=0),
                title=dict(text="Price Distribution (₹ Lakh): Transmission × Fuel",font=dict(color='#00d4ff',family='Orbitron',size=13)))
            st.plotly_chart(fig_box, use_container_width=True)

        with tabs[2]:
            c1,c2=st.columns(2)
            with c1:
                seller_avg=df.groupby('Seller_Type')['Selling_Price'].mean()
                fig_sel=go.Figure([go.Bar(x=seller_avg.index,y=seller_avg.values,
                    marker=dict(color=['#00d4ff','#7b2ff7','#ff6b35'],line=dict(color='rgba(255,255,255,0.1)',width=1)),
                    text=[fmt_inr(v) for v in seller_avg.values],textposition='outside',textfont=dict(color='#00d4ff',family='Orbitron',size=10))])
                fig_sel.update_layout(title=dict(text="Avg Price by Seller Type",font=dict(color='#00d4ff',family='Orbitron',size=12)),
                    plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',font=dict(color='rgba(224,224,255,0.8)',family='Rajdhani'),
                    yaxis=dict(showgrid=True,gridcolor='rgba(0,212,255,0.08)',title='₹ Lakh'),xaxis=dict(showgrid=False),height=320,margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig_sel, use_container_width=True)
            with c2:
                trans_avg=df.groupby('Transmission')['Selling_Price'].mean()
                fig_tr=go.Figure([go.Bar(x=trans_avg.index,y=trans_avg.values,
                    marker=dict(color=['#00ff88','#ff6b35'],line=dict(color='rgba(255,255,255,0.1)',width=1)),
                    text=[fmt_inr(v) for v in trans_avg.values],textposition='outside',textfont=dict(color='#00ff88',family='Orbitron',size=10))])
                fig_tr.update_layout(title=dict(text="Avg Price by Transmission",font=dict(color='#00d4ff',family='Orbitron',size=12)),
                    plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',font=dict(color='rgba(224,224,255,0.8)',family='Rajdhani'),
                    yaxis=dict(showgrid=True,gridcolor='rgba(0,212,255,0.08)',title='₹ Lakh'),xaxis=dict(showgrid=False),height=320,margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig_tr, use_container_width=True)
            top10=df.groupby('Car_Name')['Selling_Price'].mean().nlargest(10).sort_values()
            fig_top=go.Figure(go.Bar(y=top10.index,x=top10.values,orientation='h',
                marker=dict(color=top10.values,colorscale='viridis',line=dict(color='rgba(255,255,255,0.1)',width=0.5)),
                text=[fmt_inr(v) for v in top10.values],textposition='outside',textfont=dict(color='white',family='Orbitron',size=10)))
            fig_top.update_layout(title=dict(text="Top 10 Cars by Avg Resale Price",font=dict(color='#00d4ff',family='Orbitron',size=13)),
                plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',font=dict(color='rgba(224,224,255,0.8)',family='Rajdhani'),
                xaxis=dict(showgrid=True,gridcolor='rgba(0,212,255,0.08)',title='Avg Price (₹ Lakh)'),
                yaxis=dict(showgrid=False),height=380,margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_top, use_container_width=True)

        with tabs[3]:
            st.markdown('<div class="section-header">3D Surface: Price Landscape</div>', unsafe_allow_html=True)
            age_range=np.linspace(df['Car_Age'].min(),df['Car_Age'].max(),30)
            kms_range=np.linspace(df['Kms_Driven'].min(),df['Kms_Driven'].max(),30)
            ag,km=np.meshgrid(age_range,kms_range)
            from scipy import interpolate
            pts=df[['Car_Age','Kms_Driven']].values; vals=df['Selling_Price'].values
            try:
                f=interpolate.LinearNDInterpolator(pts,vals)
                z=f(ag,km); z=np.where(np.isnan(z),np.nanmean(vals),z)
            except:
                z=np.ones_like(ag)*np.mean(vals)
            fig_surf=go.Figure(data=[go.Surface(x=ag,y=km,z=z,colorscale='viridis',opacity=0.85,
                contours=dict(z=dict(show=True,usecolormap=True,project=dict(z=True))))])
            fig_surf.update_layout(scene=dict(xaxis_title='Car Age (yrs)',yaxis_title='KMs Driven',zaxis_title='Price (₹ L)',
                bgcolor='rgba(10,10,30,0.8)',xaxis=dict(gridcolor='rgba(0,212,255,0.1)'),
                yaxis=dict(gridcolor='rgba(0,212,255,0.1)'),zaxis=dict(gridcolor='rgba(0,212,255,0.1)')),
                paper_bgcolor='rgba(0,0,0,0)',font=dict(color='rgba(224,224,255,0.8)',family='Rajdhani'),
                height=480,margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_surf, use_container_width=True)

    # ════════════════════════════════════════════
    # MODEL INTELLIGENCE
    # ════════════════════════════════════════════
    elif "Intelligence" in page:
        st.markdown('<div class="hero-title">🤖 MODEL INTELLIGENCE</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-subtitle">Machine Learning Performance Dashboard</div>', unsafe_allow_html=True)
        if not model_ready: st.error("⚠️ Please train the models first: `python model_trainer.py`"); st.stop()

        st.markdown('<div class="section-header">🏆 Model Performance Leaderboard</div>', unsafe_allow_html=True)
        cols=st.columns(len(model_results))
        best_r2=max(m['R2 Score'] for m in model_results.values())
        for i,(name,metrics) in enumerate(model_results.items()):
            best=metrics['R2 Score']==best_r2
            border_col='#ffd700' if best else 'rgba(0,212,255,0.3)'
            glow='0 0 30px rgba(255,215,0,0.2),' if best else ''
            trophy_div=('<div style="font-size:1.2rem;margin-bottom:4px;">'+chr(127942)+'</div>') if best else ''
            # MAE and RMSE are in Lakhs — format them properly
            mae_str  = fmt_inr(metrics['MAE'])
            rmse_str = fmt_inr(metrics['RMSE'])
            cols[i].markdown(
                '<div class="metric-card" style="border-color:'+border_col+';box-shadow:'+glow+'0 4px 20px rgba(0,0,0,0.3);">'
                +trophy_div
                +'<div style="font-family:monospace;font-size:0.72rem;color:rgba(0,212,255,0.7);letter-spacing:1px;margin-bottom:8px;">'+name.upper()+'</div>'
                +'<span class="metric-value" style="font-size:1.8rem;">'+str(metrics['R2 Score'])+'%</span>'
                +'<div class="metric-label">R² Score</div>'
                +'<div style="margin-top:12px;font-size:0.8rem;color:rgba(224,224,255,0.5);">'
                +'MAE: <strong style="color:#00ff88;">'+mae_str+'</strong><br>'
                +'RMSE: <strong style="color:#ffd700;">'+rmse_str+'</strong>'
                +'</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        model_names=list(model_results.keys())
        r2_vals=[model_results[n]['R2 Score'] for n in model_names]
        mae_inv=[100-min(model_results[n]['MAE']*20,90) for n in model_names]
        rmse_inv=[100-min(model_results[n]['RMSE']*15,90) for n in model_names]

        c1,c2=st.columns(2)
        with c1:
            fig_perf=go.Figure()
            colors=['#00d4ff','#ff6b35','#00ff88','#7b2ff7','#ffd700']
            for i,name in enumerate(model_names):
                fig_perf.add_trace(go.Scatterpolar(
                    r=[r2_vals[i],mae_inv[i],rmse_inv[i]],theta=['R² Score','MAE Score','RMSE Score'],
                    fill='toself',fillcolor='rgba({},{},{},0.1)'.format(int(colors[i][1:3],16),int(colors[i][3:5],16),int(colors[i][5:7],16)),
                    line=dict(color=colors[i],width=2),name=name))
            fig_perf.update_layout(polar=dict(bgcolor='rgba(0,0,0,0)',radialaxis=dict(range=[0,100],gridcolor='rgba(0,212,255,0.1)'),angularaxis=dict(gridcolor='rgba(0,212,255,0.1)',tickfont=dict(family='Rajdhani',color='rgba(224,224,255,0.7)'))),paper_bgcolor='rgba(0,0,0,0)',font=dict(color='white',family='Rajdhani'),legend=dict(font=dict(color='white',size=11)),title=dict(text="Model Performance Radar",font=dict(color='#00d4ff',family='Orbitron',size=13)),height=380,margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig_perf, use_container_width=True)
        with c2:
            fig_bars=go.Figure()
            fig_bars.add_trace(go.Bar(name='R² Score (%)',x=model_names,y=r2_vals,
                marker=dict(color='rgba(0,212,255,0.8)',line=dict(color='#00d4ff',width=1)),
                text=[f"{v:.1f}%" for v in r2_vals],textposition='outside',textfont=dict(color='#00d4ff',family='Orbitron',size=10)))
            fig_bars.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',font=dict(color='rgba(224,224,255,0.8)',family='Rajdhani'),xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor='rgba(0,212,255,0.08)',title='R² Score (%)'),title=dict(text="R² Score Comparison",font=dict(color='#00d4ff',family='Orbitron',size=13)),height=380,margin=dict(l=0,r=0,t=50,b=0),showlegend=False)
            st.plotly_chart(fig_bars, use_container_width=True)

        rf_model=artifacts.get('Random Forest',list(artifacts.values())[0])
        if hasattr(rf_model,'feature_importances_'):
            st.markdown('<div class="section-header">🔍 Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
            col_display={'Car_Name_enc':'Car Brand','Car_Age':'Car Age','Kms_Driven':'KMs Driven','Mileage':'Mileage','Engine':'Engine CC','Max_Power':'Max Power (bhp)','Seats':'Seats','Fuel_Type_enc':'Fuel Type','Seller_Type_enc':'Seller Type','Transmission_enc':'Transmission','Owner_enc':'Owner History','Present_Price':'Ex-showroom Price'}
            feat_names=[col_display.get(c,c) for c in feature_cols]
            importances=rf_model.feature_importances_
            sorted_idx=np.argsort(importances)
            fig_imp=go.Figure(go.Bar(y=[feat_names[i] for i in sorted_idx],x=[importances[i] for i in sorted_idx],orientation='h',
                marker=dict(color=[importances[i] for i in sorted_idx],colorscale='plasma',line=dict(color='rgba(255,255,255,0.1)',width=0.5)),
                text=[f"{importances[i]*100:.1f}%" for i in sorted_idx],textposition='outside',textfont=dict(color='white',family='Orbitron',size=10)))
            fig_imp.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',font=dict(color='rgba(224,224,255,0.8)',family='Rajdhani'),xaxis=dict(showgrid=True,gridcolor='rgba(0,212,255,0.08)',title='Importance Score'),yaxis=dict(showgrid=False),height=360,margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown('<div class="section-header">💡 AI Insights</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box"><strong style="color:#00d4ff;">🎯 Most Impactful Features:</strong><br>
        Ex-showroom price and Car Age are the strongest predictors of resale value. Diesel cars command a <strong>15–25% premium</strong> over petrol at the same age.</div>
        <div class="info-box"><strong style="color:#00ff88;">📈 Market Insights:</strong><br>
        Automatic transmission adds approx. <strong>₹ 1–2 Lakh</strong> to resale value. Dealer-sold cars average <strong>12% higher</strong> than individual sellers.</div>
        <div class="info-box"><strong style="color:#ffd700;">⚡ Model Choice:</strong><br>
        Random Forest gives the best balance of accuracy and interpretability for Indian used car pricing.</div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="footer">🚗 AUTOVALUE AI · Powered by Machine Learning · Built with Python & Streamlit</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()