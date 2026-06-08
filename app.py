"""
app.py - Application Streamlit pour la prediction du prix de voiture d'occasion
Dataset : eBay Kleinanzeigen | Projet ML MSDE7
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(
    page_title="Prediction Prix Voiture",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_model():
    pipeline = joblib.load("car_price_pipeline.joblib")
    with open("model_metadata.json") as f:
        meta = json.load(f)
    return pipeline, meta

try:
    pipeline, meta = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Impossible de charger le modele : {e}")
    st.info("Executez d'abord le notebook pour generer car_price_pipeline.joblib et model_metadata.json.")

st.markdown("""
<style>
    .main-title { font-size:2.2rem; font-weight:700; color:#1f4e79; text-align:center; margin-bottom:4px; }
    .subtitle   { font-size:1rem; color:#666; text-align:center; margin-bottom:24px; }
    .pred-box   {
        background: linear-gradient(135deg, #1f4e79, #2980b9);
        color:white; padding:24px; border-radius:12px; text-align:center;
    }
    .pred-value { font-size:2.8rem; font-weight:800; }
    .pred-label { font-size:.9rem; opacity:.8; margin-bottom:6px; }
    .conf-box   {
        background:#f0f9ff; border-left:4px solid #2980b9;
        padding:14px; border-radius:6px;
    }
    .summary    {
        background:#fafafa; border:1px solid #ddd;
        padding:12px 18px; border-radius:8px; font-size:.88rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Prediction du Prix de Revente - Voiture d\'Occasion</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Estimation basee sur le dataset eBay Kleinanzeigen (Allemagne)</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Modele ML")
    if model_loaded:
        st.success("Modele charge avec succes")
        st.markdown(f"**Algorithme :** {meta['nom_modele']}")
        st.markdown(f"**R2 (test) :** {meta['r2_test']}")
        st.markdown(f"**RMSE (test) :** {int(meta['rmse_test']):,} EUR")
        st.markdown(f"**MAE (test) :** {int(meta['mae_test']):,} EUR")
        st.divider()
    st.markdown("""### A propos
Predit le prix de revente a partir des caracteristiques du vehicule.

**Dataset :** eBay Kleinanzeigen
**Taille :** 371 528 annonces
**Projet :** MSDE7 - Module ML
""")

if model_loaded:
    st.subheader("Caracteristiques du vehicule")
    col1, col2, col3 = st.columns(3)

    with col1:
        brand = st.selectbox(
            "Marque",
            options=meta["marques"],
            index=meta["marques"].index("volkswagen") if "volkswagen" in meta["marques"] else 0,
        )
        vehicle_type = st.selectbox(
            "Type de carrosserie",
            options=[""] + meta["types_caross"],
            index=0,
            help="Laisser vide si inconnu"
        )
        fuel_type = st.selectbox(
            "Carburant",
            options=[""] + meta["types_carb"],
            index=0,
        )
        gearbox = st.selectbox(
            "Boite de vitesses",
            options=[""] + meta["types_boite"],
            index=0,
        )

    with col2:
        model_car = st.selectbox(
            "Modele",
            options=[""] + meta["modeles"][:100],
            index=0,
            help="Top 100 modeles"
        )
        year = st.slider(
            "Annee d'immatriculation",
            min_value=1980, max_value=2016, value=2010
        )
        month_reg = st.selectbox(
            "Mois d'immatriculation",
            options=list(range(0, 13)),
            index=0,
            help="0 = inconnu"
        )
        not_repaired = st.selectbox(
            "Dommages non repares ?",
            options=[""] + meta["dommages"],
            index=0,
        )

    with col3:
        kilometer = st.number_input(
            "Kilometrage (km)",
            min_value=0, max_value=300_000, value=80_000, step=5000
        )
        power_ps = st.number_input(
            "Puissance (CV)",
            min_value=1, max_value=500, value=100, step=5
        )
        postal_code = st.number_input(
            "Code postal (Allemagne)",
            min_value=10000, max_value=99999, value=50000,
        )

    st.divider()
    predict_btn = st.button("Estimer le prix", type="primary", use_container_width=True)

    if predict_btn:
        car_age = 2016 - year

        input_df = pd.DataFrame([{
            "kilometer"          : kilometer,
            "powerPS"            : power_ps,
            "monthOfRegistration": month_reg,
            "age_vehicule"       : car_age,
            "postalCode"         : postal_code,
            "vehicleType"        : vehicle_type if vehicle_type else None,
            "gearbox"            : gearbox if gearbox else None,
            "model"              : model_car if model_car else None,
            "fuelType"           : fuel_type if fuel_type else None,
            "brand"              : brand,
            "notRepairedDamage"  : not_repaired if not_repaired else None,
        }])

        log_pred   = pipeline.predict(input_df)[0]
        price_est  = np.expm1(log_pred)
        rmse       = meta['rmse_test']
        price_low  = max(0, price_est - rmse)
        price_high = price_est + rmse

        st.subheader("Recapitulatif")
        st.markdown(f"""
        <div class="summary">
        <b>Marque :</b> {brand} &nbsp;|&nbsp;
        <b>Modele :</b> {model_car or '-'} &nbsp;|&nbsp;
        <b>Type :</b> {vehicle_type or '-'} &nbsp;|&nbsp;
        <b>Annee :</b> {year} (age : {car_age} ans) &nbsp;|&nbsp;
        <b>Kilometrage :</b> {kilometer:,} km<br>
        <b>Puissance :</b> {power_ps} CV &nbsp;|&nbsp;
        <b>Carburant :</b> {fuel_type or '-'} &nbsp;|&nbsp;
        <b>Boite :</b> {gearbox or '-'} &nbsp;|&nbsp;
        <b>Dommages :</b> {not_repaired or '-'} &nbsp;|&nbsp;
        <b>CP :</b> {postal_code}
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Estimation du prix")
        c1, c2 = st.columns([2, 1])

        with c1:
            st.markdown(f"""
            <div class="pred-box">
                <div class="pred-label">Prix de revente estime</div>
                <div class="pred-value">{price_est:,.0f} EUR</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="conf-box">
                <b>Intervalle +/- 1 RMSE</b><br><br>
                <b>Bas :</b> {price_low:,.0f} EUR<br>
                <b>Haut :</b> {price_high:,.0f} EUR<br><br>
                <b>RMSE modele :</b> {rmse:,.0f} EUR<br>
                <b>R2 modele :</b> {meta['r2_test']}
            </div>
            """, unsafe_allow_html=True)

        confidence = min(100, meta["r2_test"] * 100)
        st.progress(confidence / 100, text=f"Confiance : {confidence:.1f}% (R2 du modele)")

        if price_est < 2000:
            tranche = "moins de 2 000 EUR (entree de gamme)"
        elif price_est < 5000:
            tranche = "2 000 - 5 000 EUR"
        elif price_est < 10000:
            tranche = "5 000 - 10 000 EUR"
        elif price_est < 20000:
            tranche = "10 000 - 20 000 EUR"
        else:
            tranche = "plus de 20 000 EUR (haut de gamme)"

        st.info(f"Tranche estimee : {tranche}")

else:
    st.warning("Modele non charge. Executez le notebook pour generer les fichiers necessaires.")
