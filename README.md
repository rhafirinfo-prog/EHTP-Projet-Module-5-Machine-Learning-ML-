# Prediction du Prix de Vente d'une Voiture d'Occasion

## Description du projet

Ce projet met en oeuvre un pipeline complet de Machine Learning pour predire le prix de revente de voitures d'occasion, a partir d'un dataset d'annonces issues d'une plateforme allemande (eBay Kleinanzeigen). La tache est une regression supervisee : on cherche a estimer la valeur numerique de la variable cible `price` (prix en EUR) a partir de 11 features decrivant les caracteristiques du vehicule.

---

## Application deployee

[Lien Streamlit Cloud](https://youssefrhafir.streamlit.app/)

---

## Structure du projet

```
notebook.ipynb              # Pipeline ML complet (EDA, preprocessing, modelisation, evaluation)
app.py                      # Application Streamlit
car_price_pipeline.joblib   # Pipeline serialise (genere par le notebook)
model_metadata.json         # Metadonnees du modele (genere par le notebook)
requirements.txt            # Dependances Python
rapport.pdf                 # Rapport detaille du projet
README.md                   # Ce fichier
```

---

## Installation et execution

```bash
# github
https://github.com/rhafirinfo-prog/EHTP-Projet-Module-5-Machine-Learning-ML-
cd car-price-predictor

# Installer les dependances
pip install -r requirements.txt

# Telecharger le dataset
# Placer autos.csv a la racine du projet

# Executer le notebook (genere car_price_pipeline.joblib et model_metadata.json)
jupyter notebook notebook.ipynb

# Lancer l'application Streamlit
streamlit run app.py
```

---

## Pipeline ML

| Etape | Description |
|-------|-------------|
| EDA | Analyse exploratoire : distribution, correlations, valeurs aberrantes |
| Preprocessing | Imputation (mediane / mode), encodage OHE, normalisation StandardScaler |
| Feature Engineering | Age du vehicule (2016 - annee), transformation log(price) |
| Modelisation | 12 algorithmes evalues (Ridge, Decision Tree, RF, XGBoost, LightGBM...) |
| Cross-validation | 5-fold KFold sur sous-echantillon de 50 000 lignes |
| Tuning | RandomizedSearchCV (20 iter, 3-fold) sur les 3 meilleurs modeles |
| Deploiement | Pipeline serialise joblib + application Streamlit |

---

## Dataset

- **Source** : eBay Kleinanzeigen (annonces de voitures d'occasion en Allemagne)
- **Taille** : 371 528 lignes x 21 colonnes
- **Variable cible** : `price` (en EUR)
- **Variables utilisees** : brand, model, vehicleType, yearOfRegistration, gearbox, powerPS, kilometer, monthOfRegistration, fuelType, notRepairedDamage, postalCode

---

## Auteur

**YOUSSEF RHAFIR** MSDE7 
