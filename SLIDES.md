---
marp: true
theme: gaia
class: lead
backgroundColor: #fff
backgroundImage: url('https://marp.app/assets/hero-background.svg')
paginate: true
math: true
style: |
  section { font-size: 24px; }
  h1 { font-size: 40px; color: #2c3e50; }
  h2 { font-size: 30px; color: #34495e; }
  table { font-size: 18px; }
  th { background-color: #ecf0f1; }
---

# Apprentissage Fédéré pour le Diagnostic Médical
### Projet : Implémentation avec le Framework Fluke

**Auteurs :** Mathis Derenne, Marie Saccucci
**Contexte :** Prédiction de Réadmission (Diabète)

---

## 1. Introduction et Problématique

**Le contexte**
* Les données médicales sont sensibles et fragmentées entre différents hôpitaux.
* La centralisation des données (Data Lake) pose des problèmes de confidentialité (GDPR/HIPAA).

**Notre Objectif**
* Entraîner un modèle prédictif **sans que les données ne quittent les hôpitaux**.
* Simuler des conditions réelles : Hétérogénéité, Confidentialité, Équité.

**Le Framework**
* **Fluke** (Federated Learning Utility for Knowledge Exchange).

---

## 2. Présentation du Dataset

**Diabetes 130-US Hospitals for Years 1999-2008**

* **Tâche :** Classification (prédiction de la réadmission hospitalière pour le diabète).
* **Caractéristiques :**
    * Données tabulaires multivariées.
    * 47 features.
    * 101 766 instances.
* **Attributs sensibles :** âge, genre, race des patients.

---

## 3. Configuration Expérimentale

**Jeu de Données**
* **Diabetes 130-US Hospitals** (1999-2008).
* **Tâche :** Classification binaire (Réadmission < 30 jours).

**Paramètres Globaux**
* **Clients :** 5 (Simulations standards) à 50 (Scalabilité).
* **Rounds :** 10 rounds globaux.
* **Modèle :** Réseau de neurones simple.
* **Optimiseur :** SGD (Learning Rate = 0.01).

---

## 4. Scénario 1 : Base de Référence (IID)

**Hypothèse "Idéale"**
* Les données sont distribuées uniformément entre les hôpitaux (Indépendantes et Identiquement Distribuées).
* Algorithme : **FedAvg** (Moyenne classique).

**Résultats**
* **Précision :** $63.21\%$
* **F1-Score :** $0.6236$
* **Temps :** ~62s

> **Analyse :** Le modèle converge rapidement car les propriétés statistiques sont cohérentes entre tous les clients.

---

## 5. Scénario 2 : Le Défi de l'Hétérogénéité (Non-IID)

**La Réalité**
* Les hôpitaux ont des démographies biaisées (Distribution de Dirichlet).
* Problème : **"Weight Divergence"** (Les modèles locaux partent dans des directions opposées).

**Résultats (FedAvg sur Non-IID)**
* **Précision :** $62.86\%$ ($\downarrow$ vs Base)
* **Stabilité :** Moins bonne convergence.

> **Analyse :** Le moyennage simple (FedAvg) perd en efficacité quand les distributions locales diffèrent trop.

---

## 6. Scénario 3 : Solution Robustesse (FedProx)

**L'Algorithme FedProx**
* Ajout d'un **terme proximal** ($\mu = 0.1$) à la fonction de perte.
* **But :** Empêcher le modèle local de trop s'éloigner du modèle global.

**Résultats**
* **Précision :** $61.95\%$
* **Temps :** ~74s ($\uparrow 19\%$ coût de calcul).

> **Compromis :** On sacrifie un peu de précision immédiate et de temps de calcul pour gagner en **stabilité** à long terme.

---

## 7. Scénario 4 : Confidentialité (Differential Privacy)

**Méthode : DP-FedAvg**
* Ajout de bruit aux gradients + Clipping pour empêcher la reconstruction des données patients.

**Résultats**
* **Confidentialité Modérée (Bruit=1.0) :** Précision $62.03\%$.
* **Coût technique :** Temps $\times 4.5$ (~282s).

> **Compromis Confidentialité-Utilité :** La sécurité mathématique forte a un coût très élevé en performance (latence).

---

## 8. Scénario 5 : Équité et Scalabilité

**Objectif A : Équité (Fairness)**
* Régularisation pour éviter la discrimination (Race/Genre).
* **Résultat :** Précision maintenue ($63.23\%$) avec un biais quasi-nul ($\Delta \approx 0.05$).

**Objectif B : Scalabilité**
* Test sur **50 Clients** (Participation partielle 20%).
* **Résultat :** Le système passe à l'échelle ($61.01\%$ de précision) malgré la fragmentation.

---

## 9. Synthèse des Résultats

| Scénario | Algo | Précision | F1 | Temps | Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Base (IID)** | FedAvg | **63.21%** | 0.62 | Rapide | Idéal théorique |
| **2. Hétérogène** | FedAvg | 62.86% | 0.60 | Rapide | Sensible aux biais |
| **3. Robuste** | **FedProx** | 61.95% | 0.59 | Moyen | Stable mais coûteux |
| **4. Privé (DP)** | DP-FedAvg | 62.03% | 0.59 | **Lent** | Coût sécu. élevé |
| **5. Équitable** | Fairness | **63.23%** | - | Rapide | **Bon compromis** |

---

## Conclusion

1.  **Faisabilité :** Le FL est viable pour le diagnostic du diabète (~63% acc).
2.  **Robustesse :** **FedProx** est nécessaire pour les réseaux hétérogènes, malgré le coût.
3.  **Confidentialité :** La **Differential Privacy** protège les patients mais ralentit fortement l'entraînement (x4.5).
4.  **Responsabilité :** L'intégration de contraintes d'équité est possible sans perte de performance.

### Avez-vous des questions ?