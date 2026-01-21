### Rapport technique : Apprentissage Fédéré pour le Diagnostic Médical

**Projet :** Implémentation de l'Apprentissage Fédéré avec le Framework Fluke  
**Contexte :** Diagnostic Médical (Prédiction de Réadmission pour le Diabète)  
**Auteurs :** Mathis Derenne, Marie Saccucci

---

## 1. Introduction et Objectifs

Ce document résume l'implémentation technique et les résultats de notre projet d'Apprentissage Fédéré (FL). L'objectif était de simuler un environnement d'entraînement décentralisé à travers plusieurs hôpitaux (clients) pour prédire la réadmission des patients tout en préservant la confidentialité des données et en traitant l'hétérogénéité des données.

Nous avons utilisé le jeu de données **Diabetes 130-US Hospitals**, en nous concentrant sur la classification binaire de la réadmission.

## 2. Configuration Expérimentale

*   **Jeu de données :** Diabetes 130-US Hospitals (1999-2008)
*   **Framework :** Fluke (Federated Learning Utility for Knowledge Exchange)
*   **Paramètres Globaux :**
    *   Nombre de Clients : 5 (défaut), 50 (test de scalabilité)
    *   Rounds Globaux : 10
    *   Optimiseur : SGD (Taux d'apprentissage = 0.01)
    *   Taille de Batch : 32

---

## 3. Scénarios et Résultats

### Scénario 1 : Base de référence (Données IID)
**Objectif :** Établir une base de performance où les données des patients sont distribuées uniformément (Indépendantes et Identiquement Distribuées) entre les hôpitaux.
**Algorithme :** FedAvg (Federated Averaging).

*   **Résultats :**
    *   **Précision (Accuracy) :** 63.21%
    *   **Macro F1-Score :** 0.6236
    *   **Temps d'exécution :** ~62.5s
*   **Analyse :** Ce scénario représente le cas "idéal". Le modèle converge bien car les propriétés statistiques sont cohérentes entre tous les clients.

### Scénario 2 : Le Défi de l'Hétérogénéité (Non-IID)
**Objectif :** Simuler des conditions réelles où les hôpitaux ont des démographies de patients biaisées (Non-IID). Nous avons utilisé une distribution de Dirichlet pour induire un biais dans les données.
**Algorithme :** FedAvg.

*   **Résultats :**
    *   **Précision :** 62.86% (-0.35% vs Base)
    *   **Macro F1-Score :** 0.6033 (-0.02 vs Base)
    *   **Temps d'exécution :** ~59.4s
*   **Analyse :** La performance chute légèrement en raison de la "divergence des poids". Les clients mettent à jour leurs modèles locaux dans des directions différentes en fonction de leurs distributions de données locales spécifiques, rendant le moyennage moins efficace.

### Scénario 3 : Gérer l'Hétérogénéité (FedProx)
**Objectif :** Atténuer les effets des données Non-IID en utilisant **FedProx**, qui ajoute un terme proximal à la fonction de perte pour limiter la déviation du modèle local par rapport au modèle global.
**Algorithme :** FedProx (mu = 0.1).

*   **Résultats :**
    *   **Précision :** 61.95%
    *   **Macro F1-Score :** 0.5956
    *   **Temps d'exécution :** ~74.0s (+19% de coût)
*   **Analyse :** 
    *   **Compromis :** FedProx stabilise efficacement l'entraînement mais engendre un coût de calcul (temps d'exécution plus élevé) et une légère baisse de précision pour cette configuration spécifique à court terme (10 rounds).
    *   Le "terme proximal" agit comme un régularisateur, empêchant les clients de sur-apprendre sur leurs données locales biaisées, ce qui est crucial pour la stabilité à long terme dans des environnements hautement hétérogènes.

### Scénario 4 : Préservation de la Confidentialité (Differential Privacy)
**Objectif :** Implémenter des garanties strictes de confidentialité en utilisant la Differential Privacy (DP). Nous ajoutons du bruit aux gradients pour empêcher la reconstruction des données individuelles des patients.
**Algorithme :** DP-FedAvg.

*   **Résultats :**
    *   **Confidentialité Modérée (Bruit=1.0) :** Précision 62.03% | F1 0.5942 | Temps ~282s
    *   **Confidentialité Élevée (Bruit=2.0) :** Précision 61.14% | F1 0.5673 | Temps ~269s
*   **Analyse :** 
    *   **Compromis Confidentialité-Utilité :** L'augmentation du bruit (confidentialité plus forte) réduit directement l'utilité du modèle (baisse de la précision/F1).
    *   **Coût :** Le temps d'exécution augmente considérablement (~4.5x) en raison du surcoût du découpage des gradients par échantillon (clipping) et de la génération de bruit (moteur Opacus).

### Scénario 5 : Équité et Scalabilité
**Objectif A (Équité) :** S'assurer que le modèle ne discrimine pas sur la base d'attributs protégés (ex. : race/genre). Nous avons appliqué un terme de régularisation d'équité (Lambda=0.5).
*   **Résultats :**
    *   **Précision :** 63.23% (comparable à la base)
    *   **Différence de Parité Démographique :** 0.052
    *   **Différence d'Égalité des Chances :** 0.059
*   **Analyse :** Le modèle maintient une précision élevée tout en gardant des métriques d'équité basses (proche de 0 est idéal), suggérant que la stratégie d'atténuation a réussi à équilibrer performance et équité.

**Objectif B (Scalabilité) :** Tester la robustesse du système avec plus de clients.
*   **Configuration :** 50 Clients, 20% de participation par round (10 clients sélectionnés/round), 5 Rounds.
*   **Résultats :**
    *   **Précision :** 61.01%
    *   **Temps d'exécution :** 9.54s (rapide en raison du moins grand nombre de rounds)
*   **Analyse :** Le système passe à l'échelle efficacement. Même avec une participation éparse (ne voyant qu'une fraction des clients à chaque round), le modèle apprend des motifs utiles (précision > 60%), validant l'approche pour des réseaux hospitaliers plus larges.

---

## 4. Conclusion pour la Présentation

1.  **Faisabilité :** Le FL est viable pour le diagnostic médical, atteignant ~63% de précision sans centraliser les données.
2.  **Robustesse :** L'hétérogénéité pose un défi ; FedProx offre de la stabilité mais nécessite un réglage minutieux.
3.  **Confidentialité :** La Differential Privacy est efficace mais coûteuse (4x temps d'exécution) et impacte l'utilité (~2% de perte de précision).
4.  **Responsabilité Sociale :** Nous avons démontré que des contraintes d'équité peuvent être intégrées sans pénalités de performance significatives.
