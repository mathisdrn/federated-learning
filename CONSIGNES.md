# Projet : Apprentissage fédéré pour le diagnostic médical

## Travaux Pratiques – Federated Learning

## 1. Objectifs du projet

Ce projet a pour objectif de mettre en œuvre un système d'apprentissage fédéré (Federated Learning, FL) dans un contexte médical, en utilisant le framework Fluke.

À l'issue du projet, vous devez être capables de :

- comprendre et implémenter un protocole FedAvg dans un contexte médical ;
- gérer une forte hétérogénéité entre plusieurs clients ;
- intégrer un mécanisme de préservation de la vie privée ;
- étudier la fairness du modèle entraîné.

Le projet est réalisé en binôme.

## 2. Contexte et description générale

Dans le domaine médical, le partage de données des patients entre établissements est fortement limité pour des raisons de confidentialité, de réglementation et de sécurité. L'apprentissage fédéré permet d'entraîner un modèle global sur les données de plusieurs centres médicaux sans jamais centraliser les données brutes.

Dans ce projet, vous allez :

- choisir un jeu de données médical ;
- simuler plusieurs clients représentant des hôpitaux ou centres médicaux ;
- entraîner un modèle en fédéré avec Fluke ;
- étudier l'hétérogénéité des données, la préservation de la vie privée, et l'équité.

## 3. Jeux de données

Vous êtes libres de choisir un jeu de données dans un contexte médical pour réaliser votre projet. Les jeux de données ci-dessous sont proposés à titre d'exemple.

### 3.1. Breast Cancer Wisconsin (Diagnostic)

- Type : données tabulaires.
- Tâche : classification binaire (bénin / malin).
- Attributs démographiques : âge.
- Lien : https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

### 3.2. Heart Disease UCI

- Type : données tabulaires.
- Tâche : diagnostic de maladie cardiaque.
- Attributs démographiques : âge, sexe.
- Lien : https://archive.ics.uci.edu/ml/datasets/Heart+Disease

### 3.3. Diabetes 130-US hospitals

- Type : données tabulaires issues de dossiers hospitaliers.
- Tâche : prédiction de réadmission.
- Attributs démographiques : âge, sexe, origine ethnique.
- Lien : https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008

## 4. Cadre technique

- Langage : Python.
- Framework de deep learning : PyTorch ou équivalent.
- Framework d'apprentissage fédéré : Fluke.

## 5. Tâches à réaliser

### Partie 1 : Préparation des données

1.1 Chargement et pré-traitement du jeu de données choisi :

- séparation train / test ;
- normalisation ou standardisation ;
- gestion des valeurs manquantes.

1.2 Choisir un modèle d'apprentissage adapté aux données.

### Partie 2 : Mise en place de l'apprentissage fédéré avec Fluke

2.1 Définir un scénario où chaque client représente un hôpital ou un centre médical.

2.2 Répartir les données en plusieurs clients :

- Cas IID ;
- Cas non-IID.

2.3 Comparer les performances dans les deux scénarios et appliquer une méthode de traitement de l'hétérogénéité des données, en mesurant son coût et son impact sur la qualité du modèle.

### Partie 3 : Préservation de la vie privée

Implémenter un mécanisme de protection parmi les suivants :

- ajout de bruit (LDP, CDP) ;
- secure aggregation ;
- model pruning ;

Étudier l'impact de ces mécanismes sur les performances et mesurer leurs coûts.

### Partie 4 : Étude de l'équité et de la scalabilité

Étudier l'équité du modèle par rapport aux attributs démographiques présents dans les données (ex. : sexe, âge). Des métriques de fairness doivent être calculées et analysées.

Appliquer une méthode de mitigation du biais et évaluer son efficacité.

### Bonus : Apprentissage fédéré vertical (VFL)

On considère qu'en plus des hôpitaux participant à l'apprentissage fédéré, une compagnie d'assurance collabore également au processus d'apprentissage. Dans ce scénario, les hôpitaux et l'assureur partagent les mêmes patients, mais disposent de caractéristiques différentes.

L'objectif de ce bonus est d'implémenter un processus d'apprentissage fédéré vertical (Vertical Federated Learning, VFL).

## 6. Livrables

- Une présentation de 10 minutes ;
- Le code.
