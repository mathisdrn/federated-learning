# Synthèse du TP : Découverte de l'Apprentissage Fédéré avec Fluke

## 1\. Aperçu du Projet

L'objectif de cette séance de travaux pratiques (TP) est de découvrir et manipuler **Fluke**, un framework dédié à l'Apprentissage Fédéré (FL). À la fin de la séance, vous devriez être capable de configurer, exécuter et analyser des scénarios FL, allant de cas IID simples à des intégrations personnalisées complexes.

## 2\. Configuration de l'Environnement

Avant d'exécuter les expériences, vous devez établir un environnement FL fonctionnel.

  * **Prérequis Python :** Créez un environnement avec Python $\ge 3.10$ pour assurer la compatibilité.
  * **Installation :**
      * **Rapide (Mode Utilisateur) :**
        ```bash
        conda create -n fluke310 python=3.10
        conda activate fluke310
        pip install fluke-fl
        ```
      * **Mode Développement (Mode Éditable) :** Clonez le dépôt et installez avec `pip install -e`.
  * **Vérification :** Assurez-vous que l'installation est réussie en exécutant `fluke --help`.

## 3\. Tâches Principales & Expériences

### Phase 1 : Première Expérience (MNIST IID)

  * **Configuration :** Générez les fichiers de configuration par défaut pour l'expérience (`exp.yaml`) et l'algorithme (`fedavg.yaml`) en utilisant :
      * `fluke-get config exp`
      * `fluke-get config fedavg`
  * **Exécution :** Lancez une session d'entraînement d'Apprentissage Fédéré standard :
      * Commande : `fluke federation config/exp.yaml config/fedavg.yaml`.
  * **Visualisation :** Surveillez l'évolution des performances au cours des rounds d'entraînement.

### Phase 2 : Analyse Comparative

Vous devez comparer l'approche Fédérée avec deux autres paradigmes :

1.  **Entraînement Centralisé :** Exécutez un entraînement classique non fédéré en utilisant `fluke centralized ...`.
2.  **Décentralisé (Clients uniquement) :** Exécutez un entraînement local indépendant en utilisant `fluke clients-only ...`.
3.  **Analyse :** Comparez les courbes d'apprentissage et discutez des différences entre les approches FL, centralisée et clients uniquement.

### Phase 3 : Journalisation & Métriques

  * **Configuration du Logger :** Modifiez la section `logger` dans la configuration (les options standard incluent `Log`, `WandBLog`, `TensorboardLog`).
  * **Export des Données :** Configurez la sauvegarde des métriques pour générer deux fichiers CSV : un pour les métriques globales et un pour les métriques spécifiques aux clients.
  * **Tracés :**
      * Tracez la précision globale au fil des rounds.
      * Tracez la précision par client et comparez leurs trajectoires.

### Phase 4 : Scénarios Non-IID

Testez l'impact de la distribution hétérogène des données (Non-IID).

  * **Configuration :** Modifiez le paramètre `distribution` dans le fichier d'expérience pour utiliser une distribution de Dirichlet.
    ```yaml
    distribution:
      name: dir
      beta: 0.02
    ```
  * **Analyse :** Entraînez, sauvegardez les traces et comparez les performances avec le scénario IID de la Phase 1.

### Phase 5 : Atténuation de l'Hétérogénéité (SCAFFOLD)

Utilisez des algorithmes avancés pour gérer l'hétérogénéité des données.

  * **Configuration :** Récupérez la configuration SCAFFOLD en utilisant `fluke get scaffold`.
  * **Exécution :** Réexécutez l'expérience Non-IID en utilisant SCAFFOLD au lieu de FedAVG.
  * **Évaluation :** Comparez les performances avec FedAVG Non-IID et répondez : *Est-ce que SCAFFOLD améliore l'apprentissage dans cette configuration ? Pourquoi ?*

### Phase 6 : Exploration des Jeux de Données Internes

  * **Sélection :** Choisissez un jeu de données différent déjà inclus dans Fluke (par exemple, CIFAR-10, Shakespeare, FEMNIST).
  * **Adaptation :** Sélectionnez un modèle approprié (par exemple, CNN pour les images, LSTM pour le texte).
  * **Exécution :** Exécutez et analysez un scénario FL complet.

## 4\. Mini-Projet : Intégration Personnalisée

L'objectif final est d'étendre Fluke en intégrant des ressources externes.

  * **Exigences :**
    1.  **Importer un Jeu de Données Externe :** Intégrez un jeu de données non fourni par Fluke.
    2.  **Définir un Modèle Personnalisé :** Implémentez un modèle de votre choix (CNN, Transformer, MLP, etc.).
    3.  **DataContainer :** Construisez un `DataContainer` conforme aux spécifications de Fluke.
  * **Scénario :** Développez un scénario fédéré complet (distribution, algorithme d'agrégation, exécution) et analysez les performances globales/locales.
  * **Ressources :**
      * Tutoriel sur les Jeux de Données : [Fluke Custom Dataset](https://makgyver.github.io/fluke/examples/tutorials/flukecustomataset.html).
      * Tutoriel sur les Modèles : [Fluke Custom NN](https://makgyver.github.io/fluke/examples/tutorials/flukecustom.nn.html).