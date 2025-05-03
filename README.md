# Group 8 - Movie Recommender System

## Description

Ce projet contient un **analytics notebook** qui fournit une analyse approfondie des données disponibles dans le dataset MovieLens. L'objectif principal de cet outil est de permettre une exploration visuelle et statistique des données, facilitant ainsi la découverte de tendances et de patterns utiles pour la construction d'un système de recommandation performant.

## Fonctionnalités

- **Résumé des données** : Affichage de diverses informations générales, comme le nombre de films.
- **Visualisation des données** : Représentation graphique des données, comme la distribution des fréquences de notation (*3. Long-tail property*)
- **Matrice de notation** : Génération de la matrice de notation.

## Utilisation

1. **Installation des dépendances** :
   Assurez-vous d'avoir installé les dépendances nécessaires en utilisant le fichier `Pipfile` ou `requirements.txt` :
   ```bash
   pip install -r requirements.txt
   ```
2. **Choix du jeu de données** :
   Dans constants.py, changez la ligne *DATA_PATH = Path('data/small')* selon le jeu de données souhaité (*small*, *tiny* ou *test*).

3. **Exécution de l'analytics notebook** :
   Ouvrez et exécutez le fichier `analytics.ipynb` dans un environnement compatible comme Jupyter Notebook ou VS Code.

4. **Exploration des résultats** :
   Les visualisations produites, comme la distribution des fréquences des évaluations ou la matrice de sparsité, offrent une meilleure compréhension du comportement des données.

## Objectif d'un Analytics Notebook

Un analytics notebook est un outil interactif qui combine du code, des visualisations et des explications textuelles. Il est particulièrement utile pour :

- **Comprendre les données** : Fournir une vue d'ensemble claire et détaillée.
- **Communiquer les résultats** : Partager des insights avec des équipes techniques et non techniques.

## Auteurs

Ce projet a été développé par le Groupe 8 dans le cadre du cours de Recommender Systems.