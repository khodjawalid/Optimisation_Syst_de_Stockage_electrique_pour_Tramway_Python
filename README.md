# Projet : Dimensionnement de stockage embarqué dans un tramway  

## **Description**
Ce projet consiste à dimensionner un système de stockage embarqué par batterie dans un tramway afin de récupérer et d'utiliser l'énergie de freinage. Il simule les conditions électriques d’un tramway alimenté par une ligne aérienne de contact (LAC) et explore comment un système hybride batterie/LAC peut améliorer les performances tout en limitant les coûts.  

## **Objectifs**
1. Comprendre le fonctionnement des réseaux ferroviaires électriques.
2. Modéliser le système électrique d’un tramway en mouvement.
3. Optimiser la capacité de la batterie en minimisant les chutes de tension.
4. Mettre en œuvre des algorithmes d’optimisation comme NSGA-2.  

## **Structure du projet**
- **data/** : Contient les fichiers de données (e.g., `marche_train.m`) pour les profils de déplacement du train.  
- **scripts/** : Implémentation des modèles électriques et algorithmes d’optimisation.  
  - `simulation_basique.py` : Simulation sans batterie.  
  - `simulation_batterie.py` : Simulation avec ajout d'un stockage.  
  - `nsga2_optimization.py` : Implémentation de l’algorithme génétique NSGA-2.  
- **results/** : Résultats des simulations et des optimisations.  

## **Étapes principales**
1. **Modélisation du système** :  
   - Simuler les tensions, courants et pertes Joule dans le réseau sans batterie.  
   - Ajouter un modèle pour la batterie avec des règles de gestion simples.  

2. **Dimensionnement de la batterie** :  
   - Évaluer l’impact de différentes capacités sur la chute de tension maximale.  
   - Utiliser Monte-Carlo pour explorer l’espace des solutions.  

3. **Optimisation NSGA-2** :  
   - Générer les solutions Pareto-optimales pour le compromis "capacité batterie" et "chute de tension maximale".  

## **Prérequis**
- Python 3.8+  
- Bibliothèques Python :  
  - `numpy`, `matplotlib`, `pandas`  
  - `deap` (pour NSGA-2)  
  - `scipy`  
