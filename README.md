projet – Analyse d'Émotions en Temps Réel
Ce projet consiste à développer une application web capable de détecter les émotions faciales d'un utilisateur en temps réel à l'aide de la webcam. L'application repose sur une architecture full-stack intégrant un frontend en React.js et un backend en Flask (Python), avec un modèle d'intelligence artificielle entraîné pour la classification d'expressions faciales.

L’utilisateur accède à une interface intuitive qui active la webcam, capture automatiquement des images toutes les 3 secondes, et envoie ces images au backend. Le backend traite l'image à l'aide d’un modèle de deep learning basé sur un réseau de neurones convolutif (CNN) et renvoie l’émotion prédite. L’émotion est ensuite affichée dynamiquement côté frontend.

Le projet est entièrement conteneurisé avec Docker et orchestré via Docker Compose, permettant un déploiement rapide et cohérent sur n’importe quel environnement. Un workflow GitHub Actions assure l’intégration continue (CI) : à chaque push, les tests et le build sont automatiquement lancés pour vérifier la validité du code.

Ce projet illustre l'intégration de l'intelligence artificielle dans une application web moderne, tout en appliquant des pratiques DevOps comme la CI/CD et la conteneurisation, ce qui le rend pertinent dans un contexte de développement professionnel.
