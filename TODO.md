# Analyse descriptive sur Rmarkdown
Au point?

# Feature eng
- Etude de la corrélation des variables? Eliminer une des deux weighted/non-weighted?
- Modifier également la weighted nebulosity

- Garder validation sur la dernière année ou implémenter crossval? (Perso je suis pour garder juste la dernière année, plus simple, déjà implémenté ça pour le RF, moins de temps pris pour faire tourner les modèles, après avec la crossval timeseries ca nous ferait des trucs à dire dans le rapport.. ptet juste l'implémenter pour un modèle de l'ensemble?)
- Implémenter la crossval, ou validation sur les 365 derniers jours?

# Models
- Faire méthode d'ensemble (MLP?) -> Besoin des pred sur train et test set de tous les modèles
- Modèles:
    - RF OK
    - GAM A faire
    - LSTM Voir si faisable, sinon juste MLP?
    - Bootstrap arbre?
    - XGBoost
    - ARIMA?
    - Modèle quantile?
    - Modèle linéaire du prof? Optimisé sur loss quantile?

# Idées rapport
- Importance des variables avec RF
- Validation OOB avec RF, mtry et ntree
- Crossval timeseries si implémentée

# NPO+++
- Mieux de réduire un peu les prédictions du fait de l'erreur pinball 0.8, faire genre un -500 sur le modèle final?
- Changer les cheat set en validation avant la fin, OOB pour RF