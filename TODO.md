# Analyse descriptive sur Rmarkdown
Au point? -> oui mais en fait beaucoup plus intéressant la recherche de bruit blanc en résidus de gam (d'un pdv analyse descriptive interprétation des données)

# Feature eng
- variable one hot été hiver
- variable wkd en plus de lundi_vendredi
- j'ai remis le toy ça améliore un peu mon modèle

- Changer représentation date, voir RBF, Fourrier
- Tester TFT, faire plusieurs datasets pour les représentations de date? tester TFT avec truc simple?
- Etude de la corrélation des variables? Eliminer une des deux weighted/non-weighted?  => regarder correlation après feature engineering
- Modifier également la weighted nebulosity

- Garder validation sur la dernière année ou implémenter crossval? (Perso je suis pour garder juste la dernière année, plus simple, déjà implémenté ça pour le RF, moins de temps pris pour faire tourner les modèles, après avec la crossval timeseries ca nous ferait des trucs à dire dans le rapport.. ptet juste l'implémenter pour un modèle de l'ensemble?)
- Implémenter la crossval, ou validation sur les 365 derniers jours?

- Cross val déjà implémenté par prof, oui c'est plus long mais aussi plus significatif autant la garder je dis, après pr les GAM ya rien a changer pr tft ou autre on peut garder la dernière année

# Models
- Faire méthode d'ensemble (MLP?) -> Besoin des pred sur train et test set de tous les modèles
- Modèles:
    - RF OK
    - GAM En train de faire l'affinage des variables et l'explication dans rapport
    - LSTM Voir si faisable, sinon juste MLP?
    - Bootstrap arbre? => RF suffit 
    - XGBoost => prioritaire après RF je pense
    - ARIMA? => kalman arima a faire pr rectifier residus gam => tt seul à voir
    - Modèle quantile? => oui dans agrégation avec plusieurs modèles (quantiles autour de 0.8)
    - Modèle linéaire du prof? Optimisé sur loss quantile? => pq pas l'optimiser un peu et le mettre dans l'agregation

# Idées rapport
- Importance des variables avec RF
- choix variables gam et justification théorique
- Validation OOB avec RF, mtry et ntree
- Crossval timeseries si implémentée

# NPO+++
- Mieux de réduire un peu les prédictions du fait de l'erreur pinball 0.8, faire genre un -500 sur le modèle final?
- Changer les cheat set en validation avant la fin, OOB pour RF  => pas compris 
- Mettre Data/covid_data.csv sur github
