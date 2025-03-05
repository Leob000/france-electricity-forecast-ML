# Cours 2 mode pred :
#   
#   
#   
#     régression linéaire 
# X1,...,Xn dans vecteur X     Y
# -l:R2->R+ f° de coût 
# -espace de fonction dans lequel la sol° est cherché (lestimateur) => 
# - objectif min(E(loss)) dans espace de f°
# objectif inconnu => estimation via l'échantillon de réalisation (Y,f(Y))
# Y=(y1,..,yn) vecteur taille n      X matrice n*p 
# estimateur 1/n sum(loss(Yi,f(X[,i])))
# 
# modèle linéaire  : yi=B*xi+epsi  (modèle additif)  modèle multiplicatif c'est *epsilon
# epsi indépandents d'esperance nulle variance cst
# en passant au log un multiplicatif ça devient additif => comparer les deux additifs 
# multiplicatif = variation croissante => terme t*eps dans modèle de variance t**2*sigma**2
# variance saisonnière => normaliser les donnéees en f° de la saison
# 
# 
# f° perte quadratique => eps gaussien 
# on veut estimer le beta optimal => emv disponible dans les données
# slide 11 12
# multivarié  matrice X de rang p => pas de relation linéaire entre les x ( pas de corrélation importante)
# => nettoyer la matrice X sinon inversion de matrice => valeur proche de 0 inversé tend vers l'infini => merdoie
# beta estimateur MCO (moindre carré ordinaire) =(XtX)-1XtY pr moindre carrés de base 
# beta MCO non biaisé (E(BMCO)=B) variance = (XtX)-1*sigma**2
# 
# On peut élargir au EMV gaussien mais aussi cas exponentiel
# si g(E(Y))=X*B on peut faire un modèle GLM 
# SLIDE 18 exemples liste de modèle estimables (poisson gamma binomiale gaussienne)
# ce qu'on fait dans ce challenge c'est la gaussienne
# modèle généralisé estimateur optimal dépend de g => pas de calcul direct de la varaisemblance => algo d'opti de newton
# yt=at +bcos(wt)+epst    epst=ut*(Bt+gammacos(wt)) possibilité d'avoir ça avec les deux qui suivent la même loi ???
# 
# choisir variables rentrant dans estimation de varaicne et esperance soigneusement
# 
# 
#       Regression pénalisé 
# on prédit yn avec notre modèle et on le compare avec yn réel avec MSE
# E((yn-y^n)**2)=sigma**2+Xn*Var(Beta^)Xn      +(E(beta-beta^))**2
# erreur en prévision        terme de vraiance terme de biais
# on doit trouver le meilleur compromis entre minimiser variance ou biais 
# 
# modèle peu complexe prédiction pourrie => biais haut variance faible
# modèle complexe  petite erreur de prévision  biais bas variance haute (chopper le moment ou on minimise l'erreur de prediction et la variance)
# 
# si X1 X2 fortement corrélé on peut faire un modèle bizarre ou on a pas de biais mais une forte variance => phénom_ène pathologique
# => imposer contraintes sur coefficients  => norme < nbr par ex
# C'est la REGRESSION RIDGE
#  on imposer norme carré de beta =<cst
#  => argmin MSE +lambda*norme carré de beta
#  beta dans une sphère de rayon r
#  plus lambda grand plus r petit 
#  f° convexe en beta => derivable optimum trouvable 
#  beta ^ =  (XtX+lambda*idp)-1XtY    (le lambda peut permettre l'inversion de la matrice)
#  E(B^)=B-lambda(XtX+lambdaIdp)-1*B
#  Var(B^)=...
# => minimisation validation croisé 
# 
# REGRESSION LASSO
# au lieu de mettre terme quadratique norme carré de beta on met norme 1 de beta 
# => pénalité L1 = réduction du nbr de coeff Betai
# lambda petit grd variance faible biais  et inversement
# lambda cool parce que sélection de modèle devenu selection de lambda 
# 
# sélection du meilleur modèle :
# splittage train test => verifier qu'ils ont la même loi
# données dépendante temporellement entre elles => ajustement à faire pr technique cross validation 
# technique de sélection de modèle => AIC BIC CP de malose => min(erre+penalisation(complexité cp))  penalisation=2*cp
#   => prblm évaluer cp => dimension du vecteur beta pr reg linéaire, 
#      pr regression ridge lambda=0 = reg linaire, lambda infini cp=0 entre les deux cp indexé par lambda mais plus petite que celle de reg linéaire
#      moindre carrés =reg linéaire cp=nbr param  Y^=X(XtX)-1XtY=AY projection linéaire, trace(A)=p (=rg(X))
#     en ridge trace de X(XtX+lambdaID)-1Xt = cp = ddl degré de liberté 
#     nbr de ddl de espace sur lequel je projjète est inversement proportionnel à lambda 
#     lien entre lambda et cp
#      
# critère de validation croisé 
# => on enlève une obs on entraine sur reste on estime sur cette obs on regarde l'erreur => n modèle de reg lin = couteux 
# = alternative => erreur validation croisé =erreur modèle complet / 1-iéme coeff de matrice prediction A
# => test != lambda et chaque lambda un calcul de modèle à faire uniquement => cool trouver lambda optimal
# => prblm on dépend de la valeur diagonale A _lambdai,i => on remplace ce A par la moyenne de ces A => résout des problèmes = cross validation généralisé 
# 
# 
# 
#      regréssion quantile 
# ci dessus modèle linéaire pr estimer param de loi exp  (estimation modèles parametriques) avec penalisation ou pas 
# on veut gagner en souplesse, au lieu de modeliser param loi puis calcul quantile on calcul direct quantile 
# histogramme = "estimation non parametrique de densité "
# on va estimer le quantiles de cette manière un peu 
# 
# => pinball loss l_alpha(y-q)=alpha|y-q|+ +(1-alpha)|y-q|-  (parties positives / négative)
# valeur ebsolue = médiane conditionnelle 
# (modèle linéaire très simple Y=q ici mais même chose avec des beta etc)
# pondération plus ou moins forte de perte négative ou positive = estimation de différents quantiles
# 


# mape<-function(y,ychap)  # moyenne de lerreur absolue relative => en % d'erreur, soumise aux caractéristique des données (y doit pas etre proche de 0) pas adapté pr prod° solaire car tombe à 0 la nuit
  




















