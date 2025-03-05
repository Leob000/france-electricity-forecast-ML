rm(list=objects())
###############packages
library(mgcv)
library(yarrr)
library(magrittr)
library(forecast)
library(tidyverse)
source('R/score.R')

Data0 <- read_delim("Data/train.csv", delim=",")
Data1<- read_delim("Data/test.csv", delim=",")

range(Data0$Date)

Data0$Time <- as.numeric(Data0$Date)
Data1$Time <- as.numeric(Data1$Date)


sel_a <- which(Data0$Year<=2021)
sel_b <- which(Data0$Year>2021)


plot(Data0$Temp, Data0$Load)

par(mfrow=c(1,1))
g0 <- gam(Load~s(Temp, k=3, bs="cr"), data=Data0)
# premier modèle g0 conso f° temp ddl max=3 
# choix de k a l'oiel => combien de morceaux linéaires pour ajuster notre f°, 1ddl par noeud et 1 par bout 
# on voit que n ddl suffirait on prend un peu au dessus et on le laisse regulariser 
plot(g0, residuals=T)
# on voit f° cool mais un peu haute dans le nuage = biais (pas assz local dans certains endroits)
summary(g0)
# edf = 2 proche degré max k => indicateur k trop petit, estimateur trop contraint 
plot(Data0$Temp, g0$residuals, pch=16)
# residu f° temp => il reste un pattern = modèle explique mal 
# soit on le voit a l'oil soit on les regresse encore  :
g_prov <- gam(g0$residuals~ s(Data0$Temp, k=5, bs="cr"))
summary(g_prov)
# effet temperature encore significatif dans nuage de point residu = on a raté des choses dans le premier modèle 
# => on doit revenir sur precedent modèle et augmenter k 
# k=10 edf=6.8 => on est bien on est pas allé taper la contrainte
# 30% en dessous règle qui marche bien 
#residu nouveau modèle => on voit tjrs 2 strates liés aux jours de semaines
# s(Temp,k=10,bs='cr',by=as.factor(WeekDays))+as.factor(WeekDays) 
#(si by=variablenumeric)= 1 seul effet => erreur il a pas pris as factor a rectifier 
# pas oublier d'ajouter la variable categorique linéairement après 
# cool effet different selon les jours de la semaine 

g1 <- gam(Load~s(Temp, k=10, bs="cr"), data=Data0)
summary(g1)
plot(Data0$Temp, g0$residuals, pch=16, col='grey')
points(Data0$Temp, g1$residuals, pch=16)

(g0$gcv.ubre-g1$gcv.ubre)/g0$gcv.ubre
sqrt(g0$gcv.ubre)
sqrt(g1$gcv.ubre)
# on a regardé le GCV => on gagne 200megawatt en rmse     

 # g1.forecast  <- predict(g1, newdata=Data1)
 # rmse.old(Data1$Load-g1.forecast)
plot(g1,residuals=T)
# on fitte mieux le nuage en temp haute 



# validation croisé temporelle par bloc consecutifs 
Nblock<-10
borne_block<-seq(1, nrow(Data0), length=Nblock+1)%>%floor
block_list<-list()
l<-length(borne_block)
for(i in c(2:(l-1)))
{
  block_list[[i-1]] <- c(borne_block[i-1]:(borne_block[i]-1))
}
block_list[[l-1]]<-c(borne_block[l-1]:(borne_block[l]))


############################################################################
##############GAM model considering all predictors
############################################################################

# gam(target~x1 +s(x2,k=15,bs="cr")+s(x3,x4,k=50),family=poisson(link=log),data=df)
# k= ddl (dim base - contrainte)   cr=cubic regression splines
# s x3 x4 = intéraction entre deux variables ddl bcp plus grd car effet bivarié 
# f° bam big additive model = si gam trop lent (données assez grosses)
# autre type de splines dans bs = détail 
# intéraction soit avec f° s = un seul parametre à pénaliser (régulière de même façon selon axe x3 ou x4)
# pas forcement tt le temps vrai (t° nébulosité rien a voir) => y~te(x3,x4,k=(5,10)
# te = on a deux paramètre de regularisation, préciser le ddl pr chaque bases
# possibilité de préciser a la main les noeuds knots =c(1,2,3,4,5,6,7,8,9)
# knots=list(x=knots) pour affiner 
# option by = s(x1,by=x2)  x1 quantitatif x2 qualitatif 
# si on veut effet de niveau = rajouter x2 après +x2 (sinon trucs bizarres)
# summary(modele_gam) : edf=estimated degree of freedom : 
# k=10=dimension espace sur lequel j'ai projeté edf=7.2
# logique que <10 car k ddl max, à quel point edf en dessous dépend du lambda choisit 
# si proche de 10 pas besoin de pénaliser pour ajuster les données 
# si inférieur a 10 on a plus ragularisé/pénalisé   lambda>0
# test de fisher significativité
# R2 comme d'hab    sqrt(GCV)=RMSE
# plot(model,residuals=T,rug=T,se=F,pch=20)
# residuals = t = rajoutes vraies données corrigés autres effets additifs (y-x2-x3) pour x1
# se=T = donne intervalle de confiance 
# predict(g,newdata=dfval,type='terms') => matrice dans chaque colonne prevision de chaque terme additif 


# modeliser une tendance dans nos données => projeter nos données sur une spline avec très petit nbr de ddl
gam1<-gam(Load~s(as.numeric(Date), k=3,  bs='cr'),  data=Data0)
summary(gam1)
plot(gam1)
# croissance puis décroissance dans nos données en tendance générale 
# 20ddl => plus tendance mais periodicité dans nos données 
# a nous de choisir ce qu'on veut estimer 
# k=10 cool il voit le covid dans la tendance, attention au bord il fait de la merde descend trop vite extrapolation an suivant pourrie 
# ce prblm = trop de noeuds = regularise pas assez 

plot(Data0$toy, Data0$Load, pch=16)


# cycle annuel, tendance, température = 3 effet
gam1<-gam(Load~s(as.numeric(Date), k=3,  bs='cr')+s(toy,k=30, bs='cc')+s(Temp,k=10, bs='cr'), data=Data0)
summary(gam1)
# toy bcp plus complexe => k=30 => trouve edf=26 = ok coherent 
plot(gam1) 
# tendance plutot a la baisse
#cycle annuel => voit bien rupturre aout et vacance de noel, voit conso plus élevé hiver qu'été
# temperature, effet clim plus net que modéle qu'avec température (cool !)

blockRMSE<-function(equation, block)
{
  g<- gam(as.formula(equation), data=Data0[-block,])
  forecast<-predict(g, newdata=Data0[block,])
  return(Data0[block,]$Load-forecast)
} 


#####model 1
equation <- Load~s(as.numeric(Date),k=3, bs='cr')+s(toy,k=30, bs='cr')+s(Temp,k=10, bs='cr')
Block_residuals<-lapply(block_list, blockRMSE, equation=equation)%>%unlist
rmseBloc1<-rmse.old(Block_residuals)
rmseBloc1 # 3724 sur rmse, sqrt(gcv)=3687  
# gcv bon estimateur données prevision mais légérement optimiste
# bloc légérement dans futur (test) on aurait uen erreur un peu plus rgande 

gam1<-gam(equation, data=Data0[sel_a,])
gam1$gcv.ubre%>%sqrt

boxplot(Block_residuals)
plot(Block_residuals, type='l')
# effet covid pas corrigé = predit mal
# gros pic et cycles => on a du boulot encore
hist(Block_residuals)
# mélange gaussien avec deux regime a ameliorer 

boxplot(Block_residuals~Data0$WeekDays)
plot(Data0$Temp, Block_residuals, pch=16)
plot(Data0$toy, Block_residuals, pch=16)
plot(Data0$Load.1, Block_residuals, pch=16)
rmse1 <- rmse.old(Block_residuals)
rmse1
# type de jour pas pris en compte = important a corriger 

gam1.forecast<-predict(gam1,  newdata= Data0[sel_b,])
rmse1.forecast <- rmse.old(Data0$Load[sel_b]-gam1.forecast)
# rmse sur bloc laissé de cote 3264<erreur par bloc (qui inclut covid)

#####model 2
# on rajoute days pour modeliser type de jour
equation <- Load~s(as.numeric(Date),k=3, bs='cr')+s(toy,k=30, bs='cc')+s(Temp,k=10, bs='cr')+as.factor(WeekDays)
Block_residuals<-lapply(block_list, blockRMSE, equation=equation)%>%unlist
hist(Block_residuals, breaks=20)
rmse2 <- rmse.old(Block_residuals)
rmse2
# on perd les deux regimes vu plus haut 
# hist plus proche de gausien mais reste avec assymétrie valeur du jour ferié etc (atypique)
gam2<-gam(equation, data=Data0[sel_a,])
summary(gam2)
# +7 parma = +10pts R2 adj = significatif 

boxplot(Block_residuals~Data0$WeekDays)
# on voit que ça a  bien fonctionné 

plot(Data0$Time, Block_residuals, type='l')

plot(Data0$Temp, Block_residuals, pch=16)
# erreur f° temp => plutot centré = cool,  on a tjrs dispersion plus grd pr temp froide que chaud => chiant 
plot(Data0$Load.1, Block_residuals, pch=16)
# pattern retrouvé => effet a ajouter
# conso élevé dépendance forte => a ajouter
plot(Data0$Load.1, Data0$Load, pch=16)


gam2.forecast<-predict(gam2,  newdata= Data0[sel_b,])
rmse2.forecast <- rmse.old(Data0[sel_b,]$Load-gam2.forecast)


#####model 3
# on rajoute conso de la veille :
# bien sur affiner plus jouer sur les k, bricoler quand nous on bosse dessus
equation <- Load~s(as.numeric(Date),k=3, bs='cr')+s(toy,k=30, bs='cc')+s(Temp,k=10, bs='cr')+
  s(Load.1, bs='cr')+as.factor(WeekDays)
Block_residuals<-lapply(block_list, blockRMSE, equation=equation)%>%unlist
rmse3 <- rmse.old(Block_residuals)
rmse3 # on gagne encore de manière significative 
gam3<-gam(equation, data=Data0[sel_a,])
summary(gam3)
# quelque % de r2 adj on a bcp moins gagné mais tjrs un peu
plot(gam3,select=4)
# plot juste un effet, ici la conso de la veille => quasi linéaire d'aspect, 
#mais en ddl 8.7=> pas petit => lié aux oscillations au début 

hist(Block_residuals, breaks=20)
# resséré )=> bcp amélioré pr le jour classqiue  mais tjrs queue des jours atypiques mal représenté 
plot(Block_residuals,  type='l')
# résidus très différent variance plus faible, on voit bcp plus 
#   nettement les pics d'erreus, on voit rjs la ruprure covid et le second confinement
# tt les ans on voit un cycle => boulot pas parfait (vacances ?)
boxplot(Block_residuals~Data0$WeekDays)
Acf(Block_residuals)
# autocorrelogramme des residus par blocs (correlation entre epst et eps_t-1 puis epst-1 epst-2)
# malgré la dépendance a conso de la veille on atjrs une correlation entre les residus, 
#plus pic 7eme jour cycle hebdomadaire erreur lundi prochain corrélé à erreur lundi 25%
plot(Data0$Load.7, Block_residuals, pch=16)
# residus f° conso -7j = petite droite vite fais pas très pentue
cor(Data0$Load.7,Block_residuals)
# correlation 0.08

gam3.forecast<-predict(gam3,  newdata= Data0[sel_b,])
rmse3.forecast <- rmse.old(Data0[sel_b,]$Load-gam3.forecast)


#####model 4
# on rajoute la conso -7j
equation <- Load~s(as.numeric(Date),k=3,  bs='cr')+s(toy,k=30, bs='cc')+s(Temp,k=10, bs='cr') + s(Load.1, bs='cr')+ s(Load.7, bs='cr') + as.factor(WeekDays)
Block_residuals<-lapply(block_list, blockRMSE, equation=equation)%>%unlist
rmse4 <- rmse.old(Block_residuals)
rmse4 #  1349->1318 avec un effet en plus pas inninteressant
gam4<-gam(equation, data=Data0[sel_a,])
summary(gam4)
# plus fisher significatif => pas mauvaise idee 
plot(gam4)

plot(Data0$Date, Block_residuals, type='l')
# pas trop de difference 
boxplot(Block_residuals~Data0$BH)
# jours fériés => effet significatif 
boxplot(Block_residuals~Data0$Christmas_break)
# moins significatif (variance plus grande en hiver pas sur que y'ait un effet +deja modélisé dans partie saisonnière )
boxplot(Block_residuals~Data0$Summer_break)
# pas de relatione vidente aussi
gam4.forecast<-predict(gam4,  newdata= Data0[sel_b,])
rmse4.forecast <- rmse.old(Data0[sel_b,]$Load-gam4.forecast)


#####model 5
# on ajoute les jours fériés (déjà 0 1 pas besoin de la convertir en facteur)
equation <- Load~s(as.numeric(Date),k=3, bs='cr') + s(toy,k=30, bs='cc') + s(Temp,k=10, bs='cr') + s(Load.1, bs='cr')+ s(Load.7, bs='cr') + as.factor(WeekDays) +BH
Block_residuals<-lapply(block_list, blockRMSE, equation=equation)%>%unlist
rmse5 <- rmse.old(Block_residuals)
#rmse5 <- rmse.old.old(Block_residuals)
rmse5 # 1025 => bcp amélioré l'erreur quadratique (points ectrêmes ont un gros impact)
gam5<-gam(equation, data=Data0[sel_a,])
summary(gam5)
# r2 adj 0.99 très fort cool (proche de 1 pas super modèle attention juste dit qu'on a reduit le bruit ou qqchose comme ça)
plot(Data0$Date, Block_residuals, type='l')
# réduit bcp les extrêmes mais il reste des jours où on a surestimé la conso vers le bas et sousestimé vers lehaut

plot(Data0$Temp_s95, Block_residuals, pch=16)
# temp lissé on  voit plus trop si ça apporte qq chose 
test <- gam(Block_residuals~s(Data0$Temp_s95))
# gam avec lissé => est ce qu'il se passe un truc pr voir si ça apporte qq chose 
summary(test)
# edf=1 plus test fisher aide pas 
# edf=1 spline=linéaire ou plutot lambda tend vers l'infini => très négatif 
# GAM peut pas éliminer les variable => edf=1 enlever la variable elle sert a rien

plot(Data0$Temp_s99, Block_residuals, pch=16)
test <- gam(Block_residuals~s(Data0$Temp_s99))
summary(test)
# ici il trouve qq chose edf=2.57 test de fisher pas significatif 
# on a un doute => regarder approfondir 

sqrt(gam5$gcv.ubre)

gam5.forecast<-predict(gam5,  newdata= Data0[sel_b,])
rmse5.forecast <- rmse.old(Data0[sel_b,]$Load-gam5.forecast)

#####model 6
# + temp lissé 99
equation <- Load~s(as.numeric(Date),k=3, bs='cr') + s(toy,k=30, bs='cc') + s(Temp,k=10, bs='cr') + s(Load.1, bs='cr')+ s(Load.7, bs='cr') +
  s(Temp_s99,k=10, bs='cr') + as.factor(WeekDays) +BH
Block_residuals<-lapply(block_list, blockRMSE, equation=equation)%>%unlist
rmse6 <- rmse.old(Block_residuals)
rmse6 # 1031>1025 modèle 5 on a dégradé
# gcv = on a amélioré de 3megawatt (sur mille pas significatif)
gam6<-gam(equation, data=Data0[sel_a,])
sqrt(gam6$gcv.ubre)
summary(gam6)

plot(Data0$Date, Block_residuals, pch=16)
Acf(Block_residuals)
# cette variable semble a voir dégradé les choses 
plot(Data0$Temp, Block_residuals, pch=16)

plot(Data0$Temp_s95_max, Block_residuals, pch=16)

test <- gam(Block_residuals~s(Data0$Temp_s95_max))    # pas significatif
summary(test)
test <- gam(Block_residuals~s(Data0$Temp_s99_max)) # pas significatif 
summary(test)

test <- gam(Block_residuals~te(Data0$Temp_s95_max, Data0$Temp_s99_max))
# interaction => signifjcatif => effet croisés des vraibles => traduit vague de froid ou phénomène mais cool

summary(test)

gam6.forecast<-predict(gam6,  newdata= Data0[sel_b,])
rmse6.forecast <- rmse.old(Data0[sel_b,]$Load-gam6.forecast)
# 497 erreur sur le test public 


##### GRAPHIQUE CUSTOM STYLE PR RAPPORT #########################################
par(mfrow=c(2,2))
plot(gam6, select=c(2), scale=0, residuals=T, shade=TRUE, shade.col="pink", col='red', lwd=2, rug=F, pch=20, cex=0.01)
# on voit bien le cycle annuel, les vacances effet hiver 
plot(gam6, select=3, scale=0, residuals=T, shade=TRUE, shade.col="lightblue", col='blue', lwd=2, rug=F, pch=20, cex=0.01)
# temperature 
plot(gam6, select=4, scale=0, residuals=T, shade=TRUE, shade.col="seagreen1", col='darkolivegreen2', lwd=2, rug=F, pch=20, cex=0.01)
# effet de la veille
plot(gam6, select=5, scale=0, residuals=T, shade=TRUE, shade.col="thistle3", col='violetred1', lwd=2, rug=F, pch=20, cex=0.01)
# conso haute basse on voit la diff
#####################################################################################


# cours suivant correction d'erreur avec residus : 
####################################################################################################
######################### residual correction
####################################################################################################


### partie time series :
# modèle autorégressif   yt=a1yt-1+epst  epst=teta1.epst-1+teta2.epst-2+ut
# modèle ARIMA             autorgressif     moyenne mobile du bruit 
#  I= integrated = on modélise les variations de yt 
# on veut coupler gam a arima pour voir s'il reste de l'erreur dans les résidus

# onprend les résidus par corss val par bloc (important pas prendre le train)
Block_residuals.ts <- ts(Block_residuals, frequency=7)

# technique de selection de modèle automatique selon critère précisé ici AIC=>prend le AIC le plus eptit possible
fit.arima.res <- auto.arima(Block_residuals.ts,max.p=3,max.q=4, max.P=2, max.Q=2, trace=T,ic="aic", method="CSS")
# maxp maxq = jusqu'a quel ordre je regarde mes dépendances q autoregressif p arima
# ci dessus modèle sarima saisonnier maxP maxQ => modèle autoregressif modulo 7 o
#Best model: ARIMA(3,0,4)(1,0,0)[7] with zero mean  
#  ARIMA(2,0,1)(2,0,0)[7] with zero mean  modèle autoregressif ordre 2 (2 dernières semaines)
# partie non saisonnière lire erreur autoregresive sur dernier jour (2 modèles autoregressifs emboités)
#saveRDS(fit.arima.res, "Results/tif.arima.res.RDS")

# prévision avec modèle autoregresif => soit boucle à la main soit technique avec refit param fixé = génère prediction à horizon 1
ts_res_forecast <- ts(c(Block_residuals.ts, Data0[sel_b,]$Load-gam6.forecast),  frequency= 7)
# residus apprentissage et partie que je veux prevoir (selb)
refit <- Arima(ts_res_forecast, model=fit.arima.res) 
# réapprend sur tt ça en fixant les paramètres pr pas surapprendre sur le test
prevARIMA.res <- tail(refit$fitted, nrow(Data0[sel_b,]))

rmse6.arima <- rmse.old(fit.arima.res$residuals)
# 1100gam->979arima
# peu d'effort grosse amélioration => cool
# souvent le cas que arima nous fait gagner (que le cas pr prévision jour pour lendemain)
# dépend fortement autocorrélation des données, bcp moins le cas pr d'une année a l'autre par ex

# prevision finale 
gam6.arima.forecast <- gam6.forecast + prevARIMA.res



rmse.old(Data0[sel_b,]$Load-gam6.forecast)
rmse.old(Data0[sel_b,]$Load-gam6.arima.forecast)
mape(Data0[sel_b,]$Load, gam6.arima.forecast)

rmse6.arima.forecast <- rmse.old(Data0[sel_b,]$Load-gam6.arima.forecast)

################################################################################
##########synthèse
################################################################################

# cross val par bloc
rmseCV <- c(rmse1, rmse2, rmse3, rmse4, rmse5, rmse6, rmse6.arima)
# prevision sur echantillon b
rmse.old.forecast<- c(rmse1.forecast, rmse2.forecast, rmse3.forecast, rmse4.forecast, rmse5.forecast, rmse6.forecast, 
                   rmse6.arima.forecast)
#generalized cross val
rgcv <- c(gam1$gcv.ubre, gam2$gcv.ubre, gam3$gcv.ubre, gam4$gcv.ubre, gam5$gcv.ubre, gam6$gcv.ubre)%>%sqrt

par(mfrow=c(1,1))
plot(rmseCV, type='b', pch=20, ylim=range(rmseCV, rmse.old.forecast))
lines(rmse.old.forecast, type='b', pch=20, col='blue')
lines(rgcv, col='red', type='b', pch=20)
points(7, rmse6.arima.forecast)
points(7, rmse6.arima)
legend("topright", col=c("red","black","blue"), c("gcv","blockCV","test"), pch=20, ncol=1, bty='n', lty=1)

# on voit que mêmes dynamiques au début pr minimiser le mgcv
# à un moment l'erreur de test décroche alors que le mgcv continue a descendre 
# intéressant de se poser pour analyser nos choix de cette manière (sur public test/valid etc)
# plafonnage sur les gam, arima refait gagner pas mal cool




################################################################################
##########NetDemand
################################################################################

##### modélisation net demand => ajouter terme expliquant solaire eolien
# ajouter terme expliquant ces productions (vent nebulosité)

equation <- Net_demand ~ s(as.numeric(Date),k=3, bs='cr') + s(toy,k=30, bs='cc') + s(Temp,k=10, bs='cr') + s(Temp_s99,k=10, bs='cr') + WeekDays +BH 
gam_net0 <- gam(equation,  data=Data0[sel_a,])
summary(gam_net0)         
sqrt(gam_net0$gcv.ubre)
gam_net1.forecast<-predict(gam_net0,  newdata= Data0[sel_b,])
rmse_net1.forecast <- rmse(Data0[sel_b,]$Net_demand, gam_net0.forecast)
res <- Data0$Net_demand[sel_b] - gam_net1.forecast
quant <- qnorm(0.95, mean= mean(res), sd= sd(res))
pinball_loss(y=Data0$Net_demand[sel_b], gam_net1.forecast+quant, quant=0.95, output.vect=FALSE)

#843
# même modèle que la load 

#####

equation <- Net_demand ~ s(as.numeric(Date),k=3, bs='cr') + s(toy,k=30, bs='cc') + s(Temp,k=10, bs='cr') + s(Load.1, bs='cr')+ s(Load.7, bs='cr') +
  s(Temp_s99,k=10, bs='cr') + WeekDays +BH +
  s(Wind) + s(Nebulosity)
gam_net1 <- gam(equation,  data=Data0[sel_a,])
summary(gam_net1)         
sqrt(gam_net1$gcv.ubre)
gam_net1.forecast<-predict(gam_net1,  newdata= Data0[sel_b,])
mape(Data0[sel_b,]$Net_demand, gam_net1.forecast)
rmse_net1.forecast <- rmse(Data0[sel_b,]$Net_demand, gam_net1.forecast)
res <- Data0$Net_demand[sel_b] - gam_net1.forecast
quant <- qnorm(0.95, mean= mean(res), sd= sd(res))
pinball_loss(y=Data0$Net_demand[sel_b], gam_net1.forecast+quant, quant=0.95, output.vect=FALSE)

# 460
# avec vent et nébulosité 
# approche sur conso march aussi bien avec net deamnd avec deux variables en plus

#####
equation <- Net_demand ~ s(as.numeric(Date),k=3, bs='cr') + s(toy,k=30, bs='cc') + s(Temp,k=10, bs='cr') + s(Load.1, bs='cr')+ s(Load.7, bs='cr') +
  s(Temp_s99,k=10, bs='cr') + WeekDays +BH +
  s(Wind) + te(as.numeric(Date), Nebulosity, k=c(4,10))
gam_net2 <- gam(equation,  data=Data0[sel_a,])
summary(gam_net2)         
sqrt(gam_net2$gcv.ubre)
gam_net2.forecast<-predict(gam_net2,  newdata= Data0[sel_b,])
rmse_net2.forecast <- rmse(Data0[sel_b,]$Net_demand, gam_net2.forecast)
res <- Data0$Net_demand[sel_b] - gam_net2.forecast
quant <- qnorm(0.95, mean= mean(res), sd= sd(res))
pinball_loss(y=Data0$Net_demand[sel_b], gam_net2.forecast+quant, quant=0.95, output.vect=FALSE)

# rmse meilleur pinball pas d'amélioration à nous de bricoler 
# pallier au problème de basse qualité nébulosité dans le passé k=c(4,10)  
# peu de ddl imposé sur leffet tendance pr eviter qu'il el surapprene 

#####
equation <- Net_demand ~ s(as.numeric(Date),k=3, bs='cr') + s(toy,k=30, bs='cc') + s(Temp,k=10, bs='cr') + s(Load.1, bs='cr')+ s(Load.7, bs='cr')  +
  s(Temp_s99,k=10, bs='cr') + WeekDays +BH +
  s(Wind) + te(as.numeric(Date), Nebulosity, k=c(4,10)) +
  s(Net_demand.1, bs='cr') +  s(Net_demand.7, bs='cr')

gam_net3 <- gam(equation,  data=Data0[sel_a,])
sqrt(gam_net3$gcv.ubre)
gam_net3.forecast<-predict(gam_net3,  newdata= Data0[sel_b,])
rmse_net3.forecast <- rmse(Data0[sel_b,]$Net_demand, gam_net3.forecast)
res <- Data0$Net_demand[sel_b] - gam_net3.forecast
quant <- qnorm(0.95, mean= mean(res), sd= sd(res))
pinball_loss(y=Data0$Net_demand[sel_b], gam_net3.forecast+quant, quant=0.95, output.vect=FALSE)

# netdeamnd veille et semaine d'avant au lieu de que veille pr aspect autoregressif 
# améliore un peu pas tant que ça 

mean(Data0[sel_b,]$Net_demand<gam_net3.forecast+quant)



hist(gam_net3$residuals, breaks=50)
# résidus ressembles pas mal a une gaussienne quand même on est en bonne voie 

# résidus gaussiens blanc et tt => j'ai gagné !!!!
# oui mais ? => est ce qu'on a bien pris l'échantillon de test ? parce que facile de prendre res blanc sur train 
# on doit avoir un bon lmoyen d'evaluer notre modèle 
# aussi res dépend plus de rien par rapport aux variables utilisés, pourrai dépendre d'autres variables pas disponible

#################gamlss
