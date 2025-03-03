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

range(Data1$Date)



range(Data0$Date)

Data0$Time <- as.numeric(Data0$Date)
Data1$Time <- as.numeric(Data1$Date)


sel_a <- which(Data0$Year<=2021)
sel_b <- which(Data0$Year>2021)



Nblock<-10
borne_block<-seq(1, nrow(Data0), length=Nblock+1)%>%floor
block_list<-list()
l<-length(borne_block)
for(i in c(2:(l-1)))
{
  block_list[[i-1]] <- c(borne_block[i-1]:(borne_block[i]-1))
}
block_list[[l-1]]<-c(borne_block[l-1]:(borne_block[l]))



blockRMSE<-function(equation, block)
{
  g<- gam(as.formula(equation), data=Data0[-block,])
  forecast<-predict(g, newdata=Data0[block,])
  return(forecast)
}




#####model 6 de base d'avant
equation <- Load~s(as.numeric(Date),k=3, bs='cr') + s(toy,k=30, bs='cc') + s(Temp,k=10, bs='cr') + s(Load.1, bs='cr')+ s(Load.7, bs='cr') +
  s(Temp_s99,k=10, bs='cr') + as.factor(WeekDays) +BH
Block_residuals<-lapply(block_list, blockRMSE, equation=equation)%>%unlist
rmse6 <- rmse(Data0$Load, Block_residuals)
rmse6
gam6<-gam(equation, data=Data0[sel_a,])
sqrt(gam6$gcv.ubre)
summary(gam6)

gam6.forecast<-predict(gam6,  newdata= Data0[sel_b,])
gam6$gcv.ubre%>%sqrt
rmse(Data0[sel_b,]$Load, gam6.forecast)


#####model 7 variante ti

# interaction temperature brute et lissé (inertie batiments)
equation <- Load~s(as.numeric(Date),k=3, bs='cr') + s(toy,k=30, bs='cc') + s(Temp,k=10, bs='cr') + s(Load.1, bs='cr')+ s(Load.7, bs='cr') +
  s(Temp_s99,k=10, bs='cr') + as.factor(WeekDays) +BH  + te(Temp_s99, Temp, bs=c('cr','cr'), k=c(10,10))

Block_forecast<-lapply(block_list, blockRMSE, equation=equation)%>%unlist
Block_residuals <- Data0$Load-Block_forecast
rmse_te <- rmse(Data0$Load, Block_forecast)
rmse_te
gam7<-gam(equation, data=Data0[sel_a,])
summary(gam7)
# terme interaction significatif terme lissé 99 plus significatif => interaction contient que le 99 ?
# prevision 20megawatt mieux mais en ajoutant 16 ddl pas fou
gam7.forecast<-predict(gam7,  newdata= Data0[sel_b,])
gam7$gcv.ubre%>%sqrt
rmse(Data0[sel_b,]$Load, gam7.forecast)

# on fait bien le ti avec ti temp ti temp lissé ti interaction
equation <- Load~s(as.numeric(Date),k=3, bs='cr') + s(toy,k=30, bs='cc') + ti(Temp,k=10, bs='cr') + s(Load.1, bs='cr')+ s(Load.7, bs='cr') +
  ti(Temp_s99,k=10, bs='cr') + as.factor(WeekDays) +BH  + ti(Temp_s99, Temp, bs=c('cr','cr'), k=c(10,10))

Block_forecast<-lapply(block_list, blockRMSE, equation=equation)%>%unlist
rmse_ti <- rmse(Data0$Load, Block_forecast)
rmse_ti
gam7_ti<-gam(equation, data=Data0[sel_a,])
summary(gam7_ti)
gam7_ti.forecast<-predict(gam7_ti,  newdata= Data0[sel_b,])
gam7_ti$gcv.ubre%>%sqrt
rmse(Data0[sel_b,]$Load, gam7_ti.forecast)
# effet marginal temp pas effet templissé effet interaction confirme ce quo'n a vu ci dessus c'est l'interaction





##############################################################################
#############Anova selection
##############################################################################


# equation <- Load~s(as.numeric(Date),k=3, bs='cr') + s(toy,k=30, bs='cc') + s(Temp,k=10, bs='cr') + s(Load.1, bs='cr')+ s(Load.7, bs='cr') +
#   s(Temp_s99,k=10, bs='cr') + WeekDays +BH  + te(Temp_s95_max, Temp_s99_max) + Summer_break  + Christmas_break + te(Temp_s95_min, Temp_s99_min)
# 
# equation_by <- Load~s(as.numeric(Date),k=3, bs='cr') + s(toy,k=30, bs='cc') + s(Temp,k=10, bs='cr') + s(Load.1, bs='cr', by= as.factor(WeekDays))+ s(Load.7, bs='cr') +
#   s(Temp_s99,k=10, bs='cr') + WeekDays +BH  + te(Temp_s95_max, Temp_s99_max) + Summer_break  + Christmas_break + te(Temp_s95_min, Temp_s99_min)

# test RV entre deux modèles emboités 
# difference effet conso veille ou effet cponso veille by type de jour=> croiser effet non lineaire avec variable qualitative
equation <- Load~s(as.numeric(Date),k=3, bs='cr') + s(toy,k=30, bs='cc') + s(Temp,k=10, bs='cr') + s(Load.1, bs='cr')+ s(Load.7, bs='cr') +
  s(Temp_s99,k=10, bs='cr') + as.factor(WeekDays) +BH

equation_by <- Load~s(as.numeric(Date),k=3, bs='cr') + s(toy,k=30, bs='cc') + s(Temp,k=10, bs='cr') + s(Load.1, bs='cr',  by= as.factor(WeekDays))+ s(Load.7, bs='cr') +
  s(Temp_s99,k=10, bs='cr') + as.factor(WeekDays) +BH


# modèle avec le by chaque effet a l'ir significatif 
gam6<-gam(equation, data=Data0)
gam6_by<-gam(equation_by, data=Data0)
gam6$gcv.ubre%>%sqrt
gam6_by$gcv.ubre%>%sqrt

summary(gam6_by)
anova(gam6, gam6_by, test = "Chisq") ####p value <0.05 interaction is significant
# on rejette hypothèse nullité effet autoregressif pa typoe de jour sont significatifs 

gam6_by<-gam(equation_by, data=Data0[sel_a,])
gam6_by.forecast<-predict(gam6_by,  newdata= Data0[sel_b,])
rmse(Data0[sel_b,]$Load, gam6_by.forecast)

rmse(Data0[sel_b,]$Load, gam6.forecast)

# 1100 modèle sans type de jour 1060 avec


###exemple2
equation <- Load~ ti(Load.1, bs='cr')+ ti(Load.7, bs='cr')
equation2 <- Load~ ti(Load.1, bs='cr')+ ti(Load.7, bs='cr') + ti(Load.1, Load.7)
fit1<-gam(equation, data=Data0)
fit2<-gam(equation2, data=Data0)

anova(fit1, fit2, test = "Chisq") ####p value <0.05 interaction is significant

# testé par prof au tableau modèle autoregressif pur =>2785 rmse 


########################################################################################################
########shrinkage approach
########################################################################################################
set.seed(100)
Data0$Var1 <- rnorm(nrow(Data0), mean=0, sd=sd(Data0$Load))
# ecrt type conso electrique bruite pas mal, on peut le bruiter plus en augmentant la avriabel

# analyser il y a des manières d'améliorer le modèles la 
equation <- Load~s(as.numeric(Date),k=3, bs='cr') + s(toy,k=30, bs='cc') + s(Temp,k=10, bs='cr') + s(Load.1, bs='cr', by= as.factor(WeekDays))+ s(Load.7, bs='cr') +
  s(Temp_s99,k=10, bs='cr') + as.factor(WeekDays) +BH  + BH_before + BH_after+ s(Temp_s95) +ti(Nebulosity_weighted) + ti(Wind_weighted) +ti(Wind_weighted, Temp) +s(Var1)
# + terme nebulosité weighted / interaction temperature vent pour temperature ressentie
# + svar1= ajout de bruit dans les données comment le modèle réagit => robustesse 
# si modèle trouve bruit significatif = problème
# ici rejet test fischer mais il met pas une penalité infinie dessus 
#1044
gam8<-gam(equation, data=Data0[sel_a,])
summary(gam8)
gam8$gcv.ubre%>%sqrt
gam8.forecast<-predict(gam8,  newdata= Data0[sel_b,])
rmse(Data0[sel_b,]$Load, gam8.forecast)

# on met shrinkage (une seule penalité mais une proportionnelle a l'autre) sur notre bruit
# => 2.2ddl ->0.6ddl ça a bien marché
#1045 meilleur en gcv equivalent en previsiuon
# shrinkage shrink la variable parasite  => ca marche !
equation_cs <- Load~s(as.numeric(Date),k=3, bs='cr') + s(toy,k=30, bs='cc') + s(Temp,k=10, bs='cr') + s(Load.1, bs='cr', by= as.factor(WeekDays))+ s(Load.7, bs='cr') +
  s(Temp_s99,k=10, bs='cr') + as.factor(WeekDays) +BH  + BH_before + BH_after+ s(Temp_s95) +ti(Nebulosity_weighted) + ti(Wind_weighted) +ti(Wind_weighted, Temp) + s(Var1, bs='cs')
# equation_cs <- Load~s(as.numeric(Date),k=3, bs='cs') + s(toy,k=30, bs='cr') + s(Temp,k=10, bs='cr') + s(Load.1, bs='cs',  by= as.factor(WeekDays))+ s(Load.7, bs='cr') +
#   s(Temp_s99,k=10, bs='cs') + as.factor(WeekDays) +BH  + BH_before + BH_after +s(Temp_s95, bs='cs') +ti(Nebulosity_weighted, bs='ts') + ti(Wind_weighted, bs='ts')+ti(Wind_weighted, Temp, bs='ts') + s(Var1, bs='cs')
gam8_cs<-gam(equation_cs, data=Data0[sel_a,])
summary(gam8_cs)
gam8_cs.forecast<-predict(gam8_cs,  newdata= Data0[sel_b,])
gam8_cs$gcv.ubre%>%sqrt
rmse(Data0[sel_b,]$Load, gam8_cs.forecast)



# premier modèle noirdeuième rouge = represente juste efeft de toy 
# idée voir comment effet shrinkage a fait bouger effet de variable : toy / vent => affecte pas c'est normal c'est cool
# pr variable shrinké on voit pas du tt superposé 
# noir = non shrinké     rouge=shrinké quasiment une droite (ddl0.6 coherent)
# il a pas réussi a virer totalement l'effet : on voit ça = virer la variable du modèle

# toy sasn spleen cubique, contrainte cyclique force modèle a avoir même valeur à gauche et a droite 
# peut etre bien ou pas (possible ruptures)

##toy test à faire en retirant cc de l'effet toy
terms <- predict(gam8, newdata=Data0, type='terms')
terms_cs <- predict(gam8_cs, newdata=Data0, type='terms')
sel.column <- which(colnames(terms)=="s(toy)")
o <- order(Data0$toy)
plot(Data0$toy[o] , terms[o, sel.column], type='l', ylim=range(terms[o, sel.column], terms_cs[o, sel.column]))
lines(Data0$toy[o] , terms_cs[o, sel.column], col='red')


##Wind_weighted
terms <- predict(gam8, newdata=Data0, type='terms')
terms_cs <- predict(gam8_cs, newdata=Data0, type='terms')
sel.column <- which(colnames(terms)=="ti(Wind_weighted)")
o <- order(Data0$Wind_weighted)
plot(Data0$Wind_weighted[o] , terms[o, sel.column], type='l', ylim=range(terms[o, sel.column], terms_cs[o, sel.column]))
lines(Data0$Wind_weighted[o] , terms_cs[o, sel.column], col='red')


##Var1
terms <- predict(gam8, newdata=Data0, type='terms')
terms_cs <- predict(gam8_cs, newdata=Data0, type='terms')
sel.column <- which(colnames(terms)=="s(Var1)")
o <- order(Data0$Var1)
plot(Data0$Var1[o] , terms[o, sel.column], type='l', ylim=range(terms[o, sel.column], terms_cs[o, sel.column]))
lines(Data0$Var1[o] , terms_cs[o, sel.column], col='red')



########################################################################################################
########double penalty shrinkage
########################################################################################################

equation <- Load~s(as.numeric(Date),k=3, bs='cr') + s(toy,k=30, bs='cc') + s(Temp,k=10, bs='cr') + s(Load.1, bs='cr', by= as.factor(WeekDays))+ s(Load.7, bs='cr') +
  s(Temp_s99,k=10, bs='cr') + as.factor(WeekDays) +BH  + BH_before + BH_after+ s(Temp_s95) +ti(Nebulosity_weighted) + ti(Wind_weighted) +ti(Wind_weighted, Temp) + s(Var1, bs='cs')

# select =True     bcp plus long 
# on voit que var1 bcp plus shrinké (0.39ddl)
gam9_select<-gam(equation, data=Data0[sel_a,], select=TRUE, gamma=1.5)
summary(gam9_select)
gam9_select$gcv.ubre%>%sqrt
gam9_select.forecast<-predict(gam9_select,  newdata= Data0[sel_b,])
rmse(Data0[sel_b,]$Load, gam9_select.forecast)

# gam8 sum(ddl tt f°)=106ddl
# avec shrinkage => 88ddl
# perf équivalente en prevision 


##Var1
terms <- predict(gam8, newdata=Data0, type='terms')
terms_cs <- predict(gam8_cs, newdata=Data0, type='terms')
terms_select <- predict(gam9_select, newdata=Data0, type='terms')
sel.column <- which(colnames(terms)=="s(Var1)")
o <- order(Data0$Var1)
plot(Data0$Var1[o] , terms[o, sel.column], type='l', ylim=range(terms[o, sel.column], terms_cs[o, sel.column]))
lines(Data0$Var1[o] , terms_cs[o, sel.column], col='red')
lines(Data0$Var1[o] , terms_select[o, sel.column], col='blue')

# f° shrinké : on voit que shrinkage pas top alors que double penalité a mis le coeff a 0





##################################################################
######online learning
##################################################################

# partie qui fait gagner en perf le plus dans le projet 

# pas le modèle le plus compliqué, pas le modèle le plus compliqué forcement le meilleur en online, à tester
equation <- Load~s(as.numeric(Date),k=3, bs='cr') + s(toy,k=30, bs='cc') + s(Temp,k=10, bs='cr') + 
  s(Load.1, bs='cr', by= as.factor(WeekDays))+ s(Load.7, bs='cr')  + as.factor(WeekDays) +BH 
# possible de faire une partioe du modèle cst et une partie online 

gam9<-gam(equation%>%as.formula, data=Data0[sel_a,])
gam9.forecast <- predict(gam9, newdata=Data0[sel_b,])

### partie kalman
# equation d'observation et equation d'état  
# tetat=tetat-1+mut mut suit gaussienne 0,Q  et tetat suit gaussienne 0 sigma**2
# static Q=0

X <- predict(gam9, newdata=Data0, type='terms')
###scaling columns
for (j in 1:ncol(X))
  X[,j] <- (X[,j]-mean(X[,j])) / sd(X[,j])
X <- cbind(X,1)
d <- ncol(X)
# matrice X avec f^x1 f^x2 etc comme colonnes 
# on standardise
# on rajoute la cst cbind

# a modifier après avec la net demand
y <- Data0$Load

# static KALMAN
ssm <- viking::statespace(X, y)
ssm #= statespace modèle, prevision moyenne et ecart type = obejt qui nous intéresse
ssm$kalman_params
# predivision du kalman = fin du predmean
gam9.kalman.static <- ssm$pred_mean%>%tail(length(sel_b))

# évolution des paramètres au court du telmps => cv des param
# param lag et tendance bouge bcp au debut puis se stabilise
plot(ssm)

rmse(y=Data0$Load[sel_b], ychap=gam9.forecast)
rmse(y=Data0$Load[sel_b], ychap=gam9.kalman.static)
#1080->1060 20megawatt   cool car avec param très basiques
# dernier paramètre teta =55000 normal = param cst = moyenne de conso

# pas meilleure façon de faire, sous exploitation du kalman 
# sig <- sd(Data0$Load[sel_a]-ssm$pred_mean[sel_a])
# quant <- qnorm(0.8, mean= gam9.kalman.static, sd= sig)
# pinball_loss(y=Data0$Load[sel_b], quant, quant=0.8, output.vect=FALSE)

# bonne manière
# quantile de gaussienne dont mloyenne et ecart typoe varie dans le temps 
sig <- ssm$pred_sd%>%tail(length(sel_b))*sd(Data0$Load[sel_a]-ssm$pred_mean[sel_a])
quant <- qnorm(0.8, mean= gam9.kalman.static, sd= sig)
pinball_loss(y=Data0$Load[sel_b], quant, quant=0.8, output.vect=FALSE)
# 420, pas forcement comparable avec netdemand mais cool
# gam on avait ... qqchose a regarder

#sd(Data0$Load[sel_b]-gam9.kalman.static)
#ssm$kf
#ssm$pred_sd*sd(Data0$Load[sel_a]-ssm$pred_mean%>%head(length(sel_a)))



# dynamic
# vrai kalman, question quelle meilleure valeur de Q =plusieurs manièe de faire
# using iterative grid search => Q = matrice diagonale et il regarde parmi ces matrices
ssm_dyn <- viking::select_Kalman_variances(ssm, X[sel_a, ], y[sel_a], q_list = 2^(-30:0), p1 = 1, 
                                           ncores = 6) # ncore=parallelisation
# assez long
#saveRDS(ssm_dyn, "Results/ssm_dyn.RDS")
ssm_dyn <- readRDS("Results/ssm_dyn.RDS")
ssm_dyn <- predict(ssm_dyn, X, y, type='model', compute_smooth = TRUE)
gam9.kalman.Dyn <- ssm_dyn$pred_mean%>%tail(length(sel_b))
# on récupère juste sur la fin des données 
# kalman => prevision => le faire sur train+test concaténé en prenannt que la tail
# temps de chauffedu kalman 
rmse(y=Data0$Load[sel_b], ychap=gam9.kalman.Dyn)
ssm_dyn$kalman_params$Q

quant <- qnorm(0.8, mean= gam9.kalman.Dyn, sd= ssm_dyn$pred_sd%>%tail(length(sel_b)))
pinball_loss(y=Data0$Load[sel_b], quant, quant=0.8, output.vect=FALSE)

# on voit que bcp param bouge pas du tt, temperature bouge assez lentement, cycle annuel bouge pas mal
# changement brutaux selon type de jour
# effet autoregresif bouge pas mal aussi
plot(ssm_dyn, pause=F, window_size = 14, date = Data0$Date, sel = sel_b)

# on peut utiliser kalman pour faire des prevision mpais aussi pour faire des smoothings


# EMV 
# donne matrice pas forcement diagonale 
# using expectation-maximization # 2-3min
ssm_em <- viking::select_Kalman_variances(ssm, X[sel_a,], y[sel_a], method = 'em', n_iter = 10^3,
                                          Q_init = diag(d), verbose = 10, mode_diag = T)
ssm_em <- predict(ssm_em, X, y, type='model', compute_smooth = TRUE)

#saveRDS(ssm_em, "Results/ssm_em.RDS")
ssm_em <-readRDS("Results/ssm_em.RDS")

gam9.kalman.Dyn.em <- ssm_em$pred_mean%>%tail(length(sel_b))
ssm_em$kalman_params$Q

# reparti bcp plus la variation entre les parametres 
plot(ssm_em, pause=F, window_size = 14, date = Data0$Date, sel = sel_b)
rmse(y=Data0$Load[sel_b], ychap=gam9.kalman.Dyn.em)
#1060
quant <- qnorm(0.8, mean= gam9.kalman.Dyn.em, sd= ssm_em$pred_sd%>%tail(length(sel_b)))
pinball_loss(y=Data0$Load[sel_b], quant, quant=0.8, output.vect=FALSE)
#273

par(mfrow=c(1,1))
plot(Data0$Date[sel_b], Data0$Load[sel_b], type='l')
lines(Data0$Date[sel_b], gam9.forecast, col='red')
lines(Data0$Date[sel_b], gam9.kalman.static, col='blue')
lines(Data0$Date[sel_b], gam9.kalman.Dyn, col='green')
lines(Data0$Date[sel_b], gam9.kalman.Dyn.em, col='purple')
# representation des differentes previsions , toutes assez proches 


r <- range(cumsum(Data0$Load[sel_b]-gam9.kalman.Dyn), cumsum(Data0$Load[sel_b]- gam9.forecast))
plot(Data0$Date[sel_b], cumsum(Data0$Load[sel_b]- gam9.forecast), type='l', ylim=r) # gam
lines(Data0$Date[sel_b], cumsum(Data0$Load[sel_b]-gam9.kalman.static), col='blue') # kalman static
lines(Data0$Date[sel_b], cumsum(Data0$Load[sel_b]-gam9.kalman.Dyn), col='green') # kalamn grid search
lines(Data0$Date[sel_b], cumsum(Data0$Load[sel_b]-gam9.kalman.Dyn.em), col='purple') # EMV
# representation des valeurs cumulés des erreurs => permet de voir les biais dans les modèles
# gam a tendance a sous estimé la conso, le static améliore un peu ça  y-y^ courbe croissante = sous estimation
# modèles dynamiques = surestiment 
# deux types de prediction prendre la moyenne et ce sera encore meilleur 



# prendre kalman et 10*Q pr voir ce qui se passe
# degreader l'erreur mais voir des comportenements interessants en terme de perte cumulé 
# chgmt Q gam param => familles modle => ensemble )=> bonne prediction 
ssm_dyn2 <- readRDS("Results/ssm_dyn.RDS")
ssm_dyn2$kalman_params$Q <- ssm_dyn2$kalman_params$Q*10
ssm_dyn2 <- predict(ssm_dyn2, X, y, type='model', compute_smooth = TRUE)
gam9.kalman.Dyn2 <- ssm_dyn2$pred_mean%>%tail(length(sel_b))
rmse(y=Data0$Load[sel_b], ychap=gam9.kalman.Dyn2)

quant <- qnorm(0.8, mean= gam9.kalman.Dyn2, sd= ssm_dyn2$pred_sd%>%tail(length(sel_b)))
pinball_loss(y=Data0$Load[sel_b], quant, quant=0.8, output.vect=FALSE)


par(mfrow=c(1,1))
r <- range(cumsum(Data0$Load[sel_b]-gam9.kalman.Dyn), cumsum(Data0$Load[sel_b]- gam9.forecast))
plot(Data0$Date[sel_b], cumsum(Data0$Load[sel_b]- gam9.forecast), type='l', ylim=r)
lines(Data0$Date[sel_b], cumsum(Data0$Load[sel_b]-gam9.kalman.static), col='blue')
lines(Data0$Date[sel_b], cumsum(Data0$Load[sel_b]-gam9.kalman.Dyn), col='green')
lines(Data0$Date[sel_b], cumsum(Data0$Load[sel_b]-gam9.kalman.Dyn.em), col='purple')
lines(Data0$Date[sel_b], cumsum(Data0$Load[sel_b]-gam9.kalman.Dyn2), col='orange')



############################################################################################################
##################qgam
############################################################################################################
library(qgam)
equation <- Load~s(as.numeric(Date),k=3, bs='cr') + s(toy,k=30, bs='cc') + s(Temp,k=10, bs='cr') + 
  s(Load.1, bs='cr', by= as.factor(WeekDays))+ s(Load.7, bs='cr')  + as.factor(WeekDays) +BH 
equation_var <- ~  s(Temp,k=10, bs='cr') + s(Load.1)
gqgam <- qgam(list(equation, equation_var), data=Data0[sel_a,], discrete=TRUE, qu=0.95)

gqgam.forecast <- predict(gqgam, newdata=Data0[sel_b,])
pinball_loss(y=Data0$Load[sel_b], gqgam.forecast, quant=0.95, output.vect=FALSE)

plot(Data0$Load[sel_b], type='l')
lines(gqgam.forecast, col='red')


gqgam95 <- qgam(list(equation, equation_var), data=Data0[sel_a,], discrete=TRUE, qu=0.95)
#gqgam05 <- qgam(list(equation, equation_var), data=Data0[sel_a,], discrete=TRUE, qu=0.05)
#X <- cbind(predict(gqgam05, newdata=Data0, type='terms'), predict(gqgam95, newdata=Data0, type='terms'))

y <- Data0$Load
X <-  predict(gqgam95, newdata=Data0, type='terms')
###scaling columns
for (j in 1:ncol(X))
  X[,j] <- (X[,j]-mean(X[,j])) / sd(X[,j])
X <- cbind(X,1)
d <- ncol(X)

# static 
ssm_q <- viking::statespace(X, y)
ssm_q
gqgam.kalman.static <- ssm_q$pred_mean%>%tail(length(sel_b))
rmse(y=Data0$Load[sel_b], ychap=gqgam.kalman.static)


# dynamic
# using iterative grid search
ssm_dyn_q <- viking::select_Kalman_variances(ssm_q, X[sel_a, ], y[sel_a], q_list = 2^(-30:0), p1 = 1, 
                                           ncores = 6)
saveRDS(ssm_dyn_q, "Results/ssm_dyn_q.RDS")

ssm_dyn_q <- readRDS("Results/ssm_dyn_q.RDS")
ssm_dyn_q <- predict(ssm_dyn_q, X, y, type='model', compute_smooth = TRUE)
gqgam.kalman.Dyn <- ssm_dyn_q$pred_mean%>%tail(length(sel_b))
rmse(y=Data0$Load[sel_b], ychap=gqgam.kalman.Dyn)



res <- Data0$Load[sel_a] - ssm_dyn_q$pred_mean%>%head(length(sel_a))
quant <- qnorm(0.95, mean= mean(res), sd= sd(res))
pinball_loss(y=Data0$Load[sel_b], gqgam.kalman.Dyn+quant, quant=0.95, output.vect=FALSE)


gqgam05 <- qgam(list(equation, equation_var), data=Data0[sel_a,], discrete=TRUE, qu=0.5)
gqgam05.forecast <- predict(gqgam, newdata=Data0[sel_b,])
rmse(y=Data0$Load[sel_b], ychap=gqgam05.forecast)





########Net Demand


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
quant <- qnorm(0.8, mean= mean(res), sd= sd(res))
pinball_loss(y=Data0$Net_demand[sel_b], gam_net3.forecast+quant, quant=0.8, output.vect=FALSE)

mean(Data0[sel_b,]$Net_demand<gam_net3.forecast+quant)



X <- predict(gam_net3, newdata=Data0, type='terms')
###scaling columns
for (j in 1:ncol(X))
  X[,j] <- (X[,j]-mean(X[,j])) / sd(X[,j])
X <- cbind(X,1)
d <- ncol(X)

y <- Data0$Net_demand



# static 
ssm <- viking::statespace(X, y)
ssm
ssm$kalman_params
gam_net3.kalman.static <- ssm$pred_mean%>%tail(length(sel_b))

rmse(y=Data0$Net_demand[sel_b], ychap=gam_net3.forecast)
rmse(y=Data0$Net_demand[sel_b], ychap=gam_net3.kalman.static)

sig <- sd(Data0$Load[sel_a]-ssm$pred_mean[sel_a])
quant <- qnorm(0.8, mean= gam_net3.kalman.static, sd= sig)
pinball_loss(y=Data0$Net_demand[sel_b], quant, quant=0.8, output.vect=FALSE)

sig <- ssm$pred_sd%>%tail(length(sel_b))*sd(Data0$Load[sel_a]-ssm$pred_mean[sel_a])
quant <- qnorm(0.8, mean= gam_net3.kalman.static, sd= sig)
pinball_loss(y=Data0$Net_demand[sel_b], quant, quant=0.8, output.vect=FALSE)


# using expectation-maximization
ssm_em <- viking::select_Kalman_variances(ssm, X[sel_a,], y[sel_a], method = 'em', n_iter = 10^3,
                                          Q_init = diag(d), verbose = 10, mode_diag = T)
ssm_em <- predict(ssm_em, X, y, type='model', compute_smooth = TRUE)

#saveRDS(ssm_em, "Results/ssm_em_net_demand.RDS")
ssm_em <-readRDS("Results/ssm_em_net_demand.RDS")

gam_net3.kalman.Dyn.em <- ssm_em$pred_mean%>%tail(length(sel_b))
ssm_em$kalman_params$Q

quant <- qnorm(0.8, mean= gam_net3.kalman.Dyn.em, sd= ssm_em$pred_sd%>%tail(length(sel_b)))
pinball_loss(y=Data0$Net_demand[sel_b], quant, quant=0.8, output.vect=FALSE)




