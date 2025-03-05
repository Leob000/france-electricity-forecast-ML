rm(list=objects())
library(tidyverse)
library(lubridate)
source('R/score.R')

Data0 <- read_delim("Data/train.csv", delim=",")
Data1<- read_delim("Data/test.csv", delim=",")

range(Data0$Date)

# transformer la date en numérique pr pouvoir avoir un ordre
Data0$Time <- as.numeric(Data0$Date)
Data1$Time <- as.numeric(Data1$Date)

# permet de splitter de manière arbitraire le dataset
sel_a <- which(Data0$Year<=2021)
sel_b <- which(Data0$Year>2021) # => test set

plot(Data0$Date,Data0$Net_demand,type='l')
lines(Data0$Date[sel_b],Data0$Net_demand[sel_b],type='l',col='red')

# possibilité de prendre un autre test set (obs avant covid) vu que covid = grosse perturbation
# données saisonnières = bonne idée de tester sur tt les saison => commencer en sept 2021
# possibilité de prendre des données tirés au hasard dans l'échantillon => répartition plus homogène +/ sous estimtion erreur d'extrapolation - 
#   => question importante pr validation=> pr que erreur observé quand on valide nos modèle equivalent à celle sur la plateforme


###############################################################################################################################################################
#####################################################feature engineering
###############################################################################################################################################################

##################################################################################Cycle hebdo
Data0$WeekDays <- as.factor(Data0$WeekDays)
Data1$WeekDays <- as.factor(Data1$WeekDays)
# conversion en facteur de week days au lieu de numérique jeudi pas plus grd que lundi
# tjrs changer sur les deux datasets

# 2 grandes categrories : variable calender et variable météorologiques 
# lesquelles sont les plus importantes ?

mod0 <- lm(Net_demand ~ WeekDays, data=Data0[sel_a,]) # linear models
summary(mod0) 
# on a une relation linéaire entre les x !, (sum(jour semaine)=cst)
# R a vu le prblm et a enlevé le dimanche( ou le lundi ???) pour rendre identifiable
# tt les facteurs/jours sont significatif sauf le jour 4 
# R2 =0.08 =8% de variance de données expliquées => on a du boulot
mod0.forecast <- predict(mod0, newdata=Data0[sel_b,])
rmse(y=Data0$Net_demand[sel_b], ychap=mod0.forecast)
# test de notre modèl# 11k megawatt = grosse erreur



######bloc CV
# on fait une partition temporelle des données au lieu de tirer au hasard
# 8 blocs consécutifs 1-433 434-857 etc
# pr chaque bloc j'enlève un bloc train et puis test sur ce bloc => erreur 
Nblock<-8
borne_block<-seq(1, nrow(Data0), length=Nblock+1)%>%floor
block_list<-list()
l<-length(borne_block)
for(i in c(2:(l-1)))
{
  block_list[[i-1]] <- c(borne_block[i-1]:(borne_block[i]-1))
}
block_list[[l-1]]<-c(borne_block[l-1]:(borne_block[l]))


# f° retire bloc et prédit
fitmod <- function(eq, block)
{
  mod <- lm(eq, data=Data0[-block,])
  mod.cvpred <- predict(mod, newdata=Data0[block,])
  return(mod.cvpred)
}

# on applique la f° a une liste de valeurs=> renvoie une liste qu'on "unlist"
mod0.cvpred<-lapply(block_list, fitmod, eq="Net_demand ~ WeekDays")%>%unlist
rmse(y=Data0$Net_demand, ychap=mod0.cvpred, digits=2)
# 10k tjrs pas fou mais mieux
# =erreur de prévision de mon modèle => on peut calculer dessus le quantile de mon modèle

res <- Data0$Net_demand - mod0.cvpred
quant <- qnorm(0.8, mean= mean(res), sd= sd(res))
pinball_loss(y=Data0$Net_demand[sel_b], mod0.forecast+quant, quant=0.8, output.vect=FALSE)
# estimateur bien meilleur que erreur d'apprentissage direct
#3637 de pinball loss (forets = 1000)

# modèle 
# validatiuon par test set
# validation par bloc cross validation => jouer sur les taille de bloc ci dessus un an
# a jouer avec 


#####regroupement de modalités

Data0$WeekDays2 <- weekdays(Data0$Date)
Data0$WeekDays3 <- forcats::fct_recode(Data0$WeekDays2, 'WorkDay'='Thursday' ,'WorkDay'='Tuesday', 'WorkDay' = 'Wednesday')
# variable qui regroupe mardi mercredi jeudi (similaires)

summary(Data0$WeekDays3)

### attention pas le même sortie que prof ici
mod0 <- lm(Net_demand ~ WeekDays3, data=Data0[sel_a,])
summary(mod0)
mod0.forecast <- predict(mod0, newdata=Data0[sel_b,])
rmse(y=Data0$Net_demand[sel_b], ychap=mod0.forecast)

mod0.cvpred<-lapply(block_list, fitmod, eq="Net_demand ~ WeekDays3")%>%unlist
rmse(y=Data0$Net_demand, ychap=mod0.cvpred, digits=2)


res <- Data0$Net_demand - mod0.cvpred
quant <- qnorm(0.8, mean= mean(res), sd= sd(res))
pb0 <- pinball_loss(y=Data0$Net_demand[sel_b], mod0.forecast+quant, quant=0.8, output.vect=FALSE)
pb0
# légérement meilleur 0.02


################################################################################################################################################################
##################################################################################Temperature
################################################################################################################################################################

###############################################polynomial transforms
# deuxième variable importante après cycle semaine = temperature

# on ajoute une terme lineaire en temp (en vrai polynomial)
mod1 <- lm(Net_demand ~ WeekDays3 + Temp, data=Data0[sel_a,])
summary(mod1)
# R2 74% amélioré
# arrive a voir les niveaux de différences entre les jours de semaines
mod1.forecast <- predict(mod1, newdata=Data0[sel_b,])
rmse(y=Data0$Net_demand[sel_b], ychap=mod1.forecast)
mod1.cvpred<-lapply(block_list, fitmod, eq="Net_demand ~ WeekDays3 + Temp")%>%unlist
rmse(y=Data0$Net_demand, ychap=mod1.cvpred)
# modèle se comporte a peu près pareille en test et cross val => c'est bien le modèle est cohérent

res <- Data0$Net_demand - mod1.cvpred
quant <- qnorm(0.8, mean= mean(res), sd= sd(res))
pb1 <- pinball_loss(y=Data0$Net_demand[sel_b], mod1.forecast+quant, quant=0.8, output.vect=FALSE)
pb1
# bien amelioré

plot(Data0[sel_a,]$Temp,Data0[sel_a,]$Net_demand)
points(Data0[sel_a,]$Temp, mod1$fitted.values, col='red')
# noir vrai donné, rouge = estimation modèle sur jeu apprentissage 
# chaque droite correspond à un jour de la semaine (!= ordonnée a l'originbe), même pente car coeff linéaire pr température 

plot(Data0[sel_a,]$Temp, mod1$residuals)
# si il reste des patterns quand on plot les résidus c'est que notre modèle n'est pas complet

# mode polynomiale de température avec la temp au carré => I pr utiliser une f°
mod2 <- lm(Net_demand ~ WeekDays3 + Temp +I(Temp^2), data=Data0[sel_a,])
mod2.forecast <- predict(mod2, newdata=Data0[sel_b,])
summary(mod2)
# temp et temp**2 significatifs 
# R2 plus il y a de variable plus il gonfle ( de manière artificielle ?) ici 86%
# interessant = variation de R2 ici on gagne 10% de R2 pr un paramètre 
# cool de tracer l'evolution du R2 en f° du nbr de param => critère du coude on arrête d'ajouter des apram
# R2 ajusté = pareil on pénalise en f° du nbr de variables 
rmse(y=Data0$Net_demand[sel_b], ychap=mod2.forecast)

mod2.cvpred<-lapply(block_list, fitmod, eq="Net_demand ~ WeekDays3 + Temp +I(Temp^2)")%>%unlist
rmse(y=Data0$Net_demand, ychap=mod2.cvpred)
# tjrs un ecart avec cross validation meilleure erreur (plus dur de prevoir dans futur)

res <- Data0$Net_demand - mod2.cvpred
quant <- qnorm(0.8, mean= mean(res), sd= sd(res)) # on suppose modèle sans biais = pas forcément vrai
pb2 <- pinball_loss(y=Data0$Net_demand[sel_b], mod2.forecast+quant, quant=0.8, output.vect=FALSE)
pb2

plot(res,type='l') # => pattern évident (cycle erreur extrèmes changement avant après 2020)
hist(res) # pas gaussien plus de residus négatifs loi asymmétrique
# pinnball on fit une loi normale et on prend le quantile 
# possibilité de prendre direct le quantile de l'echantillon, option à tester

plot(Data0[sel_a,]$Temp, mod2$residuals)
# bcp mieux que tt a l'heure on distingue moins bien une forme 
# on voit encore que quand t° faible variance plus forte 
# si on voit pas de forme dans le nuage on peut faire un lissafe des données pour voir s'il y a pas une forme discrètement caché
g0=mgcv::gam(mod2$residuals~s(Data0$Temp[sel_a]))
points(Data0$Temp[sel_a],g0$fitted,col="red")
# modèle à tendance à surestimer le Y, vaguelette au milieu = surestimation aussi, pour la clim il surestime aussi 

plot(Data0$Date,Data0$Net_demand- mod2.cvpred, type='l')
lines(Data0$Date[sel_b], Data0$Net_demand[sel_b]-mod2.forecast, col='red')


##variance des scores par bloc?
mod1.rmse_bloc <- lapply(block_list, function(x){rmse(y=Data0$Net_demand[x], ychap=mod1.cvpred[x])})%>%unlist
mod2.rmse_bloc <- lapply(block_list, function(x){rmse(y=Data0$Net_demand[x], ychap=mod2.cvpred[x])})%>%unlist

col <- yarrr::piratepal("basel")
boxplot(cbind(mod1.rmse_bloc, mod2.rmse_bloc), col=col[1:2], ylim=c(2000, 10000))
abline(h=rmse(y=Data0$Load[sel_b], ychap=mod1.forecast), col=col[1], lty='dotted')
abline(h=rmse(y=Data0$Load[sel_b], ychap=mod2.forecast), col=col[2], lty='dotted')

# quand cross validation par bloc 
# intéressant de regarder RMSE pour savoir si elle est fiable (change d'un bloc à l'autre ?) => variabilité RMSE
# est ce que variabilité RMSE similaire d'un modèle à l'autre ?
# en ligne pointillé = erreur sur test set, pour les deux modèles largeent plus élevé car dificile de prédire futur
# deuxième modèle= gain significatif de RMSE

# ici bonne manière d'avoir marge rreru sur erreur validation du modèle

###############################################truncated power functions
# for(i in c(1:11))
# {
#   x<-pmax(eval(parse(text=paste0("Data0$Station",i)))-65,0)
#   assign(paste("Station",i,".trunc.65",sep=""), x)
# }

# température tronqué
# fabrication a la main de f° spleen de degré 1 
# temp-seuil partie posiitive
# on peut ajuster des f° linéaire par morceaux avec rupture au seuils 


plot(Data0$Temp, Data0$Net_demand, pch=20)
Data0$Temp_trunc1 <- pmax(Data0$Temp-286,0)
Data0$Temp_trunc2 <- pmax(Data0$Temp-290,0)

plot(Data0$Temp, Data0$Temp_trunc1 , pch=20)


mod3 <- lm(Net_demand ~ WeekDays3 + Temp + Temp_trunc1 + Temp_trunc2, data=Data0[sel_a,])
mod3.forecast <- predict(mod3, newdata=Data0[sel_b,])
summary(mod3)
# R2 0.87 pas mal
rmse(y=Data0$Net_demand[sel_b], ychap=mod3.forecast)

plot(Data0$Temp[sel_a],  mod3$residuals, pch=20)
g0=mgcv::gam(mod3$residuals~s(Data0$Temp[sel_a]))
points(Data0$Temp[sel_a],g0$fitted,col="red")
# a gauche = mauvais = expliqué par chauffage d'appoint => rajouter un seuil à 276 pr améliorer le modèle

plot(Data0$Temp[sel_a],  Data0$Net_demand[sel_a], pch=20)
points(Data0$Temp[sel_a],  mod3$fitted.values, pch=20,col='red')
# on pourrait améliorer le modèle en disant que la pente de la clim est la même selon que c'est un jour de wkd ou de semaine
# rajouter intéraction :     Temp:Weekdays  


mod3.cvpred<-lapply(block_list, fitmod, eq="Net_demand ~ WeekDays3 + Temp + Temp_trunc1 + Temp_trunc2")%>%unlist
rmse(y=Data0$Net_demand, ychap=mod3.cvpred)
res <- Data0$Net_demand - mod3.cvpred
quant <- qnorm(0.8, mean= mean(res), sd= sd(res))
pb3 <- pinball_loss(y=Data0$Net_demand[sel_b], mod3.forecast+quant, quant=0.8, output.vect=FALSE)
pb3

mod3.rmse_bloc <- lapply(block_list, function(x){rmse(y=Data0$Net_demand[x], ychap=mod3.cvpred[x])})%>%unlist

col <- yarrr::piratepal("basel")
boxplot(cbind(mod2.rmse_bloc, mod3.rmse_bloc), col=col[1:2], ylim=c(2000, 7000))
abline(h=rmse(y=Data0$Net_demand[sel_b], ychap=mod2.forecast), col=col[1], lty='dotted')
abline(h=rmse(y=Data0$Net_demand[sel_b], ychap=mod3.forecast), col=col[2], lty='dotted')
# nouveau modèle à plus grande variance, médiane des rmse plus faible, premier moins bon globalement mais plus faible variance
#(sur les blocs hein)
# on a fait boxplot sur rmse mais on doit aussi le faire sur le pinballloss 


plot(Data0$Date[sel_b], mod3.cvpred[sel_b], type='l')
lines(Data0$Date[sel_b], mod3.forecast, col='red')

plot(Data0[sel_a,]$Temp, mod2$residuals)
points(Data0[sel_a,]$Temp, mod3$residuals, col='red')

plot(Data0$Date, Data0$Net_demand-mod3.cvpred, type='l')
# erreur de prévision du modèle 
# on voit tjrs une tendance dans les résidus, un cycle annuel mal prédit 


##################################################################################cycle annuel: fourier

# modélisation de phénomène cyclique => SERIES DE FORUIER 
# composante périodique annuelle => je fabrique des variables associé à cycle anbnuel avec truc de fourier et projeter dessus

w<-2*pi/(365)
Nfourier<-50
for(i in c(1:Nfourier))
{
  assign(paste("cos", i, sep=""),cos(w*Data0$Time*i))  # assigner des valeurs à une chaine de caractère
  assign(paste("sin", i, sep=""),sin(w*Data0$Time*i)) # w =2pi/période
}
objects()
plot(Data0$Date, cos1,type='l')  # cos10 fréquence bcp plus grd

cos<-paste('cos',c(1:Nfourier),sep="",collapse=",")                         
sin<-paste('sin',c(1:Nfourier),sep="",collapse=",")

Data0<-eval(parse(text=paste("data.frame(Data0,",cos,",",sin,")",sep="")))
names(Data0)

# paste => 50 nouvelles variables de fourier ajoutés




Nfourier<-30
lm.fourier<-list()
eq<-list()
for(i in c(1:Nfourier)) # 30 modèles emboités 
{
  cos<-paste(c('cos'),c(1:i),sep="")
  sin<-paste(c('sin'),c(1:i),sep="")
  fourier<-paste(c(cos,sin),collapse="+")
  eq[[i]]<-as.formula(paste("Net_demand~ WeekDays3 + Temp + Temp_trunc1 + Temp_trunc2+",fourier,sep=""))
  lm.fourier[[i]]<-lm(eq[[i]],data=Data0[sel_a,])
}

lm(eq[[1]], data=Data0)


#on récupère le R2 adj et le RMSE en apprentissage/test => représenter comment ça évolue en ajoutant des coeffs de fourier
#  :

adj.rsquare<-lapply(lm.fourier,
                    function(x){summary(x)$adj.r.squared})%>%unlist


fit.rmse<-lapply(lm.fourier,
                  function(x){rmse(Data0$Net_demand[sel_a],x$fitted)})%>%unlist

forecast.rmse<-lapply(lm.fourier
                      , function(x){rmse(Data0$Net_demand[sel_b],predict(x,newdata=Data0[sel_b,]))})%>%unlist

fit.mape<-lapply(lm.fourier,
                 function(x){mape(Data0$Net_demand[sel_a],x$fitted)})%>%unlist

forecast.mape<-lapply(lm.fourier
                      , function(x){mape(Data0$Net_demand[sel_b],predict(x,newdata=Data0[sel_b,]))})%>%unlist


plot(adj.rsquare,type='b',pch=20)
# a partir de 20 coeff on plafonne   0.91 R2adj cool

plot(fit.rmse,type='b',pch=20, ylim=range(fit.rmse, forecast.rmse), col='royalblue2')
lines(forecast.rmse,type='b',pch=20, col='orangered2')
legend('top', c("fit", "forecast"), col=c('royalblue2', 'orangered2'), lty=1)
# erreur apprentissage et test décroit avec coude (quadratique puis linéaire)
# pas la peine d'aller plus loin que le coude => entre 10 et 15

mod4 <- lm(formula(lm.fourier[[10]]), data=Data0[sel_a,])
mod4.cvpred<-lapply(block_list, fitmod, eq=formula(lm.fourier[[15]]))%>%unlist
mod4.forecast <- predict(mod4, newdata = Data0[sel_b,])
rmse(y=Data0$Net_demand, ychap=mod4.cvpred)
rmse(y=Data0$Net_demand[sel_b], ychap=mod4.forecast)
mod4.rmse_bloc <- lapply(block_list, function(x){rmse(y=Data0$Net_demand[x], ychap=mod4.cvpred[x])})%>%unlist

res <- Data0$Net_demand - mod4.cvpred
quant <- qnorm(0.8, mean= mean(res), sd= sd(res))
pb4 <- pinball_loss(y=Data0$Net_demand[sel_b], mod4.forecast+quant, quant=0.8, output.vect=FALSE)
pb4

plot(Data0$Date[sel_a], mod4$residuals, type='l')
# on a tjrs une tendance (patterne en variance l'hiver) mais on a tjrs des vagues, on a un peu amélioré la saisonnalité 
acf(Data0$Net_demand, lag.max=7*3) # correlation yt avec yt-1 yt-2 etc
# corrélation forte on voit saisonnalité hebdomadaire

acf( mod4$residuals, lag.max=7*3) 
# => boom autocorrelation bien baissé !, il en reste pas mal sur les jours précédents et on a un résidu de cycle hebdomadaire

acf( mod4$residuals, lag.max=7*52) # on voit pas de cycle annuel => génial on l'a bien modélisé

# on rajoute variables -1 et -7 pr améliorer modèles du pdv autocorrleation
form <- eq[[10]]
form <- buildmer::add.terms(form, "Net_demand.1")
form <- buildmer::add.terms(form, "Net_demand.7")


mod5 <- lm(form, data=Data0[sel_a,])
mod5.forecast <- predict(mod5, newdata=Data0[sel_b,])
summary(mod5)

rmse(y=Data0$Net_demand[sel_b], ychap=mod4.forecast)
rmse(y=Data0$Net_demand[sel_b], ychap=mod5.forecast)

mod5.cvpred<-lapply(block_list, fitmod, eq=form)%>%unlist
rmse(y=Data0$Net_demand, ychap=mod5.cvpred)


res <- Data0$Net_demand - mod5.cvpred
quant <- qnorm(0.8, mean= mean(res), sd= sd(res))
pb5 <- pinball_loss(y=Data0$Net_demand[sel_b], mod5.forecast+quant, quant=0.8, output.vect=FALSE)

# 2 param très signigicatifs 
# 3900->2336 sur test
# bloc 2100
# pinball 638 carré grosse améliroation (/2)


# SYNTHESE DE CI DESSUS DIFF MODELES
synthese.test <- c(rmse(y=Data0$Net_demand[sel_b], ychap=mod1.forecast),
                 rmse(y=Data0$Net_demand[sel_b],, ychap=mod2.forecast),
                 rmse(y=Data0$Net_demand[sel_b],, ychap=mod3.forecast),
                 rmse(y=Data0$Net_demand[sel_b],, ychap=mod4.forecast),
                 rmse(y=Data0$Net_demand[sel_b],, ychap=mod5.forecast)
)

synthese.cv <- c(rmse(y=Data0$Net_demand, ychap=mod1.cvpred),
  rmse(y=Data0$Net_demand, ychap=mod2.cvpred),
  rmse(y=Data0$Net_demand, ychap=mod3.cvpred),
  rmse(y=Data0$Net_demand, ychap=mod4.cvpred),
  rmse(y=Data0$Net_demand, ychap=mod5.cvpred)
)

# courbes à montrer dans rapport à la fin 
# aus sens de critère regardé et critère plateforme qu'est ce qui a marché ou pas etc 
plot(synthese.test, type='b', pch=20, ylim=c(1000, 7000))
lines(synthese.cv, col='red', pch=20, type='b')


synthese.pb <- c(pb0, pb1, pb2, pb3, pb4, pb5)
plot(synthese.pb, type='b', pch=20)


###########################################################################################
#############soumission d'une prévision
###########################################################################################
Data1$WeekDays2 <- weekdays(Data1$Date)
Data1$WeekDays3 <- forcats::fct_recode(Data1$WeekDays2, 'WorkDay'='Thursday' ,'WorkDay'='Tuesday', 'WorkDay' = 'Wednesday')

Data1$Temp_trunc1 <- pmax(Data1$Temp-286,0)
Data1$Temp_trunc2 <- pmax(Data1$Temp-290,0)



##################################################################################cycle annuel: fourier
w<-2*pi/(365)
Nfourier<-50
for(i in c(1:Nfourier))
{
  assign(paste("cos", i, sep=""),cos(w*Data1$Time*i))
  assign(paste("sin", i, sep=""),sin(w*Data1$Time*i))
}
objects()
plot(Data1$Date, cos1,type='l')

cos<-paste('cos',c(1:Nfourier),sep="",collapse=",")                         
sin<-paste('sin',c(1:Nfourier),sep="",collapse=",")

Data1<-eval(parse(text=paste("data.frame(Data1,",cos,",",sin,")",sep="")))
names(Data1)


# on refait un modèle final en apprenant sur toute nos données (train + test)

mod5final <- lm(form, data=Data0)

###prev moyenne
lm.forecast <- predict(mod5final, newdata=Data1)
###prev proba
res <- Data0$Net_demand - mod5.cvpred
quant <- qnorm(0.8, mean= mean(res), sd= sd(res))

pinball_loss(y=Data0$Net_demand[sel_b], mod5.forecast+quant, quant=0.8, output.vect=FALSE)
# pinball loss sur test

submit <- read_delim( file="Data/sample_submission.csv", delim=",")
submit$Net_demand <- lm.forecast+quant  # remplacer netdemand par notre prévision
write.table(submit, file="Data/submission_lm.csv", quote=F, sep=",", dec='.',row.names = F)
# partie soummission sur kaggle, plus qu'à l'uploader sur la plateforme 


########################################
#####glm
########################################

# variante du modèle linéaire, modèle linéaire généralisé, dans lequel on modélise également la variance de nos données 
# c'était pas le cas au dessus, au dessus que biais 
res <- Data0$Net_demand - mod5.cvpred
hist(res, breaks=50)
plot(res, type='l')
eq <- form                           # equation sur variance ou je fait dépendre 
# la varaince de variables :
eq_var <- ~ WeekDays3 + Temp + Temp_trunc1 + Temp_trunc2 

# f° gam on lui met une liste d'équation, on précise la famille de lois
# aler voir eventuellement lois inconnues intéressantes (mode valeur extrêmes, modélisation paramètre d'assymétrie)
# peut donner des idées de bricolages de voir ça 
glss <- mgcv::gam(list(eq, eq_var), data=Data0[sel_a,], family="gaulss")
summary(glss)
# toute la partie linéaire, puis les .1 correspond partie sur variance 
# outil puisssant, modélisation comportement très variés 
glss.forecast <- predict(glss, newdata=Data0[sel_b,]) 
# colonne 1 prevision esperance colonne 2 = f° lien de sd
sigma <- 0.01 + exp(glss.forecast[,2]) # inverse f° lien
gaulss_quant <- glss.forecast[,1] + qnorm(p=0.8, mean=0, sd=sigma)


pb_glm <- pinball_loss(y=Data0$Net_demand[sel_b], gaulss_quant, quant=0.8, output.vect=FALSE)
pb_glm
# 638->597 (sans optimiser la partie variance)




########################################
#####Quantile regression
########################################

# regression quantile optimise la pinballloss

library("quantreg")

mod5.rq <- rq(form, data = Data0[sel_a, ], tau=0.8)
summary(mod5.rq)

mod5.rq.forecast <- predict(mod5.rq, newdata=Data0[sel_b,])
pb_rq <- pinball_loss(y=Data0$Net_demand[sel_b], mod5.rq.forecast, quant=0.8, output.vect=FALSE)
pb_rq
# en vrai au final pas fou 602 au lieu de 597, kiff kiff mais plus simple

synthese.pb <- c(pb3, pb4, pb5, pb_glm, pb_rq)
plot(synthese.pb, type='b', pch=20)

h = hist(res, breaks=50, freq=FALSE)
x = seq(min(res), max(res), length=100)
lines(x, dnorm(x=x, mean=mean(res), sd=sd(res)), col='red')
abline(v= quantile(res, 0.8), col='blue', lwd=2)

qqnorm(res)

# modèle prevision quantile 0.7 au lieu de 0.8
mod5.rq <- rq(form, data = Data0[sel_a, ], tau=0.7)
summary(mod5.rq)

# modèle 0.7 utilisé pr prédire 0.8 => je pense que les données futures ont baissé de 10% dans le futur
mod5.rq.forecast <- predict(mod5.rq, newdata=Data0[sel_b,])
pb_rq2 <- pinball_loss(y=Data0$Net_demand[sel_b], mod5.rq.forecast, quant=0.8, output.vect=FALSE)
pb_rq2
# pinball loss =555 amélioré significativement 
# => prévision conforme ça de corriger algo en observant les nouvelles données

# objectif semaine pro= soumettre des trucs, expliquer les options retenues et leur résultats


######################################################################################################################################################
##############################Annexes
#####################################################################################################################################################

########################################
#####Méthode d'ensemble
########################################
mod5 <- lm(form, data=Data0[sel_a,])
mod5.forecast <- predict(mod5, newdata=Data0[sel_b,])
summary(mod5)
rmse(y=Data0$Net_demand[sel_b], ychap=mod5.forecast)

mod5.cvpred<-lapply(block_list, fitmod, eq=form)%>%unlist


fit.ensemble <- function(eq, block)
{
  mod <- lm(eq, data=Data0[-block,])
  mod.forecast <- predict(mod, newdata=Data1)
  return(mod.forecast)
}

mod5.ensemble <-lapply(block_list, fit.ensemble, eq=form)

mod5.ensemble <- mod5.ensemble%>%unlist%>%matrix(ncol=length(block_list), nrow=nrow(Data1), byrow=F)
mod5.ensemble%>%head
mod5.ensemble <- rowMeans(mod5.ensemble)




submit <- read_delim( file="Data/sample_submission.csv", delim=",")
submit$Net_demand <- mod5.ensemble
write.table(submit, file="Data/submission_lm_ensemble_block.csv", quote=F, sep=",", dec='.',row.names = F)



######random CV

fit.ensemble.random <- function(eq, block)
{
  mod <- lm(eq, data=Data0[block,])
  mod.forecast <- predict(mod, newdata=Data1)
  return(mod.forecast)
}

n <- nrow(Data0)
block2 <- lapply(rep(0.5, 100), function(rate){sample(c(1:n), size=floor(rate*n), replace=T)})
mod5.ensemble.random <-lapply(block2, fit.ensemble.random, eq=form)

mod5.ensemble.random <- mod5.ensemble.random%>%unlist%>%matrix(ncol=length(block2), nrow=nrow(Data1), byrow=F)
mod5.ensemble.random%>%head
mod5.ensemble.mean <- rowMeans(mod5.ensemble.random)

matplot(mod5.ensemble.random, type='l', col='gray')
lines(mod5.ensemble.mean)


fit.ensemble.random2 <- function(eq, block, Data)
{
  mod <- lm(eq, data=Data[block,])
  return(mod)
}

n <- nrow(Data0[sel_a,])
block2 <- lapply(rep(0.9, 100), function(rate){sample(c(1:n), size=floor(rate*n), replace=T)})
mod.ensemble <-lapply(block2, fit.ensemble.random2, Data=Data0[sel_a,], eq=form)
mod.ensemble.test <- lapply(mod.ensemble, predict, newdata=Data0[sel_b,])
mod.ensemble.test <- mod.ensemble.test%>%unlist%>%matrix(ncol=length(block2), nrow=nrow(Data0[sel_b,]), byrow=F)
rmse(y=Data0$Net_demand[sel_b], ychap=rowMeans(mod.ensemble.test))

rmse.random <- apply(mod.ensemble.test, 2, rmse, y=Data0$Net_demand[sel_b])
boxplot(rmse.random)
abline(h=rmse(y=Data0$Net_demand[sel_b], ychap=rowMeans(mod.ensemble.test)), col='red')

plot(synthese.test, type='b', pch=20, ylim=c(1000, 7000))
lines(synthese.cv, col='red', pch=20, type='b')
abline(h=rmse(y=Data0$Net_demand[sel_b], ychap=rowMeans(mod.ensemble.test)), col='red')




submit <- read_delim( file="Data/sample_submission.csv", delim=",")
submit$Net_demand <- mod5.ensemble.mean
write.table(submit, file="Data/submission_lm_ensemble_random.csv", quote=F, sep=",", dec='.',row.names = F)


########################################
#####Quantile regression
########################################
library("quantreg")

fitmod.rq <- function(eq, block, tau=0.5)
{
  mod <- rq(eq, data=Data0[-block,], tau)
  mod.cvpred <- predict(mod, newdata=Data0[block,])
  return(mod.cvpred)
}

mod5.rq <- rq(form, data = Data0[sel_a, ], tau=0.5)
summary(mod5.rq)

mod5.rq.forecast <- predict(mod5.rq, newdata=Data0[sel_b,])

rmse(y=Data0$Net_demand[sel_b], ychap=mod5.rq.forecast)

mod5.rq.cvpred<-lapply(block_list, fitmod.rq, eq=form)%>%unlist
rmse(y=Data0$Net_demand, ychap=mod5.rq.cvpred)




