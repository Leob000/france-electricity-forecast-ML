rm(list=objects())
library(tidyverse)
library(lubridate)
library(forecast)


#install.packages("tidyverse")

source("R/score.R")
# Data0 <- read_delim("Data/train.csv", delim=",")
# Data1<- read_delim("Data/test.csv", delim=",")


Data0 <- read_csv('Data/net-load-forecasting-during-the-soberty-period/train.csv')
Data1 <- read_csv('Data/net-load-forecasting-during-the-soberty-period/test.csv')


summary(Data0)
# type tibble = dataframe amélioré   3471*39 mixe entre liste et matrice intérêt liste = type différents selon colonnes
# 3k info = bof pr deep learning transformers etc
# variable quanti quali 
# date => format date obtenu  mars 2023-sept 2022
# load = Y à prévoir= conso électrique en Mwatt exprimé en moyenne journalière 
# load1 et 7= load des 7 jours précédents (cycle hebdomadaire de la conso électrique) 
# second Y = net demand = conso électrique - production de renouvelable (prod° solaire et autres variables)
# solar power (tendance d'augmentation en arrivant à ajd)  au max = 3 tranches de centrales nucléaires
# wind power
# => variable renouvelable = étudier distribution variabilité évolution
# température kelvin (représentative de tt la france (moyenne spatiale)) =>possibilité d'améliorer cette variable en allant chercher les données sur météo france
# par ex prendre que t° et autre du sud de la france au lieu de moyenne spatiale pr prod° solaire
# temp s95 s99 = temp lissage exponentielle 
# temp s95 99 min max => amplitude de température de la journée 
# wind =vitesse moyenne du vent /demi heure 
# wind weigthed = pondéré selon les différentes régions s'il y a plus d'éolienne 
# nébulosité =couverture nuageuse / opacité en %
# nébulosité weighted =selon répartition panneaux solaires 
# variables déterministe calendar :
# toy time of year 0 début année 1 fin => cyclicité
# weekdays jours de la semaine à mettre en facteur
# bh= bank holiday =0 1 (3% des jours) => gros impacte /conso électrique
# bh before after pour les ponts       bh= samedi dimanche mix des deux
# year
# mois
# dls = changement d'heure 
# vacances d'été / hiver
# vacances communes a tt les zones
# vacances par zone
# bh holiday jour férié pendant les vacances 
# solar wind net demande 1 et 7 jours précédents 
# jours précédents => Mettre les autres il y a que les -1 et -7 mettre -2 3 4 5 6



range(Data0$Date)
range(Data1$Date)

names(Data0)
head(Data0[, c("Date", "WeekDays")])

weekdays(head(Data0$Date, 7))

# 5: Saturday
# 6: Sunday
# 0: Monday
# ...

###################################analyse de la demande nette

############################################trend
plot(Data0$Date, Data0$Net_demand, type='l', xlim=range(Data0$Date, Data1$Date))

# cycle annuel 
# variabilité beaucoup plus haute en hiver car t° influe davantage et change bcp plus brutalement
# période intersaison plus dures a prévoir
# conso 2020 un peu plus faible 
# consommation nette = conso -renouvelable 
# baisse de conso => 2 causes confinement et post confinement (délocalisation) et aussi augmentation de production énergie renouvelable
# plat jusqu'a 2020 puis baisse en gros 
# pics vers le haut forte conso
# pic vers le bas = jours fériés 
# creux du mois d'aout tt le temps

# tendance de base 
# cycle annuel (t° vacances jours fériés variance été hiver)


col <- yarrr::piratepal("basel")
par(mfrow=c(3,1))
plot(Data0$Date, Data0$Load, type='l', col=col[1])
plot(Data0$Date, Data0$Solar_power, type='l', col=col[2])
plot(Data0$Date, Data0$Wind_power, type='l', col=col[3])

# bleu conso rouge prod solaire vert prod eolienne 
# => forte augmentation prod éolienne/solaire  éolienne stagne un peu solaire + de potentiel de développement
# variabilité augmente au cours du temps aussi => typique modèle multiplicatif (variance multiplicative pas additive ) => passage au log peut être utile
# stratégie sur le modèle = soit demande nette soit différents modèles pr conso prod sol vent puis soustraction
# cycles annuels + solaires en été + vent en hiver 
# on voit que c'est une petite partie de la conso donc si on modélise bien la conso électrique on a dejà fait une grosse partie du travail


par(mfrow=c(1,1))
plot(Data0$Date, Data0$Load, type='l', ylim=range(Data0$Solar_power, Data0$Load), col=col[1])
lines(Data0$Date, Data0$Wind_power, col=col[3])
lines(Data0$Date, Data0$Solar_power, col=col[2])

# histogramme de net demand
# => pas gaussien à priori mais possible que comme données brutes en fait conditionnellement à nos variables ce soit gaussien
# par exemple conso nette hiver et conso nette été = deux gaussiennes entrecroisés
# trucs basse fréquence queues hautes = hiver      haute fréquence queues basse = été
# erreur du modèle gaussienne oupas c'est ça qu'il faudra regarder

mean(Data0$Load)
mean(Data0$Wind_power)
mean(Data0$Solar_power)

par(mfrow=c(1,1))
hist(Data0$Net_demand, breaks=100)
#plot(Data0$Date, Data0$Temp, type='l')

plot(Data0$Date, Data0$Net_demand, type='l', xlim=range(Data0$Date, Data1$Date))
K <- 7
smooth <- stats::filter(Data0$Net_demand, rep(1/K, K))
lines(Data0$Date, smooth, col='red', lwd=2)


# lissage des données avec des convolutions (K=7 pr cycle hebdomadaire)
# on fait une moyenne des données par 7j (courbe rouge) => on perd la variabilité hebdomadaire 
# => grosse baisse de variabilité => été variance données expliqués par cycle hebdomadaire 
# à jouer avec ces outils de lissage (à appliquer sur d'autres variables aussi)


############################################yearly cycle
sel <- which(Data0$Year==2021)
plot(Data0$Date[sel], Data0$Net_demand[sel], type='l')

# zoom /2021  on voit mieux le cycle hebdomadaire (surtout en été moins en hiver)
# mois de mai bcp jours fériés = plus chahuté 

plot(Data0$toy)

# time of year ~= mois

col.tr <- adjustcolor(col='black', alpha=0.3)
plot(Data0$toy, Data0$Net_demand, pch=16,  col=col.tr)

# variables en f° de toy => apparition de strate en été => plus de donnée = jour de semaine 
# weekdend plus bas 
# cycle annuel à modéliser avec des bases de splin periodique (f° périodique)

par(mfrow=c(1,1))
plot(Data0$toy, Data0$Load, pch=16, col=col.tr[1])

# même graphique pr conso : strates beaucoups plus apparentes 

col.tr <- adjustcolor(col, alpha=0.3)
par(mfrow=c(3,1))
plot(Data0$toy, Data0$Load, pch=16, col=col.tr[1])
plot(Data0$toy, Data0$Solar_power, pch=16, col=col.tr[2])
plot(Data0$toy, Data0$Wind_power, pch=16, col=col.tr[3])

# pour nos 3 variables à prédire => on voit plus grosse production en été solaire et aussi plus grosse variance 
# même chose pr éolienne en inversement avec hiver

par(mfrow=c(3,1))
boxplot(Net_demand~Month, data=Data0, col=col[1])
boxplot(Solar_power~Month, data=Data0, col=col[2])
boxplot(Wind_power~Month, data=Data0, col=col[3])

# on voit les variances plus grandes en hiver été etc encore mieux 

############################################Weekly cycle
par(mfrow=c(1,1))

sel <- which(Data0$Month==6 & Data0$Year==2021) 
plot(Data0$Date[sel], Data0$Net_demand[sel], type='l')

# mois de juin 2021 on voit encore mieux les samedi dimanche en bas 
# jours de semaine plus haut et mardi mercredi jeudi plus haut car inertie weekend


par(mfrow=c(3,1))
boxplot(Net_demand~WeekDays, data=Data0, col=col[1])
boxplot(Solar_power~WeekDays, data=Data0, col=col[2])
boxplot(Wind_power~WeekDays, data=Data0, col=col[3])

# boxplot => on voit que cycle hebdomadaire présent que sur copnsommation (logique)

par(mfrow=c(1,1))
boxplot(Net_demand~WeekDays, data=Data0)

# conso explique plus la net demand => on voit le cycle

plot(Data0$Load.1, Data0$Load)
cor(Data0$Load.1, Data0$Load)

# influence conso de la veille (conso en f° conso de la veille) => forte corrélation linéaire 0.93
# phénomène été hiver conso faible j précédent=> conso faible ajd
# 3 droites != =>transitions jours de semaines weekend

par(mfrow=c(1,3))
Acf(Data0$Load, lag.max=7*10, type = c("correlation"), col=col[1], ylim=c(0,1))
Acf(Data0$Solar_power, lag.max=7*10, type = c("correlation"), col=col[2], ylim=c(0,1))
Acf(Data0$Wind_power, lag.max=7*10, type = c("correlation"), col=col[3], ylim=c(0,1))

# autocorrélogramme = corrélation de ma série avec ma série dans le passé 
# en abscice le lag (t-1 -2 -3)  ordonné corrélation mesuré 
# pr conso on voir cycle hebdomadaire
### correlation -7 la plus forte mais -2 et -6 et -8 les plus correllés aussi à rajouter 
# -70j corrélé 0.2 => parce que saisonalité 
# aspect transitif des corrélations  yt corr à yt-1 et yt-1 corr yt-9 => yt corr yt-9
# on voit que basse fréquence dans nos données  j'ai zappé ce qu'il a dit là

# données solaires décroissance lente = saisonalité et tendance 
# vent moins de tendance et saisonalité moins marqué 
# on voit que corrélation à courtterme très forte pr les deux (caractéristqieus météo)
# on a perdu totalement le cycle hebdomadaire 
# on voit que solaire - bruité que vent
# vent très bruité )> raison pr laquelle on est plus haut à 50 qu'a 45 en vrai si tu vire le bruit c'est les mêmes


par(mfrow=c(1,3))
Acf(Data0$Load, lag.max=7*60, type = c("correlation"), col=col[1], ylim=c(-1,1))
Acf(Data0$Solar_power, lag.max=7*60, type = c("correlation"), col=col[2], ylim=c(-1,1))
Acf(Data0$Wind_power, lag.max=7*60, type = c("correlation"), col=col[3], ylim=c(-1,1))


####################################################################################################################################
############################################Meteo effect/covariates
####################################################################################################################################


### bivarié 

############################################Temperature
par(mar = c(5,5,2,5))
par(mfrow=c(1,1))
plot(Data0$Date, Data0$Net_demand, type='l')
par(new=T)
plot(Data0$Date, Data0$Temp, type='l', col='red', axes=F,xlab='',ylab='')
#plot(Data0$Temp%>%tail(1000), type='l', col='red', axes=F,xlab='',ylab='')
axis(side = 4,col='red', col.axis='red')
mtext(side = 4, line = 3, 'Temperature', col='red')

legend("top",c("Net_demand","Temperature"),col=c("black","red"),lty=1,ncol=1,bty="n")

# correlation forte négative conso température (outil superposition graphes séries temporelles)


col.tr <- adjustcolor(col='black', alpha=0.25)
plot(Data0$Temp, Data0$Net_demand, pch=3,  col=col.tr)

# baisse de 1 degré conso augmente de 2000 Mw 
# ne baisse plus en été (temp haute) points bas => vacances en vrai effet clim et légère augmentation de conso
# modèle linéaire simple en f° de t° ne poeut pas marcher

plot(Data0$Date%>%head(,n=7*3), Data0$Temp%>%head(,n=7*3), type='l')
lines(Data0$Date%>%head(,n=7*3), Data0$Temp_s95%>%head(,n=7*3), col='blue')
lines(Data0$Date%>%head(,n=7*3), Data0$Temp_s99%>%head(,n=7*3), col='red')

# lissage et lissage maxmin

plot(Data0$Date%>%head(,n=7*5), Data0$Temp_s99%>%head(,n=7*5), type='l')
lines(Data0$Date%>%head(,n=7*5), Data0$Temp_s99_min%>%head(,n=7*5), col='blue')
lines(Data0$Date%>%head(,n=7*5), Data0$Temp_s99_max%>%head(,n=7*5), col='red')

par(mfrow=c(1,1))
col.tr1 <- adjustcolor(col='black', alpha=0.25)
col.tr2 <- adjustcolor(col='red', alpha=0.25)
plot(Data0$Temp, Data0$Net_demand, pch=3,  col=col.tr1)
points(Data0$Temp_s99, Data0$Net_demand, pch=3, col=col.tr2)

# nuage de point temp brute lissé=> variabilité de conso brute plus grande en hiver (subtil)
# se voit que en hover car autres variables parasitent graphique

col.tr <- adjustcolor(col, alpha=0.25)
par(mfrow=c(3,1))
plot(Data0$Temp, Data0$Load, pch=3,  col=col.tr[1])
plot(Data0$Temp, Data0$Solar_power, pch=3,  col=col.tr[2])
plot(Data0$Temp, Data0$Wind_power, pch=3,  col=col.tr[3])

# effet clim plus net sur conso brut que nette 
# on revoit les strate hebdomadaires 
# prod solaire dépend temp pareil vent mais juste correlation du temps pas causalité 

############################################Wind
par(mfrow=c(2,1))
plot(Data0$Date, Data0$Wind, type='l')
plot(Data0$Date, Data0$Wind_weighted, type='l')


par(mfrow=c(3,1))
plot(Data0$Wind, Data0$Load, pch=3,  col=col[1])
plot(Data0$Wind, Data0$Solar_power, pch=3,  col=col[2])
plot(Data0$Wind, Data0$Wind_power, pch=3,  col=col[3])

par(mfrow=c(1,1))
plot(Data0$Wind, Data0$Wind_power, pch=3,  col=col[3])
points(Data0$Wind_weighted, Data0$Wind_power, pch=3,  col=col[4])



par(mfrow=c(1,1))
plot(Data0$Date, Data0$Net_demand, type='l')
par(new=T)
plot(Data0$Date, Data0$Wind, type='l', col='red', axes=F,xlab='',ylab='')
#plot(Data0$Temp%>%tail(1000), type='l', col='red', axes=F,xlab='',ylab='')
axis(side = 4,col='red', col.axis='red')
mtext(side = 4, line = 3, 'Wind', col='red')
legend("top",c("Net_demand","Wind"),col=c("black","red"),lty=1,ncol=1,bty="n")


K <- 7*4
smooth_net <- stats::filter(Data0$Net_demand, rep(1/K, K))
smooth_wind <- stats::filter(Data0$Wind, rep(1/K, K))
par(mfrow=c(1,1))
plot(Data0$Date, smooth_net, type='l')
par(new=T)
plot(Data0$Date, smooth_wind, type='l', col='red', axes=F,xlab='',ylab='')
#plot(Data0$Temp%>%tail(1000), type='l', col='red', axes=F,xlab='',ylab='')
axis(side = 4,col='red', col.axis='red')
mtext(side = 4, line = 3, 'Wind', col='red')
legend("top",c("Net_demand","Wind"),col=c("black","red"),lty=1,ncol=1,bty="n")

# lissage des variables : fenêtre de 4*7j = un mois => on voit apparaitre cycle annnuele et baisse de conso sur fin (confinement)
# sur le vent => moins de tendances


K <- 7*4
smooth_wp <- stats::filter(Data0$Wind_power, rep(1/K, K))
smooth_wind <- stats::filter(Data0$Wind, rep(1/K, K))
par(mfrow=c(1,1))
plot(Data0$Date, smooth_wp, type='l')
par(new=T)
plot(Data0$Date, smooth_wind, type='l', col='red', axes=F,xlab='',ylab='')
#plot(Data0$Temp%>%tail(1000), type='l', col='red', axes=F,xlab='',ylab='')
axis(side = 4,col='red', col.axis='red')
mtext(side = 4, line = 3, 'Wind', col='red')
legend("top",c("Wind power","Wind"),col=c("black","red"),lty=1,ncol=1,bty="n")

# noir prod éol lissé rouge vitesse vent
# prod éolienne croissance au cours des années 
# correlation forte entre les deux 
 # si on veut regresserr l'une en f° de l'autre il faut redresser la prod eol sinon on va pas avoir qqchose qui correspond au parc installé à la fin de nos données



############################################Solar
par(mfrow=c(1,1))
plot(Data0$Date, Data0$Nebulosity, type='l')
plot(Data0$Date, Data0$Nebulosity_weighted, type='l')

# solaire nébulosité => grosse variabilité au début puis stabilité => changement de qualité de la donnée au cour du temps 
# => lisser le passé le modifier/ la drop / considérer uniquement période 2018 2022

K <- 7*5
smooth_neb <- stats::filter(Data0$Nebulosity, rep(1/K, K))
plot(Data0$Date, smooth_neb, type='l')

# lissage on voit vraiment appraiatre cette tendance (prblm de tendance)

par(mfrow=c(3,1))
plot(Data0$Nebulosity, Data0$Load, pch=3,  col=col[1])
plot(Data0$Nebulosity, Data0$Solar_power, pch=3,  col=col[2])
plot(Data0$Nebulosity, Data0$Wind_power, pch=3,  col=col[3])

# nébulosité a un impacte sur conso électrique (eclairage) mais pas visible direct
# prod solaire très impacté, grande dispersion peut etre lié à mauavise qualité 
# vent variabilité naturelle (hiver)

sel <- which(year(Data0$Date)>=2018)
par(mfrow=c(3,1))
plot(Data0$Nebulosity[sel], Data0$Load[sel], pch=3,  col=col[1])
plot(Data0$Nebulosity[sel], Data0$Solar_power[sel], pch=3,  col=col[2])
plot(Data0$Nebulosity[sel], Data0$Wind_power[sel], pch=3,  col=col[3])

# deux régimes (été hiver)
# éolien il y a un pattern de variance
# solaire encore plus nette qu'avant

cor(Data0$Nebulosity, Data0$Solar_power)
cor(Data0$Nebulosity[sel], Data0$Solar_power[sel])
cor(Data0$Nebulosity_weighted[sel], Data0$Solar_power[sel])

# -0.24 avant 2018
# -057 après   => gros impact

############################################Lag
names(Data0)

plot(Data0$Net_demand.7, Data0$Net_demand, pch=3)
plot(Data0$Net_demand.1, Data0$Net_demand, pch=3)

cor(Data0$Net_demand.1, Data0$Net_demand)
cor(Data0$Net_demand.7, Data0$Net_demand)


############################################Holidays
boxplot(Net_demand~as.factor(Christmas_break), data=Data0[which(Data0$DLS==0),])
boxplot(Net_demand~Summer_break, data=Data0[which(Data0$DLS==1),])
boxplot(Net_demand~BH, data=Data0)


############################################DLS
boxplot(Load~DLS, data=Data0)

#########################################train/Test

# réflexe à avoir => utiliser les modèles pr faire des prévisions 
# importance => voir si distribution est la même sur le train et sur le test 
# surtout pr tt les variables météo 
# sur le test on a pas de température p^lus faible ou forte => on a pas a faire de prevsiions
# qui sont out of distributions / out of bond 
# très important ça donc c'est cool (changement climatique prevision temp extrêmes => nécessite de faire des adaptations dans le modèle)
# on voit pas de grosse différence de distribution => c'est cool 
# nébulosité même chose que temp
# on voit le gros shift avant après 2018
# pareil pr le vent

par(mfrow=c(1,2))
hist(Data0$Temp)
hist(Data1$Temp)

range(Data0$Temp)
range(Data1$Temp)

par(mfrow=c(1,1))
hist(Data0$Temp, xlim=range(Data0$Temp, Data1$Temp), col='lightblue', breaks=50, main='Temp')
par(new=T)
hist(Data1$Temp, xlim=range(Data0$Temp, Data1$Temp), col=adjustcolor('red', alpha.f=0.5), , breaks=50, main='')

par(mfrow=c(1,1))
hist(Data0$Nebulosity, xlim=range(Data0$Nebulosity, Data1$Nebulosity), col='lightblue', breaks=50, main='Neb')
par(new=T)
hist(Data1$Nebulosity, xlim=range(Data0$Nebulosity, Data1$Nebulosity), col=adjustcolor('red', alpha.f=0.5), , breaks=50, main='')

sel <- which(year(Data0$Date)>=2018)
par(mfrow=c(1,1))
hist(Data0$Nebulosity[sel], xlim=range(Data0$Nebulosity[sel], Data1$Nebulosity), col='lightblue', breaks=50, main='Neb')
par(new=T)
hist(Data1$Nebulosity, xlim=range(Data0$Nebulosity[sel], Data1$Nebulosity), col=adjustcolor('red', alpha.f=0.5), , breaks=50, main='')


sel <- which(year(Data0$Date)>=2018)
par(mfrow=c(1,1))
hist(Data0$Wind[sel], xlim=range(Data0$Wind, Data1$Wind), col='lightblue', breaks=50, main='Wind')
par(new=T)
hist(Data1$Wind, xlim=range(Data0$Wind, Data1$Wind), col=adjustcolor('red', alpha.f=0.5), , breaks=50, main='')



sel <- which(year(Data0$Date)>=2018)
par(mfrow=c(1,1))
hist(Data0$Nebulosity[sel], xlim=range(Data0$Nebulosity, Data1$Nebulosity), col='lightblue', breaks=50)
par(new=T)
hist(Data1$Nebulosity, xlim=range(Data0$Nebulosity, Data1$Nebulosity), col=adjustcolor('red', alpha.f=0.5), , breaks=50)










