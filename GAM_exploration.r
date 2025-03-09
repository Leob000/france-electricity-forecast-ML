rm(list=objects())
###############packages
library(mgcv)
library(yarrr)
library(magrittr)
library(forecast)
library(tidyverse)
source('score.R')
#setwd("C:/Users/Mrgan Scalabrino/Desktop/modelisation_predictive/yannig data challenge")

df_full=read_csv("Data/treated_data.csv")

# Définir les périodes à extraire
range_test=c("2022-09-02", "2023-10-01")
range_val=c("2021-09-01","2022-09-01")
range_train=c("2013-03-02", "2021-10-31")
# Filtrer le dataset pour les période spécifiée
df_test <- df_full %>%
  filter(Date >= range_test[1] & Date <= range_test[2])
df_val <- df_full %>%
  filter(Date >= range_val[1] & Date <= range_val[2])
df_train <- df_full %>%
  filter(Date >= range_train[1] & Date <= range_train[2])
df_train_val <- df_full %>%
  filter(Date >= range_train[1] & Date <= range_val[2])


# validation croisé temporelle par bloc consecutifs 
Nblock<-10
borne_block<-seq(1, nrow(df_train), length=Nblock+1)%>%floor
block_list<-list()
l<-length(borne_block)
for(i in c(2:(l-1)))
{
  block_list[[i-1]] <- c(borne_block[i-1]:(borne_block[i]-1))
}
block_list[[l-1]]<-c(borne_block[l-1]:(borne_block[l]))

blockRMSE<-function(equation, block)
{
  g<- gam(as.formula(equation), data=df_train[-block,])
  forecast<-predict(g, newdata=df_train[block,])
  return(df_train[block,]$Net_demand-forecast)
} 

###########################################################################################################################
########################################
############# MODEL
equation_complète <- Load~s(as.numeric(Date),k=3, bs='cr') + s(toy,k=30, bs='cc') + s(Temp,k=10, bs='cr') 
                + s(Load.1, bs='cr', by= as.factor(WeekDays))+ s(Load.7, bs='cr') +
                  s(Temp_s99,k=10, bs='cr') + as.factor(WeekDays) +BH  + BH_before + 
                  BH_after+ s(Temp_s95) +ti(Nebulosity_weighted_ratio) + ti(Wind_weighted_ratio) +
                  ti(Wind_weighted, Temp) + te(Temp_s95_min,Temp_s99_min)+ te(Temp_s95_max,Temp_s99_max)
                  +s(Wind, bs='cr') + s(Nebulosity,bs="cr") + DLS + Summer_break + Christmas_break
                  +Holiday_zone_c+Holiday_zone_b+Holiday_zone_a+BH_Holiday +s(Solar_power.1, bs='cr')
                  +s(Solar_power.7, bs='cr')+s(Wind_power.1, bs='cr')+s(Wind_power.7, bs='cr')
                  +s(y_dayofyear,bs='cr')+s(x_dayofyear,bs='cr')+s(y_dayofweek,bs='cr')+s(x_dayofweek,bs='cr')
                  +lundi_vendredi +flag_temp

equation_gary= Net_demand~s(as.numeric(Time),k=3, bs='cr') +s(toy,k=30,bs='cc')+ WeekDays + BH_before + BH_holiday+
  Holiday+stringency+ economic support + s(Load.1,bs='cr',by=WeekDays) +s(Load.7,bs='cr')+s(Wind_power.1,bs='cr')
  + s(Solar_power.1,bs='cr')+ ti(windweigh) +s(nebulo, by=year)+ ti(temp)+ti(temp,temps99max)+ti(windweigted,temp)+s(Temp,k=10, bs='cr') 

# current euation
equation <- Net_demand~s(as.numeric(Date),k=3, bs='cr')+ BH_Holiday +s(toy,k=30, bs='cc') + s(Temp,k=10, bs='cr') +s(Net_demand.1, bs='cr', by= as.factor(WeekDays))+ s(Load.1, bs='cr') + s(Load.7, bs='cr') + s(Temp_s99,k=10, bs='cr') + as.factor(WeekDays) + BH + s(Wind, bs='cr') + s(Net_demand.1,bs="cr") +ti(as.numeric(Date),Nebulosity,k=c(4,10)) + te(y_dayofyear,x_dayofyear,bs=c('cr','cr'))+s(Nebulosity,k=7,bs='cr')

# pas de toy = remplaçage par y x day of year => ils sont significatifs 
# le moins significatif est Netdemand.7, on a tjrs des cycles, le covid et des extrêmes
# fréquence extrêmement élevé autour de 0, queue valeur extrême tjrs un peu
# effets jours semaine OK, tjrs différence pr BH   # rmse val = 1552
# + ajout relation xy dayofyear => perte signif pr isolé 1509=> ti 1507 les deux isolés sont inutiles => mais plus économe en df
# RMSE bloc = 1318, sans les deux avec juste le lien on a 1332 et moins de ddl sur le lien (1352 sans rien sur dayofyear)
# ajout de load1/weekdays pr supprimer definitivement la dépendance => 1338 + perte significativité net demand7
# essai pareil avec BH plutôt perte significativité load1 netdem7 et pas significatif BH0, BH linéaire reste signif CV =1339
# les 2 1353, avec penalisation on garde que samedi et lundi mais on garde les 2 BH et Netdemand7
# 1332 avec netdemand1/weekdays (tjrs effet covid, net demand plus significatif )
# 1307 +ti(asnum date,nebulosity 4 10) 
# net demand 7/ load7 ou by weekdays améliore pas 1312 => pas significatif on le vire (mais edf =3 ?) => 1298 = 156pinball
# on ajoute nebulo seul => 1298, significatif edf=8, te reste signif aussi etedf=8 aussi 155.47 pinball
# on met les deux en ti => 1282  155.74 
# on met date en ti aussi change que dale apprait meme pas dans param
# changement te xydayofyear avec ti => 1286 157 edf 2->1.3 rjs signif
# rajout des deux isolé en ti (souvent pas nécessaire askip) => 1285 156.9  ti isolés edf e-6 -7 nons significatifs 
# on garde te pr dayofyear trigo ducoup, on regarde mtn de passer le nebulosity seul en s pas de changelment
# puis le nebulo date en te => 155.6 1299 1448 => ti meilleur
# ajout toy => 160 1258 1453 k=30 28 ddl vient taper
# k=40 160 1258 1446 ddl38 vient taper encore => on va revenir à 30 je crois, pq pas essayer 50
# k=7 nebulosity au lieu 5 pr voir si jamais tapait, ajout log nebulo sans k => change que dale edf log neb e-4
# nebulosity **2 temps de calcul trop long bordel => 160 1261 1451   edf 3 significatif, ressère un peu l'effet autour de 0 mais le supprime pas du tt
# test avec toy sans ti xy dayofyear => 159 1272 1470   on va le garder= tester avec anova la diff 
  
# tester interaction nebu weighted nebu distributions des residus opposés => compensation ?
  # => assez long a tourner chiant 160 1263 1454  ressère à peine   pas très significatif 4edf

# interactions de mistral pour supprimer l'effet restant de nebulosity :
#
# ti(Wind, Nebulosity) +  # Interaction entre le vent et la nébulosité
# 160 1261 1453 pas d'amélioration de nebulosity 0.5 edf pas significatif 
# ti(Temp, Nebulosity) +  # Interaction entre la température et la nébulosité
# 158 1259 1442 significatif 6 edf  tjrs effet
# ti(Temp, Wind) +  # Interaction entre la température et le vent
# 
# ti(Temp_s99, Temp_s95) +  # Interaction entre les températures lissées
# 
# ti(WeekDays, toy)
#   
# te(Nebulosity,Temp,Wind) # interaction super complexe pr éliminer toute relation survivante
# très long à tourner chiant 156 1252 1440 40edf 116ref_df significatif corrige à peine l'effet
  
# SINGLE VARIABLE TO ADD :
# +BH_after 
#  158 1246 1438 significatif 
# +BH_before 
# 160 1260 1449 significatif 
# +BH_holiday !!!!
#  127 1214 1374 significatif baisse significativité de BH à e-5
# réduit massivement la différence entre 0 1 BH sur residus (tjrs bcp de valeurs extrêmes de résidus sur 0)=> autre cause que BH ? = holiday ?
# +flag_temp
# 127 1211 1374 non significatif 
# +Holiday
# 126 1206 1360 significatif e-9
# +s(Temp_s95,bs='cr')
#
# +s(Temp_s9599_minmax,bs='cr')
#
# +s(Solar_power.1 7,bs='cr')
#
# +s(Wind_power.1 7,bs='cr')
#
# +s(Wind_weighted_ratio,bs='cr')
#
# +s(Nebulosity_weighted_ratio,bs='cr')
# 
 
  
### on prend comme base avant d'essayer de rajouter des variables le modèle final net demand du prof => performance sans toy à 486    180 pinball
# (date,toy,temp,load1,load7,temps99,weekdays,BH,wind,netdem1,netdem7,te(asnumdate-nebulo))          


########## MODEL CODE
Block_residuals<-lapply(block_list, blockRMSE, equation=equation)%>%unlist
rmseBloc1<-rmse.old(Block_residuals)

GAM<-gam(equation, data=df_train,select=TRUE)
#sqrt(GAM$gcv.ubre) # racine de gcv  ~RMSE (trop optimiste)
summary(GAM)

GAM.forecast<-predict(GAM,  newdata= df_val)
rmse1.forecast <- rmse.old(df_val$Net_demand-GAM.forecast)

GAM.forecast_train_val<-predict(GAM,  newdata= df_train_val)
res_train_val=df_train_val$Net_demand-GAM.forecast_train_val


res <- df_val$Net_demand - GAM.forecast
quant <- qnorm(0.95, mean= mean(res), sd= sd(res))
pinball_loss(y=df_val$Net_demand,GAM.forecast+quant, quant=0.95, output.vect=FALSE)
rmseBloc1
rmse1.forecast

### plots d'analyse des résidus 

boxplot(Block_residuals)

plot(df_train$Date, Block_residuals, type='l')

hist(Block_residuals,breaks = 50) # doit être gaussien si possible ,breaks=20

boxplot(Block_residuals~df_train$WeekDays) # verifier cycle hebdomadaire bien modélisé ( et BH et variable catégorielles)*

boxplot(Block_residuals~df_train$BH) # verifier cycle hebdomadaire bien modélisé ( et BH et variable catégorielles)

boxplot(Block_residuals~df_train$flag_temp)

# verifier variables numerique bien modélisé en changeant variable ici
plot(df_train$Temp_s95, Block_residuals, pch=16) 

plot(df_train$Date, df_train$toy, type='l')


######
modèle tendance,temp,netdemandbyweekdays,load1,load7,temps99,weekdays,bh,wind,netdemand1,texydayofyear,tedatenebulo,nebulo
 1282 cv loss 155.7 pinball 421 public loss
plots résidus / variables :
temp : forme patate allongé avec bruit autour (extrêmes => BH)
Netdemand1 by weekdays, variabilité plus importante valeur basse, patate allongé (couché+ longue)
load1 pareil
load7 moins tranché diff variabilité 
temps99 2 régimes de bruits gaussiens élevé et bas qq extrêmes
weekdays extrêmes différents selon jours, qq diff selon les jours mais petit
BH totale difference, variabilité nettement plus importante pr 0, valeurs extrêmes tjrs en haut pr 1, en haut en bas pr 0, variabilité plus importante pr 1
wind bruit blanc avec dispersion réduites pr fortes valeurs 
netdemand1 boudin avec plus forte dispersion aux faibles valeurs de demande (2 régimes ???)
xy dayofyear x=sablier couché avec plein bruit autour  y pareil moins de bruits 
=> c'est à dire qu'on fait moins d'erreur en été en fait
Nebulosity : sorte de cone, pointe vers basse valeur nebulo '

variables non utilisés : 
BH_after Bh_before => pareil
Bh_holiday 3!=0 similaire jsp quoi dire 
Holiday similaire mais variabilité plus vers le bas et vers le haut des valeurs extrêmes pr 0 1
holidays zones pareil quasiment
prod sol lagué cone pointant vers valeurs hautes  wind plus proche de bruit gaussien 
temp s95 bruit gaussien 2 régimes, min max et minmax 99 pareil quasiment => été hiver 
seuil 15° variabilité plus élevé au dessus, distributions valeurs extrêmes différentes 
wind nebu ratio weighted  wind quasi bruit blanc, nebu bruit blanc avec pointe qui en part vers valeurs hautes(cône vite fait)



#####model 7 variante ti
# interaction temperature brute et lissé (inertie batiments)
eq=eq+te(Temp_s99, Temp, bs=c('cr','cr'), k=c(10,10))
Block_forecast<-lapply(block_list, blockRMSE, equation=equation)%>%unlist
Block_residuals <- df_train_val$Load-Block_forecast
rmse_te <- rmse(df_train_val$Load, Block_forecast)
rmse_te
# terme interaction significatif terme lissé 99 n'est plus significatif => interaction contenue tt entière dans 99 ?
# prevision 20megawatt mieux mais en ajoutant 16 ddl pas fou

# on fait bien le ti avec ti temp ti temp lissé ti interaction
eq+=ti(Temp,k=10, bs='cr')+ ti(Temp_s99,k=10, bs='cr') +ti(Temp_s99, Temp, bs=c('cr','cr'), k=c(10,10))
# effet marginal temp existe, pas d'effet marginal de temp lissé 
# effet interaction significatif confirme ce qu'on a vu ci dessus c'est l'interaction qui est importante (plus que temp_s99 qu'on peut larguer)

############Anova selection (testRV)
# test modèle 1 load1 modèle 2 load1 by weekdays
anova(gam6, gam6_by, test = "Chisq") ####p value <0.05 interaction is significant
# on rejette hypothèse nullité effet autoregressif par type de jour sont significatifs 
# equations différentes :
#   + WeekDays au lieu asfactor
#   + te(Temp_s95_max, Temp_s99_max) + Summer_break  + Christmas_break + te(Temp_s95_min, Temp_s99_min)
# alors sur validation perf moins bonne du modèle 2, mais anova dit modèle meilleur
# 
# autre essai 2 modèle simple ti(load1 7 bs=cr) et test avec sans ti(load1,load7)
# significatif garder l'interaction => à tester sur modèle complexe 

########shrinkage approach
# créer du bruit et l'ajouter en tant que variable pr voir si le modèle le dégage => dégage = edf de 1 plus non significativité
# coeff =1 => pénalité infinie 
# s'implémente avec bs='ts' ou 'cs'

# + terme nebulosité weighted / interaction temperature vent pour temperature ressentie
# + svar1= ajout de bruit dans les données comment le modèle réagit => robustesse 
# si modèle trouve bruit significatif = problème

# dans gam ajouter select=TRUE, gamma=1.5 pour double penalty shrinkage plus couteux mais plus efficace
# If select is TRUE then gam can add an extra penalty to each term so that it can be 
# penalized to zero. This means that the smoothing parameter estimation that is part 
# of fitting can completely remove terms from the model. If the corresponding smoothing 
# parameter is estimated as zero then the extra penalty has no effect. Use gamma to increase level of penalization.

#### ARIMA sur résidus pr mieux prédire 
Block_residuals.ts <- ts(Block_residuals, frequency=7)
# technique de selection de modèle automatique selon critère précisé ici AIC=>prend le AIC le plus eptit possible
fit.arima.res <- auto.arima(Block_residuals.ts,max.p=3,max.q=4, max.P=2, max.Q=2, trace=T,ic="aic", method="CSS")
# prévision avec modèle autoregresif => soit boucle à la main soit technique avec refit param fixé = génère prediction à horizon 1
ts_res_forecast <- ts(c(Block_residuals.ts, df_val$Net_demand-GAM.forecast),  frequency= 7)
# residus apprentissage et partie que je veux prevoir (selb)
refit <- Arima(ts_res_forecast, model=fit.arima.res) 
# réapprend sur tt ça en fixant les paramètres pr pas surapprendre sur le test
prevARIMA.res <- tail(refit$fitted, nrow(df_val))
rmse6.arima <- rmse.old(fit.arima.res$residuals)
# 1100gam->979arima
# prevision finale 
gam6.arima.forecast <- gam6.forecast + prevARIMA.res
rmse.old(df_val$Load-gam6.forecast)
rmse.old(df_val$Load-gam6.arima.forecast)
mape(df_val$Load, gam6.arima.forecast)
rmse6.arima.forecast <- rmse.old(df_val$Load-gam6.arima.forecast)

### reste partie online learning avec kalman et tt 

####### Choix des variables à mettre dans equation 
# [1] "Date"                      "Load.1"                    "Load.7"                    "Net_demand"               
# [5] "Temp"                      "Temp_s95"                  "Temp_s99"                  "Temp_s95_min"             
# [9] "Temp_s95_max"              "Temp_s99_min"              "Temp_s99_max"              "Wind"                     
# [13] "Wind_weighted_ratio"       "Nebulosity"                "Nebulosity_weighted_ratio" "WeekDays"                 
# [17] "BH_before"                 "BH"                        "BH_after"                  "Year"                     
# [21] "DLS"                       "Summer_break"              "Christmas_break"           "Holiday"                  
# [25] "Holiday_zone_a"            "Holiday_zone_b"            "Holiday_zone_c"            "BH_Holiday"               
# [29] "Solar_power.1"             "Solar_power.7"             "Wind_power.1"              "Wind_power.7"             
# [33] "Net_demand.1"              "Net_demand.7"              "Net_demand.1_trend"        "x_dayofyear"              
# [37] "y_dayofyear"               "x_dayofweek"               "y_dayofweek"               "lundi_vendredi"           
# [41] "flag_temp" 

#### modèle 6 load
# (date,toy,temp,load1,load7,temps99,weekdays,BH)
#### modèle final net demand du prof => performance sans toy à 486
# (date,toy,temp,load1,load7,temps99,weekdays,BH,wind,netdem1,netdem7,te(asnumdate-nebulo))


