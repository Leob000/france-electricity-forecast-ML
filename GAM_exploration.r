rm(list=objects())
###############packages
library(mgcv)
library(yarrr)
library(magrittr)
library(forecast)
library(tidyverse)
source('R/score.R')

df_full=read_csv("Data/treated_data.csv")

# Normalisation de Net_demand et laggées, en conservant la moyenne et l'écart type pour faire la transformation inverse sur les prédictions
idx_train_val <- which(df_full$Date <= "2022-09-01")
df_train_val <- df_full[idx_train_val, ]
Net_demand_mean <- mean(df_train_val$Net_demand)
Net_demand_sd <- sd(df_train_val$Net_demand)

df_full$Net_demand <- (df_full$Net_demand - Net_demand_mean) / Net_demand_sd
df_full$Net_demand.1 <- (df_full$Net_demand.1 - Net_demand_mean) / Net_demand_sd
df_full$Net_demand.7 <- (df_full$Net_demand.7 - Net_demand_mean) / Net_demand_sd


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

### on prend comme base avant d'essayer de rajouter des variables le modèle final net demand du prof => performance sans toy à 486    180 pinball
# (date,toy,temp,load1,load7,temps99,weekdays,BH,wind,netdem1,netdem7,te(asnumdate-nebulo))          

equation <- Net_demand~s(as.numeric(Date),k=3, bs='cr')+ BH_Holiday +s(toy,k=30, bs='cc') + s(Temp,k=10, bs='cr') +s(Net_demand.1, bs='cr', by= as.factor(WeekDays))+ s(Load.1, bs='cr') + s(Load.7, bs='cr') + s(Temp_s99,k=10, bs='cr') + as.factor(WeekDays) + BH + s(Wind, bs='cr') + s(Net_demand.1,bs="cr") +ti(as.numeric(Date),Nebulosity,k=c(4,10)) + te(y_dayofyear,x_dayofyear,bs=c('cr','cr'))+s(Nebulosity,k=7,bs='cr')

  
########## MODEL CODE
Block_residuals<-lapply(block_list, blockRMSE, equation=equation)%>%unlist
rmseBloc1<-rmse.old(Block_residuals)

GAM<-gam(equation, data=df_train,select=TRUE)
summary(GAM)

GAM.forecast<-predict(GAM,  newdata= df_val)
rmse1.forecast <- rmse.old(df_val$Net_demand-GAM.forecast)

res <- df_val$Net_demand - GAM.forecast
quant <- qnorm(0.95, mean= mean(res), sd= sd(res))
pinball_loss(y=df_val$Net_demand,GAM.forecast+quant, quant=0.95, output.vect=FALSE)
rmseBloc1
rmse1.forecast


### plots d'analyse des résidus 

boxplot(Block_residuals)

plot(df_train$Date, Block_residuals, type='l')

hist(Block_residuals,breaks = 50) # doit être gaussien si possible

boxplot(Block_residuals~df_train$WeekDays) # verifier cycle hebdomadaire bien modélisé 

boxplot(Block_residuals~df_train$BH) # (et BH et variable catégorielles)

# verifier variables numerique bien modélisé en changeant variable ici
plot(df_train$Temp_s95, Block_residuals, pch=16) 

# soit on voit un pattern a l'oeil soit on les regresse encore  :
g_prov <- gam(g0$residuals~ s(df_train$Temp, k=5, bs="cr"))  # gam résidus~variable ou on veut 
#savoir si elle est entièrement expliqué par notre modèle
summary(g_prov) # significativité 

anova(GAM1, GAM2, test = "Chisq") ####p value <0.05 interaction is significant

Acf(Block_residuals)
# autocorrelogramme des residus par blocs


range_test=c("2022-09-02", "2023-10-01")
df_test <- df_full %>%
  filter(Date >= range_test[1] & Date <= range_test[2])
### GAM regression rectification 

GAM.forecast_train_val<-(predict(GAM,  newdata= df_train_val) * Net_demand_sd) + Net_demand_mean
df_train_val_net_dem_renorm=(df_train_val$Net_demand * Net_demand_sd) + Net_demand_mean
res_train_val=GAM.forecast_train_val-df_train_val_net_dem_renorm
df_train_val$residus=res_train_val

equation_res= residus~ Holiday+s(Solar_power.1,k=10,bs='cr')+ s(Net_demand.7,k=10,bs='cr')+s(Wind_power.1,k=10,bs='cr')+s(Wind_power.7,k=10,bs='cr')+s(Solar_power.7,k=10,bs='cr')+s(Net_demand.7,k=10,bs='cr')+s(Net_demand.1_trend,k=10,bs='cr')+s(Nebulosity_weighted_ratio,k=7,bs='cr')
GAM<-gam(equation, data=df_train_val,select=TRUE)
GAM.forecast<-(predict(GAM,  newdata= df_test) * Net_demand_sd) + Net_demand_mean
GAM_res=gam(equation_res, data=df_train_val,select=TRUE)
GAM_res.forecast<-predict(GAM_res,  newdata= df_test)

final_result_val=GAM.forecast+GAM_res.forecast



### KALMAN ONLINE LEARNING

equation <- Net_demand~s(as.numeric(Date),k=3, bs='cr')+s(Temp_s99_max,k=10,bs='cr') +BH_Holiday+s(Wind_weighted_ratio,k=3,bs='cr') +s(toy,k=30, bs='cc') + s(Temp,k=10, bs='cr') +s(Net_demand.1, bs='cr',k=12, by= as.factor(WeekDays))+ s(Load.1, bs='cr',k=10) + s(Load.7,k=10, bs='cr') + s(Temp_s99,k=10, bs='cr') + as.factor(WeekDays) + BH + s(Wind, bs='cr',k=7) + s(Net_demand.1,bs="cr",k=12) +ti(as.numeric(Date),Nebulosity,k=c(4,10)) + te(y_dayofyear,x_dayofyear,bs=c('cr','cr'))+s(Nebulosity,k=7,bs='cr')
gam_net3 <- gam(equation,  data=df_train_val,select=TRUE)

sqrt(gam_net3$gcv.ubre)
gam_net3.forecast<-(predict(gam_net3,  newdata= df_test)* Net_demand_sd) + Net_demand_mean

X <- (predict(gam_net3, newdata=df_full, type='terms')* Net_demand_sd) + Net_demand_mean
###scaling columns
for (j in 1:ncol(X))
  X[,j] <- (X[,j]-mean(X[,j])) / sd(X[,j])
X <- cbind(X,1)
d <- ncol(X)

y <- (df_full$Net_demand* Net_demand_sd) + Net_demand_mean



# static 
ssm <- viking::statespace(X, y)
gam_net3.kalman.static <- ssm$pred_mean%>%tail(length(df_test$Date))
sig <- ssm$pred_sd%>%tail(length(df_test$Date))*sd(df_train_val$Net_demand-ssm$pred_mean%>%head(length(df_train_val$Date)))
quant <- qnorm(0.8, mean= gam_net3.kalman.static, sd= sig)
pinball_loss(y=df_val_net_dem_renorm, quant, quant=0.8, output.vect=FALSE)


# using expectation-maximization (30min/1000iter)
ssm_em <- viking::select_Kalman_variances(ssm, X%>%head(length(df_train_val$Date)), y%>%head(length(df_train_val$Date)), method = 'em', n_iter = 2000,
                                          Q_init = diag(d), verbose = 10, mode_diag = T)

# or using grid search (2h)
ssm_em <- viking::select_Kalman_variances(ssm_q, X%>%head(length(df_train_val$Date)), y%>%head(length(df_train_val$Date)), q_list = 2^(-30:0), p1 = 1)


ssm_em <- predict(ssm_em, X, y, type='model', compute_smooth = TRUE)
saveRDS(ssm_em, "Data/ssm_em_net_demand.RDS")
ssm_em <-readRDS("Data/ssm_em_net_demand.RDS")
gam_net3.kalman.Dyn.em <- ssm_em$pred_mean%>%tail(length(df_test$Date))
ssm_em$kalman_params$Q

ssm <- viking::statespace(X, y)
gam_net3.kalman.static_train <- ssm$pred_mean%>%head(length(df_train_val$Date))

submit <- read_delim( file="Data/sample_submission.csv", delim=",")
submit$Net_demand <- gam_net3.kalman.static
write.table(submit, file="Data/pred_test.csv", quote=F, sep=",", dec='.',row.names = F)

df_pred=df_train_val
df_pred$Net_demand=gam_net3.kalman.static_train
write.table(df_pred, file="Data/pred_train_val.csv", quote=F, sep=",", dec='.',row.names = F)



