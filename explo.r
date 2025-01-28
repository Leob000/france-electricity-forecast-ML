# %%
rm(list = objects())
graphics.off()
library(tidyverse)
library(lubridate)
library(forecast)


# install.packages("tidyverse")

source("R/score.R")
# Data0 <- read_delim("Data/train.csv", delim=",")
# Data1<- read_delim("Data/test.csv", delim=",")


X_train <- read_csv("Data/train.csv")
X_test <- read_csv("Data/test.csv")
X_train_orig <- X_train
X_test_orig <- X_test

# %%

# Load.1 le load décalé d'un jour
# Temp_s95 température avec lissage exponentiel
# Temp_S95_min minimum de la température lissée sur la journée
# Nébulosité, a quel point le ciel est opaque (100 = très nuageux)
# Nebulosity_weighted prend en compte les panneaux solaires
# toy = time of year
# weekdays = jour de la semaine (à modifier)
# BH bank holidays = jour férié ; before/after si before/after l'est aussi
# DLS Day light savings, changement d'heure
# summer break Mois d'août


summary(X_train)

range(X_train$Date)
range(X_test$Date)

# Attention, les weekdays sont en num et pas en ~classification
names(X_train)
head(X_train[, c("Date", "WeekDays")])
str(X_train)
weekdays(head(X_train$Date, 7))
class(X_train$WeekDays)

# 5: Saturday
# 6: Sunday
# 0: Monday
# ...

################################### analyse de la demande nette

############################################ trend
plot(X_train$Date, X_train$Net_demand, type = "l", xlim = range(X_train$Date, X_test$Date))
# Forte baisse pendant le confinement, sans redécoller derrière
# Changement été/hiver en moyenne mais aussi en variance

col <- yarrr::piratepal("basel")
par(mfrow = c(3, 1))
plot(X_train$Date, X_train$Load, type = "l", col = col[1])
plot(X_train$Date, X_train$Solar_power, type = "l", col = col[2])
plot(X_train$Date, X_train$Wind_power, type = "l", col = col[3])

par(mfrow = c(1, 1))
plot(X_train$Date, X_train$Load, type = "l", ylim = range(X_train$Solar_power, X_train$Load), col = col[1])
lines(X_train$Date, X_train$Wind_power, col = col[3])
lines(X_train$Date, X_train$Solar_power, col = col[2])

mean(X_train$Load)
mean(X_train$Wind_power)
mean(X_train$Solar_power)

par(mfrow = c(1, 1))
hist(X_train$Net_demand, breaks = 100)
# plot(Data0$Date, Data0$Temp, type='l')

# Convolution avec une période de 7, on "annule" le cycle hebdomadaire, on voit que la variance
# est bien moindre
plot(X_train$Date, X_train$Net_demand, type = "l", xlim = range(X_train$Date, X_test$Date))
K <- 7
smooth <- stats::filter(X_train$Net_demand, rep(1 / K, K))
lines(X_train$Date, smooth, col = "red", lwd = 2)


############################################ yearly cycle
sel <- which(X_train$Year == 2021)
plot(X_train$Date[sel], X_train$Net_demand[sel], type = "l")

plot(X_train$toy)

col.tr <- adjustcolor(col = "black", alpha = 0.3)
plot(X_train$toy, X_train$Net_demand, pch = 16, col = col.tr)

par(mfrow = c(1, 1))
plot(X_train$toy, X_train$Load, pch = 16, col = col.tr[1])

col.tr <- adjustcolor(col, alpha = 0.3)
par(mfrow = c(3, 1))
plot(X_train$toy, X_train$Load, pch = 16, col = col.tr[1])
plot(X_train$toy, X_train$Solar_power, pch = 16, col = col.tr[2])
plot(X_train$toy, X_train$Wind_power, pch = 16, col = col.tr[3])

par(mfrow = c(3, 1))
boxplot(Net_demand ~ Month, data = X_train, col = col[1])
boxplot(Solar_power ~ Month, data = X_train, col = col[2])
boxplot(Wind_power ~ Month, data = X_train, col = col[3])


############################################ Weekly cycle
par(mfrow = c(1, 1))

sel <- which(X_train$Month == 6 & X_train$Year == 2021)
plot(X_train$Date[sel], X_train$Net_demand[sel], type = "l")

par(mfrow = c(3, 1))
boxplot(Net_demand ~ WeekDays, data = X_train, col = col[1])
boxplot(Solar_power ~ WeekDays, data = X_train, col = col[2])
boxplot(Wind_power ~ WeekDays, data = X_train, col = col[3])

par(mfrow = c(1, 1))
boxplot(Net_demand ~ WeekDays, data = X_train)


# Transition weekend - semaine, semaine - weekend ...
plot(X_train$Load.1, X_train$Load)
cor(X_train$Load.1, X_train$Load)

# Auto corélogramme
par(mfrow = c(1, 3))
Acf(X_train$Load, lag.max = 7 * 10, type = c("correlation"), col = col[1], ylim = c(0, 1))
Acf(X_train$Solar_power, lag.max = 7 * 10, type = c("correlation"), col = col[2], ylim = c(0, 1))
Acf(X_train$Wind_power, lag.max = 7 * 10, type = c("correlation"), col = col[3], ylim = c(0, 1))


par(mfrow = c(1, 3))
Acf(X_train$Load, lag.max = 7 * 60, type = c("correlation"), col = col[1], ylim = c(-1, 1))
Acf(X_train$Solar_power, lag.max = 7 * 60, type = c("correlation"), col = col[2], ylim = c(-1, 1))
Acf(X_train$Wind_power, lag.max = 7 * 60, type = c("correlation"), col = col[3], ylim = c(-1, 1))


####################################################################################################################################
############################################ Meteo effect/covariates
####################################################################################################################################

############################################ Temperature
par(mar = c(5, 5, 2, 5))
par(mfrow = c(1, 1))
plot(X_train$Date, X_train$Net_demand, type = "l")
par(new = T)
plot(X_train$Date, X_train$Temp, type = "l", col = "red", axes = F, xlab = "", ylab = "")
# plot(Data0$Temp%>%tail(1000), type='l', col='red', axes=F,xlab='',ylab='')
axis(side = 4, col = "red", col.axis = "red")
mtext(side = 4, line = 3, "Temperature", col = "red")

legend("top", c("Net_demand", "Temperature"), col = c("black", "red"), lty = 1, ncol = 1, bty = "n")

col.tr <- adjustcolor(col = "black", alpha = 0.25)
plot(X_train$Temp, X_train$Net_demand, pch = 3, col = col.tr)


plot(X_train$Date %>% head(, n = 7 * 3), X_train$Temp %>% head(, n = 7 * 3), type = "l")
lines(X_train$Date %>% head(, n = 7 * 3), X_train$Temp_s95 %>% head(, n = 7 * 3), col = "blue")
lines(X_train$Date %>% head(, n = 7 * 3), X_train$Temp_s99 %>% head(, n = 7 * 3), col = "red")

plot(X_train$Date %>% head(, n = 7 * 5), X_train$Temp_s99 %>% head(, n = 7 * 5), type = "l")
lines(X_train$Date %>% head(, n = 7 * 5), X_train$Temp_s99_min %>% head(, n = 7 * 5), col = "blue")
lines(X_train$Date %>% head(, n = 7 * 5), X_train$Temp_s99_max %>% head(, n = 7 * 5), col = "red")

par(mfrow = c(1, 1))
col.tr1 <- adjustcolor(col = "black", alpha = 0.25)
col.tr2 <- adjustcolor(col = "red", alpha = 0.25)
plot(X_train$Temp, X_train$Net_demand, pch = 3, col = col.tr1)
points(X_train$Temp_s99, X_train$Net_demand, pch = 3, col = col.tr2)


col.tr <- adjustcolor(col, alpha = 0.25)
par(mfrow = c(3, 1))
plot(X_train$Temp, X_train$Load, pch = 3, col = col.tr[1])
plot(X_train$Temp, X_train$Solar_power, pch = 3, col = col.tr[2])
plot(X_train$Temp, X_train$Wind_power, pch = 3, col = col.tr[3])

############################################ Wind
par(mfrow = c(2, 1))
plot(X_train$Date, X_train$Wind, type = "l")
plot(X_train$Date, X_train$Wind_weighted, type = "l")


par(mfrow = c(3, 1))
plot(X_train$Wind, X_train$Load, pch = 3, col = col[1])
plot(X_train$Wind, X_train$Solar_power, pch = 3, col = col[2])
plot(X_train$Wind, X_train$Wind_power, pch = 3, col = col[3])

par(mfrow = c(1, 1))
plot(X_train$Wind, X_train$Wind_power, pch = 3, col = col[3])
points(X_train$Wind_weighted, X_train$Wind_power, pch = 3, col = col[4])



par(mfrow = c(1, 1))
plot(X_train$Date, X_train$Net_demand, type = "l")
par(new = T)
plot(X_train$Date, X_train$Wind, type = "l", col = "red", axes = F, xlab = "", ylab = "")
# plot(Data0$Temp%>%tail(1000), type='l', col='red', axes=F,xlab='',ylab='')
axis(side = 4, col = "red", col.axis = "red")
mtext(side = 4, line = 3, "Wind", col = "red")
legend("top", c("Net_demand", "Wind"), col = c("black", "red"), lty = 1, ncol = 1, bty = "n")


K <- 7 * 4
smooth_net <- stats::filter(X_train$Net_demand, rep(1 / K, K))
smooth_wind <- stats::filter(X_train$Wind, rep(1 / K, K))
par(mfrow = c(1, 1))
plot(X_train$Date, smooth_net, type = "l")
par(new = T)
plot(X_train$Date, smooth_wind, type = "l", col = "red", axes = F, xlab = "", ylab = "")
# plot(Data0$Temp%>%tail(1000), type='l', col='red', axes=F,xlab='',ylab='')
axis(side = 4, col = "red", col.axis = "red")
mtext(side = 4, line = 3, "Wind", col = "red")
legend("top", c("Net_demand", "Wind"), col = c("black", "red"), lty = 1, ncol = 1, bty = "n")



K <- 7 * 4
smooth_wp <- stats::filter(X_train$Wind_power, rep(1 / K, K))
smooth_wind <- stats::filter(X_train$Wind, rep(1 / K, K))
par(mfrow = c(1, 1))
plot(X_train$Date, smooth_wp, type = "l")
par(new = T)
plot(X_train$Date, smooth_wind, type = "l", col = "red", axes = F, xlab = "", ylab = "")
# plot(Data0$Temp%>%tail(1000), type='l', col='red', axes=F,xlab='',ylab='')
axis(side = 4, col = "red", col.axis = "red")
mtext(side = 4, line = 3, "Wind", col = "red")
legend("top", c("Wind power", "Wind"), col = c("black", "red"), lty = 1, ncol = 1, bty = "n")





############################################ Solar
# Attention, variable à corriger car la nébulosité n'a pas été bien calculée auparavant
par(mfrow = c(1, 1))
plot(X_train$Date, X_train$Nebulosity, type = "l")
plot(X_train$Date, X_train$Nebulosity_weighted, type = "l")

K <- 7 * 5
smooth_neb <- stats::filter(X_train$Nebulosity, rep(1 / K, K))
plot(X_train$Date, smooth_neb, type = "l")

par(mfrow = c(3, 1))
plot(X_train$Nebulosity, X_train$Load, pch = 3, col = col[1])
plot(X_train$Nebulosity, X_train$Solar_power, pch = 3, col = col[2])
plot(X_train$Nebulosity, X_train$Wind_power, pch = 3, col = col[3])

sel <- which(year(X_train$Date) >= 2018)
par(mfrow = c(3, 1))
plot(X_train$Nebulosity[sel], X_train$Load[sel], pch = 3, col = col[1])
plot(X_train$Nebulosity[sel], X_train$Solar_power[sel], pch = 3, col = col[2])
plot(X_train$Nebulosity[sel], X_train$Wind_power[sel], pch = 3, col = col[3])

cor(X_train$Nebulosity, X_train$Solar_power)
cor(X_train$Nebulosity[sel], X_train$Solar_power[sel])
cor(X_train$Nebulosity_weighted[sel], X_train$Solar_power[sel])



############################################ Lag
names(X_train)

plot(X_train$Net_demand.7, X_train$Net_demand, pch = 3)
plot(X_train$Net_demand.1, X_train$Net_demand, pch = 3)

cor(X_train$Net_demand.1, X_train$Net_demand)
cor(X_train$Net_demand.7, X_train$Net_demand)


############################################ Holidays
boxplot(Net_demand ~ as.factor(Christmas_break), data = X_train[which(X_train$DLS == 0), ])
boxplot(Net_demand ~ Summer_break, data = X_train[which(X_train$DLS == 1), ])
boxplot(Net_demand ~ BH, data = X_train)


############################################ DLS
boxplot(Load ~ DLS, data = X_train)

######################################### train/Test

par(mfrow = c(1, 2))
hist(X_train$Temp)
hist(X_test$Temp)

range(X_train$Temp)
range(X_test$Temp)

par(mfrow = c(1, 1))
hist(X_train$Temp, xlim = range(X_train$Temp, X_test$Temp), col = "lightblue", breaks = 50, main = "Temp")
par(new = T)
hist(X_test$Temp, xlim = range(X_train$Temp, X_test$Temp), col = adjustcolor("red", alpha.f = 0.5), , breaks = 50, main = "")

par(mfrow = c(1, 1))
hist(X_train$Nebulosity, xlim = range(X_train$Nebulosity, X_test$Nebulosity), col = "lightblue", breaks = 50, main = "Neb")
par(new = T)
hist(X_test$Nebulosity, xlim = range(X_train$Nebulosity, X_test$Nebulosity), col = adjustcolor("red", alpha.f = 0.5), , breaks = 50, main = "")

sel <- which(year(X_train$Date) >= 2018)
par(mfrow = c(1, 1))
hist(X_train$Nebulosity[sel], xlim = range(X_train$Nebulosity[sel], X_test$Nebulosity), col = "lightblue", breaks = 50, main = "Neb")
par(new = T)
hist(X_test$Nebulosity, xlim = range(X_train$Nebulosity[sel], X_test$Nebulosity), col = adjustcolor("red", alpha.f = 0.5), , breaks = 50, main = "")


sel <- which(year(X_train$Date) >= 2018)
par(mfrow = c(1, 1))
hist(X_train$Wind[sel], xlim = range(X_train$Wind, X_test$Wind), col = "lightblue", breaks = 50, main = "Wind")
par(new = T)
hist(X_test$Wind, xlim = range(X_train$Wind, X_test$Wind), col = adjustcolor("red", alpha.f = 0.5), , breaks = 50, main = "")



sel <- which(year(X_train$Date) >= 2018)
par(mfrow = c(1, 1))
hist(X_train$Nebulosity[sel], xlim = range(X_train$Nebulosity, X_test$Nebulosity), col = "lightblue", breaks = 50)
par(new = T)
hist(X_test$Nebulosity, xlim = range(X_train$Nebulosity, X_test$Nebulosity), col = adjustcolor("red", alpha.f = 0.5), , breaks = 50)
