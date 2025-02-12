rm(list = objects())
source("R/score.R")
set.seed(42)
library(tidyverse)
GRAPH <- FALSE

raw_train <- read_csv("Data/train.csv")
raw_test <- read_csv("Data/test.csv")

# Création du set de validation
train_indices <- sample(seq_len(nrow(raw_train)), size = 0.8 * nrow(raw_train))
X_train <- raw_train[train_indices, ]
X_val <- raw_train[-train_indices, ]

head(raw_train$Net_demand)
head(raw_train$Net_demand.1)

axe_x <- c()
axe_y <- c()
for (i in seq(0.95, 1.2, by = 0.001)) {
  res <- pinball_loss(raw_train$Net_demand, raw_train$Net_demand.1 * i, quant = 0.8)
  if (res < min(axe_y)) {
    best <- c(i, res)
  }
  axe_x <- c(axe_x, i)
  axe_y <- c(axe_y, res)
}
plot(axe_x, axe_y)
print(best)
