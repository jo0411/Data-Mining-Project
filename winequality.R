# Performance evaluation function for regression --------------------------
perf_eval_reg <- function(tgt_y, pre_y){
  
  # RMSE
  rmse <- sqrt(mean((tgt_y - pre_y)^2))
  # MAE
  mae <- mean(abs(tgt_y - pre_y))
  # MAPE
  mape <- 100*mean(abs((tgt_y - pre_y)/tgt_y))
  
  return(c(rmse, mae, mape))
  
}

# Performance Evaluation Function for classification-----------------------------------------
perf_eval_cla <- function(cm){
  
  # True positive rate: TPR (Recall)
  TPR <- cm[2,2]/sum(cm[2,])
  # Precision
  PRE <- cm[2,2]/sum(cm[,2])
  # True negative rate: TNR
  TNR <- cm[1,1]/sum(cm[1,])
  # Simple Accuracy
  ACC <- (cm[1,1]+cm[2,2])/sum(cm)
  # Balanced Correction Rate
  BCR <- sqrt(TPR*TNR)
  # F1-Measure
  F1 <- 2*TPR*PRE/(TPR+PRE)
  
  return(c(TPR, PRE, TNR, ACC, BCR, F1))
}

# Initialize the performance matrix
perf_mat_logistic <- matrix(0, 1, 6)
colnames(perf_mat_logistic) <- c("TPR (Recall)", "Precision", "TNR", "ACC", "BCR", "F1")
rownames(perf_mat_logistic) <- "Logstic Regression"

# DATA EDA
par(mfrow = c(3,4))
hist(data$fixed.acidity)
hist(data$volatile.acidity)
hist(data$citric.acid)
hist(data$residual.sugar)
hist(data$chlorides)
hist(data$free.sulfur.dioxide)
hist(data$total.sulfur.dioxide)
hist(data$density)
hist(data$pH)
hist(data$sulphates)
hist(data$alcohol)
hist(data$quality)
qqnorm(data$fixed.acidity)
qqnorm(data$volatile.acidity)
qqnorm(data$citric.acid)
qqnorm(data$residual.sugar)
qqnorm(data$chlorides)
qqnorm(data$free.sulfur.dioxide)
qqnorm(data$total.sulfur.dioxide)
qqnorm(data$density)
qqnorm(data$pH)
qqnorm(data$sulphates)
qqnorm(data$alcohol)
qqnorm(data$quality)
dev.off()

cor(wine_x)


#------------------------------------------------------------
# regression
data <- read.csv("winequality-red.csv", header=TRUE)
str(data)

set.seed(415)
idx=sample(1:nrow(data),0.7*nrow(data))
data_train = data[idx,]
data_test = data[-idx,]

mlr <- lm(quality ~ ., data = data_train)
summary(mlr)
plot(mlr)
mlr_predict <- predict(mlr, newdata = data_test)

perf_summary_reg[1,] <- perf_eval_reg(data_test$quality, mlr_predict)
perf_summary_reg

# Variable selection method 1: Forward selection
tmp_x_selection <- paste(colnames(data_train)[-12], collapse=" + ")
tmp_xy_selection <- paste("quality ~ ", tmp_x_selection, collapse = "")
as.formula(tmp_xy_selection)
#target~1은 변수없는 모형 (1의 의미)
forward_model <- step(lm(quality ~ 1, data = data_train), 
                      scope = list(upper = as.formula(tmp_xy_selection), lower = quality ~ 1), 
                      direction="forward", trace = 1)
summary(forward_model)

#----------------------------------------------------------
# logistic regression
wine_x <- data[,-12]
wine_x <- scale(wine_x, center = TRUE, scale = TRUE)
wine_y <- data[,12]
wine_data <- data.frame(wine_x, wine_y)
library(dplyr)
wine_data_new <- wine_data %>% mutate(quality= ifelse(wine_y<7,0,1))
wine_data_new <- wine_data_new[,-12]

set.seed(12345)
trn_idx <- sample(1:nrow(wine_data_new), round(0.7*nrow(wine_data_new)))
wine_trn <- wine_data_new[trn_idx,]
wine_tst <- wine_data_new[-trn_idx,]

full_lr <- glm(quality ~ ., family=binomial, wine_trn)
summary(full_lr)

lr_response_predict <- predict(full_lr, type = "response", newdata = wine_tst)
lr_target <- wine_tst$quality
lr_predicted <- rep(0, length(lr_target))
lr_predicted[which(lr_response_predict >= 0.5)] <- 1

cm_full_logistic <- table(lr_target, lr_predicted)
cm_full_logistic

perf_mat_logistic[1,] <- perf_eval_cla(cm_full_logistic)
perf_mat_logistic

#clustering, decision tree, knn

#-----------------------------------------------------------------
#clustering
# wine_data 이용
library(clValid)
library(plotrix)
install.packages("plotrix")

wine_clValid <- clValid(wine_x, 2:10, clMethods = "kmeans", 
                        validation = c("internal", "stability"))
summary(wine_clValid)

wine_kmc <- kmeans(wine_x,2)
str(wine_kmc)
wine_kmc$centers
wine_kmc$size
wine_kmc$cluster

real_class <- wine_y
kmc_cluster <- wine_kmc$cluster
table(real_class, kmc_cluster)

cluster_kmc <- data.frame(wine_x, clusterID = as.factor(wine_kmc$cluster))
kmc_summary <- data.frame()

for (i in 1:(ncol(cluster_kmc)-1)){
  kmc_summary = rbind(kmc_summary, 
                      tapply(cluster_kmc[,i], cluster_kmc$clusterID, mean))
}

colnames(kmc_summary) <- paste("cluster", c(1:2))
rownames(kmc_summary) <- colnames(wine_x)
kmc_summary

# Radar chart
par(mfrow = c(1,2))
for (i in 1:2){
  plot_title <- paste("Radar Chart for Cluster", i, sep=" ")
  radial.plot(kmc_summary[,i], labels = rownames(kmc_summary), 
              radial.lim=c(-2,2), rp.type = "p", main = plot_title, 
              line.col = "red", lwd = 3, show.grid.labels=1)
}
dev.off()

# Compare the first and the second cluster
kmc_cluster1 <- wine_x[wine_kmc$cluster == 1,]
kmc_cluster2 <- wine_x[wine_kmc$cluster == 2,]

# t_test_result
kmc_t_result <- data.frame()

for (i in 1:11){
  
  kmc_t_result[i,1] <- t.test(kmc_cluster1[,i], kmc_cluster2[,i], 
                              alternative = "two.sided")$p.value
  
  kmc_t_result[i,2] <- t.test(kmc_cluster1[,i], kmc_cluster2[,i], 
                              alternative = "greater")$p.value
  
  kmc_t_result[i,3] <- t.test(kmc_cluster1[,i], kmc_cluster2[,i], 
                              alternative = "less")$p.value
}

kmc_t_result

#--------------------------------------------------------------
#decision tree
library(rpart)
library(party)
library(e1071)
install.packages("caret")
library(caret)

rpart_wine_gini <- rpart(quality~., data=wine_trn, parms = list(split = "gini"), method="class")
plot(rpart_wine_gini)
text(rpart_wine_gini, use.n= TRUE)
rpartpredgini <- predict(rpart_wine_gini, wine_tst, type='class')
confusionMatrix(rpartpredgini, wine_tst$quality)

printcp(rpart_wine_gini)
plotcp(rpart_wine_gini)
ptree<-prune(rpart_wine_gini, cp= rpart_wine_gini$cptable[which.min(rpart_wine_gini$cptable[,"xerror"]),"CP"])
plot(ptree)
text(ptree)

rpart_wine_info <- rpart(quality~., data=wine_trn, parms = list(split = "information"), method="class")
plot(rpart_wine_info)
text(rpart_wine_info, use.n= TRUE)
rpartpredinfo <- predict(rpart_wine_info, wine_tst, type='class')
confusionMatrix(rpartpredinfo, wine_tst$quality)

printcp(rpart_wine_info)
plotcp(rpart_wine_info)
ptreeinfo<-prune(rpart_wine_info, cp= rpart_wine_info$cptable[which.min(rpart_wine_info$cptable[,"xerror"]),"CP"])
plot(ptreeinfo)
text(ptreeinfo)

# Prediction
CART.predict <- predict(ptreeinfo, wine_tst, type = "class")
CART.cfm <- table(wine_tst$quality, CART.predict)
CART.cfm

# Performance table
Perf.Table_tree <- matrix(0, nrow = 1, ncol = 6)
rownames(Perf.Table_tree) <- c("CART")
colnames(Perf.Table_tree) <- c("TPR", "Precision", "TNR", "Accuracy", "BCR", "F1-Measure")

Perf.Table_tree[1,] <- perf_eval_cla(CART.cfm)
Perf.Table_tree

#-----------------------------------------------------------
#knn classification (6이하 -> 0 , 7이상 -> 1)
wine_trn_x <- wine_trn[,-12]
wine_trn_y <- wine_trn[,12]
wine_tst_x <- wine_tst[,-12]
wine_tst_y <- wine_tst[,12]

library(class)

knn_k_result <- numeric()

for(i in 1:20){
  wine_knn <- knn(wine_trn_x,wine_tst_x,wine_trn_y,k=i)
  accuracy_knn <- sum(wine_knn==wine_tst_y) / length(wine_tst_y)
  knn_k_result[i] <- accuracy_knn
  
}

library(ggplot2)
knn_result <- data.frame(knn_k_result)
ggplot(knn_result,aes(x=1:20, y=knn_k_result))+geom_line()

#-------------------------testing--------------------
for(i in 1:20){
  wine_knn_cla <- knn(wine_trn_x,wine_tst_x,wine_trn_y,k=i)
  cm_full_knn <- table(wine_tst_y, wine_knn_cla)
  perf_mat_knn[i,] <- perf_eval_cla(cm_full_knn)
}

# Initialize the performance matrix
perf_mat_knn <- matrix(0, 20, 6)
colnames(perf_mat_knn) <- c("TPR (Recall)", "Precision", "TNR", "ACC", "BCR", "F1")
rownames(perf_mat_knn) <- c("knn1","knn2","knn3","knn4","knn5","knn6","knn7","knn8","knn9","knn10","knn11","knn12","knn13","knn14","knn15","knn16","knn17","knn18","knn19","knn20")

#------------------------testing ----------------------------
#knn regression
install.packages("FNN")
library(FNN)
knn_reg <- knn.reg(data_train[,-12], test = data_test[,-12], data_train$quality, k=12)

knn_reg_predict <- knn_reg$pred
perf_summary_reg[2,] <- perf_eval_reg(data_test$quality, knn_reg_predict)
perf_summary_reg

#------------------------------------------------------------------------------
#ANN
n_instance <- dim(wine_data)[1]
# Split the data into the training/test sets
set.seed(12345)
ann_trn_idx <- sample(1:n_instance, round(0.7*n_instance))
ann_trn_data <- wine_data[ann_trn_idx,]
ann_tst_data <- wine_data[-ann_trn_idx,]

perf_summary_reg <- matrix(0,3,3)
rownames(perf_summary_reg) <- c("MLR", "k-NN", "ANN")
colnames(perf_summary_reg) <- c("RMSE", "MAE", "MAPE")

# Find the best number of hidden nodes in terms of BCR
# Candidate hidden nodes
nH <- seq(from=2, to=20, by=2)

# 5-fold cross validation index
val_idx <- sample(c(1:5), length(ann_trn_idx), replace = TRUE, prob = rep(0.2,5))
val_perf <- matrix(0, length(nH), 4)

for (i in 1:length(nH)) {
  
  cat("Training ANN: the number of hidden nodes:", nH[i], "\n")
  eval_fold <- c()
  
  for (j in c(1:5)) {
    
    # Training with the data in (k-1) folds (linout=TRUE 옵션은 회귀분석을 진행하게 해줌)
    # default는 classification임 (linout이 없으면) 
    tmp_trn_data <- ann_trn_data[which(val_idx != j), ]
    tmp_nnet <- nnet(wine_y ~ ., data = tmp_trn_data, size = nH[i], linout = TRUE, 
                     decay = 5e-4, maxit = 500)
    
    # Evaluate the model withe the remaining 1 fold
    tmp_val_input <- ann_trn_data[which(val_idx == j),-12]
    tmp_val_target <- ann_trn_data[which(val_idx == j),12]    
    
    eval_fold <- rbind(eval_fold, cbind(tmp_val_target, predict(tmp_nnet, tmp_val_input)))
    
  }
  
  # nH
  val_perf[i,1] <-nH[i]
  # Record the validation performance
  val_perf[i,2:4] <- perf_eval_reg(eval_fold[,1],eval_fold[,2])
}

ordered_val_perf <- val_perf[order(val_perf[,3], decreasing = FALSE),]
colnames(ordered_val_perf) <- c("nH", "RMSE", "MAE", "MAPE")
ordered_val_perf

# Find the best number of hidden node
best_nH <- ordered_val_perf[1,1]

# Train ANN with the best hidden node
best_nnet <- nnet(wine_y ~ ., data = ann_trn_data, size = best_nH, linout = TRUE, 
                  decay = 5e-4, maxit = 500)

# Test the model and compare the performance
ann_predict <- predict(best_nnet, ann_tst_data[,-12])
perf_summary_reg[3,] <- perf_eval_reg(ann_tst_data$wine_y, ann_predict)
perf_summary_reg
ann_table <- data.frame(ann_predict, ann_tst_data$wine_y)
plot(best_nnet)
library(nnet)
install.packages("devtools")
library(devtools)
library(caret)
library(ROCR)
install.packages("neuralnet")
library(neuralnet)

set.seed(123)
nnet_model <- neuralnet(wine_y ~ ., data=ann_trn_data, hidden=4, threshold=0.01)
plot(nnet_model)

