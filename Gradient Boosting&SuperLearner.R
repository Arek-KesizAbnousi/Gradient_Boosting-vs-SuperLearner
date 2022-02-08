library(mlbench)
data(Sonar)
dim(Sonar)
head(Sonar)
X  <-  log(Sonar[,1:60] + 1) # predictors
y  <-  as.numeric(Sonar$Class)-1
Sonar <- cbind(y,X)

library(gbm)
library(xgboost)
library(SuperLearner)
library(MASS)
library(e1071)

classification_accuracy_table <- matrix(0,nrow = 100,ncol = 2)

for(i in 1:100)
{
  set <- sample(1:208, 158)
  training_data <- Sonar[set,]
  testing_data <- Sonar[-set,]
  Gradient_Boosting_fit <- gbm(y~., data=training_data, distribution ="bernoulli")
  
  classification_accuracy_table[i,1] <- mean(ifelse(predict(Gradient_Boosting_fit,testing_data[,-1],type = "link")>0.5,1,0)== testing_data[,1])
  
  SuperLearner_fit <- SuperLearner(training_data$y , training_data[,-1], family = binomial(),SL.library = c("SL.randomForest", "SL.lda","SL.rpartPrune", "SL.svm"))
  
  classification_accuracy_table[i,2] <- mean(ifelse(predict(SuperLearner_fit, testing_data[,-1], onlySL = TRUE)$pred[,1]>0.5,1,0)== testing_data[,1])
}

# : column 1 :  Accuracy via  Gradient_Boosting . column 2 : Accuracy via SuperLearner
classification_accuracy_table
colMeans(classification_accuracy_table)

#We run the code above and to get the Accuracy table.
# : column 1 :  Accuracy via  Gradient_Boosting . column 2 : Accuracy via SuperLearner
   classification_accuracy_table

# Calculating the Average Mover the 100 independent replications for the two methods

 colMeans(classification_accuracy_table)
