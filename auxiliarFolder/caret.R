library("kernlab")
library("caret")
library("tidyverse")
library("recipes")
library("rlist")

rm(list = ls())
setwd("C:/Users/Miguel/Desktop/clasificacion")
source("transformarRDataPruebas.R")
RData <- readRDS("dataFrameAllSynsets.RData")
RData <- transformarRDataPruebas(RData)
RData <- transformColums(RData, "userName")
RData <- transformColums(RData, "hashtag")
RData <- transformColums(RData, "URLs")
RData <- transformColums(RData, "Emoticon")
RData <- transformColums(RData, "Emojis")
RData <- transformColums(RData, "langpropname")
RData <- transformColums(RData, "interjection")

#Dataset spliting in .75% train y .25% test
index <- caret::createDataPartition(RData$target,p = .75, list = FALSE)
spam.train <- spam[index, ]
spam.test <-  spam[-index, ]


rec <- recipes::recipe(formula=as.formula("type ~."),data=spam.train) %>% 
  step_zv(all_predictors()) %>% #remove zero variance
  step_nzv(all_predictors()) %>% #remove near-zero variance
  step_corr(all_predictors()) #%>% #remove high correlation filter.
step_center(all_predictors()) %>% #normalize data -> mean of zero (important for SVM and KNN) 
  step_scale(all_predictors()) #Scale data to have standard deviation of 1 (important for SVM and KNN).

trControl <- caret::trainControl( method="cv", #use cross-validation
                                  number = 10, #divide cross-validation into 10 folds
                                  search = "random", #"grid"
                                  savePredictions = "final", #save predictions of best model.
                                  classProbs = TRUE, #save probabilities obtained for the best model.
                                  summaryFunction = defaultSummary, #use defaultSummary function (only computes Accuracy and Kappa values)
                                  allowParallel = TRUE #execute in parallel.
)

nb.trained <- caret::train(rec,data=spam.train,method="nb",trControl=trControl, metric="Accuracy")
svm.trained <- caret::train(rec,data=spam.train,method="svmLinear",trControl=trControl, metric="Accuracy")


nb.cf <- caret::confusionMatrix(predict(nb.trained,newdata=spam.test,type="raw"), reference = spam.test$type, positive="spam")
svm.cf <- caret::confusionMatrix(predict(svm.trained,newdata=spam.test,type="raw"), reference = spam.test$type, positive="spam")

