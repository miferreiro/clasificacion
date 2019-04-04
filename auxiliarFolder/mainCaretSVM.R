rm(list = ls())
library("kernlab");library("caret");library("tidyverse");library("recipes");library("rlist")
setwd("C:/Users/Miguel/Desktop/clasificacion")

namesExtTypes <- c("tsms", "eml", "twtid", "ytbid", "warc")
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
dimnames(RData)[[2]][31:612] <-  gsub(pattern = ":", x = dimnames(RData)[[2]][31:612], replacement = "_")  
def.formula <- as.formula(paste("target",
                                paste("length","length_after_stopwords",
                                      "userName","hashtag","URLs", "Emoticon", "Emojis",
                                      "language","langpropname","interjection",
                                      paste(names(RData)[31:612], collapse = "+")
                                      ,sep = "+",collapse = "+"),sep = "~"))
#TSMS
{
  cat("Starting TSMS...\n")
  dataTsms <- subset(RData, extension == "tsms")
  indexTsms <- caret::createDataPartition(dataTsms$target, p = .75, list = FALSE)
  tsms.train <- dataTsms[indexTsms, ]
  tsms.test <-  dataTsms[-indexTsms, ]
  
  rec <- recipes::recipe(formula = def.formula, data = tsms.train) %>%
    step_zv(all_predictors()) %>% #remove zero variance
    step_nzv(all_predictors()) %>% #remove near-zero variance
    step_corr(all_predictors()) %>% #remove high correlation filter.
    step_center(all_predictors()) %>% #normalize data -> mean of zero (important for SVM and KNN) 
    step_scale(all_predictors()) #Scale data to have standard deviation of 1 (important for SVM and KNN).
  
  tsms.trControl <- caret::trainControl(  method = "cv", #use cross-validation
                                          number = 10, #divide cross-validation into 10 folds
                                          search = "random", #"grid"
                                          savePredictions = "final", #save predictions of best model.
                                          classProbs = TRUE, #save probabilities obtained for the best model.
                                          summaryFunction = defaultSummary, #use defaultSummary function (only computes Accuracy and Kappa values)
                                          allowParallel = TRUE #execute in parallel.
  )
  
  tsms.svm.trained <- caret::train(
    rec,
    data = tsms.train,
    method = "svmLinear",
    trControl = tsms.trControl,
    metric = "Accuracy"
  )
  
  tsms.svm.cf <- caret::confusionMatrix(
    predict(tsms.svm.trained, newdata = tsms.test, type = "raw"),
    reference = tsms.test$target,
    positive = "spam"
  )
  
  cat("\nFinished TSMS...\n")
}

#EML
{
  cat("Starting EML...\n")
  dataEml <- subset(RData, extension == "eml")
  indexEml <- caret::createDataPartition(dataEml$target, p = .75, list = FALSE)
  eml.train <- dataEml[indexEml, ]
  eml.test <-  dataEml[-indexEml, ]
  
  eml.rec <- recipes::recipe(formula = def.formula, data = eml.train) %>%
    step_zv(all_predictors()) %>% #remove zero variance
    step_nzv(all_predictors()) %>% #remove near-zero variance
    step_corr(all_predictors()) %>% #remove high correlation filter.
    step_center(all_predictors()) %>% #normalize data -> mean of zero (important for SVM and KNN) 
    step_scale(all_predictors()) #Scale data to have standard deviation of 1 (important for SVM and KNN).
  
  
  eml.trControl <- caret::trainControl( method = "cv", #use cross-validation
                                        number = 10, #divide cross-validation into 10 folds
                                        search = "random", #"grid"
                                        savePredictions = "final", #save predictions of best model.
                                        classProbs = TRUE, #save probabilities obtained for the best model.
                                        summaryFunction = defaultSummary, #use defaultSummary function (only computes Accuracy and Kappa values)
                                        allowParallel = TRUE #execute in parallel.
  )
  
  eml.svm.trained <- caret::train(eml.rec,
                                 data = eml.train,
                                 method = "svmLinear",
                                 trControl = eml.trControl,
                                 metric = "Accuracy"
  )
  
  eml.svm.cf <- caret::confusionMatrix(
    predict(eml.svm.trained, newdata = eml.test, type = "raw"),
    reference = eml.test$target,
    positive = "spam"
  )
  
  cat("\nFinished EML...\n")
}

#TWTID
{
  cat("Starting TWTID...\n")
  dataTwtid <- subset(RData, extension == "twtid")
  indexTwtid <- caret::createDataPartition(dataTwtid$target, p = .75, list = FALSE)
  twtid.train <- dataTwtid[indexTwtid, ]
  twtid.test <-  dataTwtid[-indexTwtid, ]
  
  twtid.rec <- recipes::recipe(formula = def.formula, data = twtid.train) %>%
    step_zv(all_predictors()) %>% #remove zero variance
    step_nzv(all_predictors()) %>% #remove near-zero variance
    step_corr(all_predictors()) %>% #remove high correlation filter.
    step_center(all_predictors()) %>% #normalize data -> mean of zero (important for SVM and KNN) 
    step_scale(all_predictors()) #Scale data to have standard deviation of 1 (important for SVM and KNN).
  
  
  twtid.trControl <- caret::trainControl( method = "cv", #use cross-validation
                                          number = 10, #divide cross-validation into 10 folds
                                          search = "random", #"grid"
                                          savePredictions = "final", #save predictions of best model.
                                          classProbs = TRUE, #save probabilities obtained for the best model.
                                          summaryFunction = defaultSummary, #use defaultSummary function (only computes Accuracy and Kappa values)
                                          allowParallel = TRUE #execute in parallel.
  )
  
  twtid.svm.trained <- caret::train(twtid.rec,
                                   data = twtid.train,
                                   method = "svmLinear",
                                   trControl = twtid.trControl,
                                   metric = "Accuracy"
  )
  
  twtid.svm.cf <- caret::confusionMatrix(
    predict(twtid.svm.trained, newdata = twtid.test, type = "raw"),
    reference = twtid.test$target,
    positive = "spam"
  )
  
  cat("\nFinished TWTID...\n")
}

#YTBID
{
  cat("Starting YTBID...\n")
  dataYtbid <- subset(RData, extension == "ytbid")
  indexYtbid <- caret::createDataPartition(dataYtbid$target, p = .75, list = FALSE)
  ytbid.train <- dataYtbid[indexYtbid, ]
  ytbid.test <-  dataYtbid[-indexYtbid, ]
  
  ytbid.rec <- recipes::recipe(formula = def.formula, data = ytbid.train) %>%
    step_zv(all_predictors()) %>% #remove zero variance
    step_nzv(all_predictors()) %>% #remove near-zero variance
    step_corr(all_predictors()) %>% #remove high correlation filter.
    step_center(all_predictors()) %>% #normalize data -> mean of zero (important for SVM and KNN) 
    step_scale(all_predictors()) #Scale data to have standard deviation of 1 (important for SVM and KNN).
  
  
  ytbid.trControl <- caret::trainControl( method = "cv", #use cross-validation
                                          number = 10, #divide cross-validation into 10 folds
                                          search = "random", #"grid"
                                          savePredictions = "final", #save predictions of best model.
                                          classProbs = TRUE, #save probabilities obtained for the best model.
                                          summaryFunction = defaultSummary, #use defaultSummary function (only computes Accuracy and Kappa values)
                                          allowParallel = TRUE #execute in parallel.
  )
  
  ytbid.svm.trained <- caret::train( ytbid.rec,
                                    data = ytbid.train,
                                    method = "svmLinear",
                                    trControl = ytbid.trControl,
                                    metric = "Accuracy"
  )
  
  ytbid.svm.cf <- caret::confusionMatrix(
    predict(ytbid.svm.trained, newdata = ytbid.test, type = "raw"),
    reference = ytbid.test$target,
    positive = "spam"
  )
  
  cat("\nFinished YTBID...\n")
}

#WARC
{
  cat("Starting WARC...\n")
  dataWarc <- subset(RData, extension == "warc")
  indexWarc <- caret::createDataPartition(dataWarc$target, p = .75, list = FALSE)
  warc.train <- dataWarc[indexWarc, ]
  warc.test <-  dataWarc[-indexWarc, ]
  
  warc.rec <- recipes::recipe(formula = def.formula, data = warc.train) %>%
    step_zv(all_predictors()) %>% #remove zero variance
    step_nzv(all_predictors()) %>% #remove near-zero variance
    step_corr(all_predictors()) %>% #remove high correlation filter.
    step_center(all_predictors()) %>% #normalize data -> mean of zero (important for SVM and KNN) 
    step_scale(all_predictors()) #Scale data to have standard deviation of 1 (important for SVM and KNN).
  
  
  warc.trControl <- caret::trainControl(  method = "cv", #use cross-validation
                                          number = 10, #divide cross-validation into 10 folds
                                          search = "random", #"grid"
                                          savePredictions = "final", #save predictions of best model.
                                          classProbs = TRUE, #save probabilities obtained for the best model.
                                          summaryFunction = defaultSummary, #use defaultSummary function (only computes Accuracy and Kappa values)
                                          allowParallel = TRUE #execute in parallel.
  )
  
  warc.svm.trained <- caret::train(  warc.rec,
                                    data = warc.train,
                                    method = "svmLinear",
                                    trControl = warc.trControl,
                                    metric = "Accuracy"
  )
  
  warc.svm.cf <- caret::confusionMatrix(
    predict(warc.svm.trained, newdata = warc.test, type = "raw"),
    reference = warc.test$target,
    positive = "spam"
  )
  
  cat("\nFinished WARC...\n")
}
