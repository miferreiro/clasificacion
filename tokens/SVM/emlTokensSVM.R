library(tm);library(pipeR);library(tokenizers);library(FSelector)
source("functions/IG-chi2.R")
emlDF <- read.csv(file = "csvs/output_spamassasin_last.csv", header = TRUE, 
                   sep = ";", dec = ".", fill = FALSE, stringsAsFactors = FALSE)
emlDF <- rbind(emlDF[1:50,],emlDF[3850:3900,])
eml.corpus <- VCorpus(VectorSource(emlDF$data))
eml.corpus <- tm_map(eml.corpus, removePunctuation)

#Creating Term-Document Matrices
eml.dtm <- DocumentTermMatrix(eml.corpus)
eml.data.frame.dtm <- as.data.frame(as.matrix(eml.dtm))
eml.data.frame.dtm$target <- as.factor(emlDF$target)

eml.chi <- chi_squared(target~., eml.data.frame.dtm)
# eml.ig <- information.gain(target~., eml.data.frame.dtm)

saveRDS(eml.chi, file = "results/eml-chi.rds")
# saveRDS(tsms.ig, file = "results/tsms-ig.rds")

################################################################################
################################################################################
################################################################################

library("kernlab");library("caret");library("tidyverse");library("recipes");library("rlist");library("dplyr")
source("transformarRDataPruebas.R")

cutoff <- cutoff.k.percent(eml.chi, 0.5)
eml.dtm.cutoff <- subset(eml.data.frame.dtm, select = cutoff)

eml.dtm.cutoff$X.userName <- emlDF$X.userName
eml.dtm.cutoff$hashtag <- emlDF$hashtag 
eml.dtm.cutoff$URLs <- emlDF$URLs
eml.dtm.cutoff$emoticon <- emlDF$emoticon   
eml.dtm.cutoff$emoji <- emlDF$emoji
eml.dtm.cutoff$interjection <- emlDF$interjection
eml.dtm.cutoff$language <- as.factor(emlDF$language)
eml.dtm.cutoff$extension <- as.factor(emlDF$extension)
eml.dtm.cutoff$target <- as.factor(emlDF$target)

eml.dtm.cutoff <- eml.dtm.cutoff %>%
  transformColums("X.userName") %>%
  transformColums("hashtag") %>%
  transformColums("URLs") %>%
  transformColums("emoticon") %>%
  transformColums("emoji") %>% 
  transformColums("interjection") 

def.formula <- as.formula("target~.")

#eml
{
  cat("Starting SVM EML...\n")
  dataEml <- subset(eml.dtm.cutoff, extension == "eml")
  indexEml <- caret::createDataPartition(dataEml$target, p = .75, list = FALSE)
  
  eml.train <- dataEml[indexEml, ]
  eml.test <-  dataEml[-indexEml, ]
  rm(dataEml)
  eml.svm.rec <- recipes::recipe(formula = def.formula, data = eml.train) %>%
    step_zv(all_predictors()) %>% #remove zero variance
    step_nzv(all_predictors()) %>% #remove near-zero variance
    step_corr(all_predictors()) %>% #remove high correlation filter.
    step_center(all_predictors()) %>% #normalize data -> mean of zero (important for SVM and KNN) 
    step_scale(all_predictors()) #Scale data to have standard deviation of 1 (important for SVM and KNN).
  
  eml.svm.trControl <- caret::trainControl(method = "cv", #use cross-validation
                                           number = 10, #divide cross-validation into 10 folds
                                           search = "random", #"grid"
                                           savePredictions = "final", #save predictions of best model.
                                           classProbs = TRUE, #save probabilities obtained for the best model.
                                           summaryFunction = defaultSummary, #use defaultSummary function (only computes Accuracy and Kappa values)
                                           allowParallel = TRUE,  #execute in parallel.
                                           seeds = set.seed(100)
  )
  cat("Training SVM EML...\n")
  eml.svm.trained <- caret::train(eml.svm.rec,
                                  data = eml.train,
                                  method = "svmLinear",
                                  trControl = eml.svm.trControl,
                                  metric = "Accuracy")
  
  # ,preProcess = c("center","scale", "zv", "corr")
  
  cat("Testing SVM EML...\n")
  eml.svm.cf <- caret::confusionMatrix(
    predict(eml.svm.trained, newdata = eml.test, type = "raw"),
    reference = eml.test$target,
    positive = "spam"
  )
  
  cat("Finished SVM EML...\n")
  saveRDS( eml.svm.trained,file = "results/eml-tokens-svm-train.rds")
  saveRDS( eml.svm.cf,file = "results/eml-tokens-svm-test.rds")
}

