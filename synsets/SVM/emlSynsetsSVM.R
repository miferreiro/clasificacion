library("kernlab");library("caret");library("tidyverse");library("recipes");library("rlist");library("dplyr")
source("transformarRDataPruebas.R")
dataEml <- read.csv(file = "csvs/outputsyns_spamassassin_last.csv", header = TRUE, 
                  sep = ";", dec = ".", fill = FALSE, stringsAsFactors = FALSE)

dataEml <- dataEml %>%
  transformColums("X.userName") %>%
  transformColums("hashtag") %>%
  transformColums("URLs") %>%
  transformColums("emoticon") %>%
  transformColums("emoji") %>% 
  transformColums("interjection") 

def.formula <- as.formula("target~.")
dataEml$language <- as.factor(dataEml$language)
dataEml$extension <- as.factor(dataEml$extension)
dataEml$target <- as.factor(dataEml$target)
dataEml <- dplyr::select(dataEml,
                      -date,
                      -NERDATE,
                      -NERMONEY,
                      -NERNUMBER,
                      -NERADDRESS,
                      -NERLOCATION,
                      -id,
                      -language,
                      -polarity
  )
#EML
{
  cat("Starting SVM EML...\n")
  dataEml <- subset(dataEml, extension == "eml")
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
                                          allowParallel = TRUE #execute in parallel.
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
}
