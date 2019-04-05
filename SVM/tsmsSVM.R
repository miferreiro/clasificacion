library("kernlab");library("caret");library("tidyverse");library("recipes");library("rlist");library("dplyr")
source("transformarRDataPruebas.R")
tsmsDF <- read.csv(file = "csvs/outputsyns_sms_last.csv", header = TRUE, 
                    sep = ";", dec = ".", fill = FALSE, stringsAsFactors = FALSE)
tsmsDF <- tsmsDF %>%
  transformColums("X.userName") %>%
  transformColums("hashtag") %>%
  transformColums("URLs") %>%
  transformColums("emoticon") %>%
  transformColums("emoji") %>% 
  transformColums("interjection") 

def.formula <- as.formula("target~.")
tsmsDF$language <- as.factor(tsmsDF$language)
tsmsDF$extension <- as.factor(tsmsDF$extension)
tsmsDF$target <- as.factor(tsmsDF$target)
tsmsDF <-
  dplyr::select(tsmsDF,
                -date,
                -id,
                -language
  )

#TSMS
{
  cat("Starting SVM TSMS...\n")
  dataTsms <- subset(tsmsDF, extension == "tsms")
  indexTsms <- caret::createDataPartition(dataTsms$target, p = .75, list = FALSE)
  tsms.train <- dataTsms[indexTsms, ]
  tsms.test <-  dataTsms[-indexTsms, ]
  
  tsms.svm.rec <- recipes::recipe(formula = def.formula, data = tsms.train) %>%
    step_zv(all_predictors()) %>% #remove zero variance
    step_nzv(all_predictors()) %>% #remove near-zero variance
    step_corr(all_predictors()) %>% #remove high correlation filter.
    step_center(all_predictors()) %>% #normalize data -> mean of zero (important for SVM and KNN) 
    step_scale(all_predictors()) #Scale data to have standard deviation of 1 (important for SVM and KNN).
  
  tsms.svm.trControl <- caret::trainControl(method = "cv", #use cross-validation
                                            number = 10, #divide cross-validation into 10 folds
                                            search = "random", #"grid"
                                            savePredictions = "final", #save predictions of best model.
                                            classProbs = TRUE, #save probabilities obtained for the best model.
                                            summaryFunction = defaultSummary, #use defaultSummary function (only computes Accuracy and Kappa values)
                                            allowParallel = TRUE #execute in parallel.
  )
  cat("Training SVM TSMS...\n")
  tsms.svm.trained <- caret::train(tsms.svm.rec,
                                   data = tsms.train,
                                   method = "svmLinear",
                                   trControl = tsms.svm.trControl,
                                   metric = "Accuracy")
  
  cat("Testing SVM TSMS...\n")
  tsms.svm.cf <- caret::confusionMatrix(
    predict(tsms.svm.trained, newdata = tsms.test, type = "raw"),
    reference = tsms.test$target,
    positive = "spam"
  )
  
  cat("Finished SVM TSMS...\n")
}
