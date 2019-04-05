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
tsmsDF <- dplyr::select(tsmsDF,
                        -date,
                        -id)

#TSMS
{
  cat("Starting Random Forest TSMS...\n")
  dataTsms <- subset(tsmsDF, extension == "tsms")
  indexTsms <- caret::createDataPartition(dataTsms$target, p = .75, list = FALSE)
  tsms.train <- dataTsms[indexTsms, ]
  tsms.test <-  dataTsms[-indexTsms, ]
  
  tsms.rf.rec <- recipes::recipe(formula = def.formula, data = tsms.train) %>%
    step_zv(all_predictors()) %>% #remove zero variance
    step_nzv(all_predictors()) %>% #remove near-zero variance
    step_corr(all_predictors()) #remove high correlation filter.
  
  tsms.rf.trControl <- caret::trainControl(method = "cv", #use cross-validation
                                           number = 10, #divide cross-validation into 10 folds
                                           search = "random", #"grid"
                                           savePredictions = "final", #save predictions of best model.
                                           classProbs = TRUE, #save probabilities obtained for the best model.
                                           summaryFunction = defaultSummary, #use defaultSummary function (only computes Accuracy and Kappa values)
                                           allowParallel = TRUE #execute in parallel.
  )
  cat("Training Random Forest TSMS...\n")
  tsms.rf.trained <- caret::train(tsms.rf.rec,
                                  data = tsms.train,
                                  method = "rf",
                                  trControl = tsms.rf.trControl,
                                  metric = "Accuracy")
  
  cat("Testing Random Forest TSMS...\n")
  tsms.rf.cf <- caret::confusionMatrix(
    predict(tsms.rf.trained, newdata = tsms.test, type = "raw"),
    reference = tsms.test$target,
    positive = "spam"
  )
  
  cat("Finished Random Forest TSMS...\n")
}






