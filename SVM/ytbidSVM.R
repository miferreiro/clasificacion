library("kernlab");library("caret");library("tidyverse");library("recipes");library("rlist");library("dplyr")
source("transformarRDataPruebas.R")
ytbidDF <- read.csv(file = "csvs/outputsyns.csv", header = TRUE, 
                    sep = ";", dec = ".", fill = FALSE, stringsAsFactors = FALSE)
ytbidDF <- ytbidDF %>%
  transformColums("X.userName") %>%
  transformColums("hashtag") %>%
  transformColums("URLs") %>%
  transformColums("emoticon") %>%
  transformColums("emoji") %>% 
  transformColums("interjection") 

def.formula <- as.formula("target~.")
ytbidDF$language <- as.factor(ytbidDF$language)
ytbidDF$extension <- as.factor(ytbidDF$extension)
ytbidDF$target <- as.factor(ytbidDF$target)
ytbidDF <-
  dplyr::select(ytbidDF,
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

#YTBID
{
  cat("Starting SVM YTBID...\n")
  dataYtbid <- subset(ytbidDF, extension == "ytbid")
  indexYtbid <- caret::createDataPartition(dataYtbid$target, p = .75, list = FALSE)
  ytbid.train <- dataYtbid[indexYtbid, ]
  ytbid.test <-  dataYtbid[-indexYtbid, ]
  
  ytbid.svm.rec <- recipes::recipe(formula = def.formula, data = ytbid.train) %>%
    step_zv(all_predictors()) %>% #remove zero variance
    step_nzv(all_predictors()) %>% #remove near-zero variance
    step_corr(all_predictors()) %>% #remove high correlation filter.
    step_center(all_predictors()) %>% #normalize data -> mean of zero (important for SVM and KNN) 
    step_scale(all_predictors()) #Scale data to have standard deviation of 1 (important for SVM and KNN).
  
  ytbid.svm.trControl <- caret::trainControl(method = "cv", #use cross-validation
                                             number = 10, #divide cross-validation into 10 folds
                                             search = "random", #"grid"
                                             savePredictions = "final", #save predictions of best model.
                                             classProbs = TRUE, #save probabilities obtained for the best model.
                                             summaryFunction = defaultSummary, #use defaultSummary function (only computes Accuracy and Kappa values)
                                             allowParallel = TRUE #execute in parallel.
  )
  cat("Training SVM YTBID...\n")
  ytbid.svm.trained <- caret::train(ytbid.svm.rec,
                                    data = ytbid.train,
                                    method = "svmLinear",
                                    trControl = ytbid.svm.trControl,
                                    metric = "Accuracy")
  
  cat("Testing SVM YTBID...\n")
  ytbid.svm.cf <- caret::confusionMatrix(
    predict(ytbid.svm.trained, newdata = ytbid.test, type = "raw"),
    reference = ytbid.test$target,
    positive = "spam"
  )
  
  cat("Finished SVM YTBID...\n")
}






