library("kernlab");library("caret");library("tidyverse");library("recipes");library("rlist");library("dplyr")
source("transformarRDataPruebas.R")
ytbidDF <- read.csv(file = "csvs/outputsyns_youtube_last.csv", header = TRUE, 
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
ytbidDF <- dplyr::select(ytbidDF,
                        -date,
                        -id)

#YTBID
{
  cat("Starting NB YTBID...\n")
  dataYtbid <- subset(ytbidDF, extension == "ytbid")
  indexYtbid <- caret::createDataPartition(dataYtbid$target, p = .75, list = FALSE)
  ytbid.train <- dataYtbid[indexYtbid, ]
  ytbid.test <-  dataYtbid[-indexYtbid, ]
  
  ytbid.nb.rec <- recipes::recipe(formula = def.formula, data = ytbid.train) %>%
    step_zv(all_predictors()) %>% #remove zero variance
    step_nzv(all_predictors()) %>% #remove near-zero variance
    step_corr(all_predictors()) #remove high correlation filter.
  
  ytbid.nb.trControl <- caret::trainControl( method = "cv", #use cross-validation
                                          number = 10, #divide cross-validation into 10 folds
                                          search = "random", #"grid"
                                          savePredictions = "final", #save predictions of best model.
                                          classProbs = TRUE, #save probabilities obtained for the best model.
                                          summaryFunction = defaultSummary, #use defaultSummary function (only computes Accuracy and Kappa values)
                                          allowParallel = TRUE #execute in parallel.
  )
  cat("Training NB YTBID...\n")
  ytbid.nb.trained <- caret::train( ytbid.nb.rec,
                                    data = ytbid.train,
                                    method = "nb",
                                    trControl = ytbid.nb.trControl,
                                    metric = "Accuracy")
  
  cat("Testing NB YTBID...\n")
  ytbid.nb.cf <- caret::confusionMatrix(
    predict(ytbid.nb.trained, newdata = ytbid.test, type = "raw"),
    reference = ytbid.test$target,
    positive = "spam"
  )
  
  cat("Finished NB YTBID...\n")
}






