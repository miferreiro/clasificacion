library("kernlab");library("caret");library("tidyverse");library("recipes");library("rlist");library("dplyr")
source("transformarRDataPruebas.R")
emlDF <- read.csv(file = "csvs/outputsyns_spam_ass.csv", header = TRUE, 
                  sep = ";", dec = ".", fill = FALSE, stringsAsFactors = FALSE)

emlDF <- emlDF %>%
  transformColums("X.userName") %>%
  transformColums("hashtag") %>%
  transformColums("URLs") %>%
  transformColums("emoticon") %>%
  transformColums("emoji") %>% 
  transformColums("interjection") 

def.formula <- as.formula("target~.")
emlDF$language <- as.factor(emlDF$language)
emlDF$extension <- as.factor(emlDF$extension)
emlDF$target <- as.factor(emlDF$target)
emlDF <- dplyr::select(emlDF,
                        -date,
                        -NERDATE,
                        -NERMONEY,
                        -NERNUMBER,
                        -NERADDRESS,
                        -NERLOCATION,
                        -id)
#EML
{
  cat("Starting NB EML...\n")
  dataEml <- subset(emlDF, extension == "eml")
  indexEml <- caret::createDataPartition(dataEml$target, p = .75, list = FALSE)
  eml.train <- dataEml[indexEml, ]
  eml.test <-  dataEml[-indexEml, ]
  
  eml.nb.rec <- recipes::recipe(formula = def.formula, data = eml.train) %>%
    step_zv(all_predictors()) %>% #remove zero variance
    step_nzv(all_predictors()) %>% #remove near-zero variance
    step_corr(all_predictors()) #remove high correlation filter.
  
  eml.nb.trControl <- caret::trainControl(method = "cv", #use cross-validation
                                          number = 10, #divide cross-validation into 10 folds
                                          search = "random", #"grid"
                                          savePredictions = "final", #save predictions of best model.
                                          classProbs = TRUE, #save probabilities obtained for the best model.
                                          summaryFunction = defaultSummary, #use defaultSummary function (only computes Accuracy and Kappa values)
                                          allowParallel = TRUE #execute in parallel.
  )
  cat("Training NB EML...\n")
  eml.nb.trained <- caret::train(eml.nb.rec,
                                 data = eml.train,
                                 method = "nb",
                                 trControl = eml.nb.trControl,
                                 metric = "Accuracy",
                                 preProcess = c("center","scale", "zv", "corr")
  )
  cat("Testing NB EML...\n")
  eml.nb.cf <- caret::confusionMatrix(
    predict(eml.nb.trained, newdata = eml.test, type = "raw"),
    reference = eml.test$target,
    positive = "spam"
  )
  
  cat("Finished NB EML...\n")
}




