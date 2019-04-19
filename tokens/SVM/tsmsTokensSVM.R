library(tm);library(pipeR);library(tokenizers);library(FSelector)
source("functions/chi2.R")
source("functions/IG.R")
tsmsDF <- read.csv(file = "csvs/output_sms_last.csv", header = TRUE, 
                    sep = ";", dec = ".", fill = FALSE, stringsAsFactors = FALSE)

tsms.corpus <- VCorpus(VectorSource(tsmsDF$data))
tsms.corpus <- tm_map(tsms.corpus, removePunctuation)
tsms.corpus <- tm_map(tsms.corpus, stripWhitespace)
removeLongWords <- content_transformer(function(x, length) {
  
  return(gsub(paste("(?:^|[[:space:]])[[:alnum:]]{", length, ",}(?=$|[[:space:]])", sep = ""), "", x, perl = T))
})
tsms.corpus <- tm_map(tsms.corpus, removeLongWords, 25)

#Creating Term-Document Matrices
tsms.dtm <- DocumentTermMatrix(tsms.corpus)
tsms.data.frame.dtm <- as.data.frame(as.matrix(tsms.dtm))
tsms.data.frame.dtm$target <- as.factor(tsmsDF$target)

tsms.chi <- chi_squared(target~., tsms.data.frame.dtm)
tsms.ig <- information_gain(target~., tsms.data.frame.dtm)

saveRDS(tsms.chi, file = "results/tsms-chi.rds")
saveRDS(tsms.ig, file = "results/tsms-ig.rds")

################################################################################
################################################################################
################################################################################
library("kernlab");library("caret");library("tidyverse");library("recipes");library("rlist");library("dplyr")
source("transformarRDataPruebas.R")

cutoff <- cutoff.k.percent(tsms.chi, 0.5)
tsms.dtm.cutoff <- subset(tsms.data.frame.dtm, select = cutoff)

tsms.dtm.cutoff$X.userName <- tsmsDF$X.userName
tsms.dtm.cutoff$hashtag <- tsmsDF$hashtag 
tsms.dtm.cutoff$URLs <- tsmsDF$URLs
tsms.dtm.cutoff$emoticon <- tsmsDF$emoticon   
tsms.dtm.cutoff$emoji <- tsmsDF$emoji
tsms.dtm.cutoff$interjection <- tsmsDF$interjection
tsms.dtm.cutoff$language <- as.factor(tsmsDF$language)
tsms.dtm.cutoff$extension <- as.factor(tsmsDF$extension)
tsms.dtm.cutoff$target <- as.factor(tsmsDF$target)

tsms.dtm.cutoff <- tsms.dtm.cutoff %>%
  transformColums("X.userName") %>%
  transformColums("hashtag") %>%
  transformColums("URLs") %>%
  transformColums("emoticon") %>%
  transformColums("emoji") %>% 
  transformColums("interjection") 

def.formula <- as.formula("target~.")

#TSMS
{
  cat("Starting SVM TSMS...\n")
  dataTsms <- subset(tsms.dtm.cutoff, extension == "tsms")
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
                                            allowParallel = TRUE, #execute in parallel.
                                            seeds = set.seed(100)
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
  saveRDS( tsms.svm.trained,file = "results/tsms-tokens-svm-train.rds")
  saveRDS( tsms.svm.cf,file = "results/tsms-tokens-svm-test.rds")
}
