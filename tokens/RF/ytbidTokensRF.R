library(tm);library(pipeR);library(tokenizers);library(FSelector)
source("functions/chi2.R")
source("functions/IG.R")
ytbidDF <- read.csv(file = "csvs/output_youtube_last.csv", header = TRUE, 
                          sep = ";", dec = ".", fill = FALSE, stringsAsFactors = FALSE)
  
ytbid.corpus <- VCorpus(VectorSource(ytbidDF$data))
ytbid.corpus <- tm_map(ytbid.corpus, removePunctuation)

#Creating Term-Document Matrices
ytbid.dtm <- DocumentTermMatrix(ytbid.corpus)
ytbid.data.frame.dtm <- as.data.frame(as.matrix(ytbid.dtm))
ytbid.data.frame.dtm$target <- as.factor(ytbidDF$target)

ytbid.chi <- chi_squared(target~., ytbid.data.frame.dtm)
ytbid.ig <- information_gain(target~., ytbid.data.frame.dtm)

saveRDS(ytbid.chi, file = "results/ytbid-chi.rds")
saveRDS(ytbid.ig, file = "results/ytbid-ig.rds")

################################################################################
################################################################################
################################################################################

library("kernlab");library("caret");library("tidyverse");library("recipes");library("rlist");library("dplyr")
source("transformarRDataPruebas.R")

cutoff <- cutoff.k.percent(ytbid.chi, 0.5)
ytbid.dtm.cutoff <- subset(ytbid.data.frame.dtm, select = cutoff)

ytbid.dtm.cutoff$X.userName <- ytbidDF$X.userName
ytbid.dtm.cutoff$hashtag <- ytbidDF$hashtag 
ytbid.dtm.cutoff$URLs <- ytbidDF$URLs
ytbid.dtm.cutoff$emoticon <- ytbidDF$emoticon   
ytbid.dtm.cutoff$emoji <- ytbidDF$emoji
ytbid.dtm.cutoff$interjection <- ytbidDF$interjection
# ytbid.dtm.cutoff$language <- as.factor(ytbidDF$language)
ytbid.dtm.cutoff$extension <- as.factor(ytbidDF$extension)
ytbid.dtm.cutoff$target <- as.factor(ytbidDF$target)

ytbid.dtm.cutoff <- ytbid.dtm.cutoff %>%
  transformColums("X.userName") %>%
  transformColums("hashtag") %>%
  transformColums("URLs") %>%
  transformColums("emoticon") %>%
  transformColums("emoji") %>% 
  transformColums("interjection") 

def.formula <- as.formula("target~.")

#YTBID
{
  cat("Starting Random Forest YTBID...\n")
  set.seed(100)
  dataYtbid <- subset(ytbid.dtm.cutoff, extension == "ytbid")
  indexYtbid <- caret::createDataPartition(dataYtbid$target, p = .75, list = FALSE)
  ytbid.train <- dataYtbid[indexYtbid, ]
  ytbid.test <-  dataYtbid[-indexYtbid, ]
  
  ytbid.rf.rec <- recipes::recipe(formula = def.formula, data = ytbid.train) %>%
    step_zv(all_predictors()) %>% #remove zero variance
    step_nzv(all_predictors()) %>% #remove near-zero variance
    step_corr(all_predictors()) #remove high correlation filter.
  
  ytbid.rf.trControl <- caret::trainControl( method = "cv", #use cross-validation
                                             number = 10, #divide cross-validation into 10 folds
                                             search = "random", #"grid"
                                             savePredictions = "final", #save predictions of best model.
                                             classProbs = TRUE, #save probabilities obtained for the best model.
                                             summaryFunction = defaultSummary, #use defaultSummary function (only computes Accuracy and Kappa values)
                                             allowParallel = TRUE, #execute in parallel.
                                             seeds = set.seed(100)
  )
  cat("Training Random Forest YTBID...\n")
  ytbid.rf.trained <- caret::train( ytbid.rf.rec,
                                    data = ytbid.train,
                                    method = "rf",
                                    trControl = ytbid.rf.trControl,
                                    metric = "Accuracy")
  
  cat("Testing Random Forest YTBID...\n")
  ytbid.rf.cf <- caret::confusionMatrix(
    predict(ytbid.rf.trained, newdata = ytbid.test, type = "raw"),
    reference = ytbid.test$target,
    positive = "spam"
  )
  
  cat("Finished Random Forest YTBID...\n")
  saveRDS( ytbid.rf.trained,file = "results/ytbid-tokens-rf-train.rds")
  saveRDS( ytbid.rf.cf,file = "results/ytbid-tokens-rf-test.rds")
}
