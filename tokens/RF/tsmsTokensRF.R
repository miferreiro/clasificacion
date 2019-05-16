library(tm);library(pipeR);library(tokenizers);library(FSelector)
source("functions/chi2.R")
source("functions/IG.R")
tsmsDF <- read.csv(file = "csvs/output_sms_last.csv", header = TRUE, 
                    sep = ";", dec = ".", fill = FALSE, stringsAsFactors = FALSE)

tsms.corpus <- VCorpus(VectorSource(tsmsDF$data))
tsms.corpus <- tm_map(tsms.corpus, content_transformer(gsub), pattern = '[!"#$%&\'()*+,.\\/:;<=>?@\\\\^_\\{\\}|~-]+', replacement = ' ')
tsms.corpus <- tm_map(tsms.corpus, stripWhitespace)
tsms.corpus <- tm_map(tsms.corpus, removeNumbers)
removeLongWords <- content_transformer(function(x, length) {
  
  return(gsub(paste("(?:^|[[:space:]])[[:alnum:]]{", length, ",}(?=$|[[:space:]])", sep = ""), "", x, perl = T))
})
tsms.corpus <- tm_map(tsms.corpus, removeLongWords, 25)

#Creating Term-Document Matrices
tsms.dtm <- DocumentTermMatrix(tsms.corpus)
tsms.matrix.dtm <- as.matrix(tsms.dtm)
tsms.matrix.dtm <- cbind(as.factor(tsmsDF$target), tsms.matrix.dtm)
colnames(tsms.matrix.dtm)[1] <- "targetHamSpam"
tsms.data.frame.dtm <- as.data.frame(tsms.matrix.dtm)

# tsms.chi <- chi_squared("targetHamSpam", tsms.data.frame.dtm)
# tsms.ig <- information_gain("targetHamSpam", tsms.data.frame.dtm)
# 
# saveRDS(tsms.chi, file = "results/tsms-chi.rds")
# saveRDS(tsms.ig, file = "results/tsms-ig.rds")

################################################################################
################################################################################
################################################################################
library("kernlab");library("caret");library("tidyverse");library("recipes");library("rlist");library("dplyr")

technique.reduce.dimensionality <- readRDS("results/tsms-ig.rds")
order <- order(technique.reduce.dimensionality, decreasing = TRUE)
tsms.dtm.cutoff <- tsms.data.frame.dtm[,order[1:2000]]

tsms.dtm.cutoff$X.userName <- tsmsDF$X.userName
tsms.dtm.cutoff$hashtag <- tsmsDF$hashtag 
tsms.dtm.cutoff$URLs <- tsmsDF$URLs
tsms.dtm.cutoff$emoticon <- tsmsDF$emoticon   
tsms.dtm.cutoff$emoji <- tsmsDF$emoji
tsms.dtm.cutoff$interjection <- tsmsDF$interjection
tsms.dtm.cutoff$language <- as.factor(tsmsDF$language)
tsms.dtm.cutoff$extension <- as.factor(tsmsDF$extension)
tsms.dtm.cutoff$targetHamSpam <- as.factor(tsmsDF$target)

source("transformColums.R")

tsms.dtm.cutoff <- tsms.dtm.cutoff %>%
  transformColums("X.userName") %>%
  transformColums("hashtag") %>%
  transformColums("URLs") %>%
  transformColums("emoticon") %>%
  transformColums("emoji") %>% 
  transformColums("interjection") 

def.formula <- as.formula("targetHamSpam~.")

#TSMS
{
  cat("Starting Random Forest TSMS...\n")
  set.seed(100)
  dataTsms <- subset(tsms.dtm.cutoff, extension == "tsms")
  indexTsms <- caret::createDataPartition(dataTsms$targetHamSpam, p = .75, list = FALSE)
  tsms.train <- dataTsms[indexTsms, ]
  tsms.test <-  dataTsms[-indexTsms, ]
  
  tsms.rf.rec <- recipes::recipe(formula = def.formula, data = tsms.train) %>%
    step_zv(all_predictors()) %>% #remove zero variancer
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
    reference = tsms.test$targetHamSpam,
    positive = "spam"
  )
  
  cat("Finished Random Forest TSMS...\n")
  saveRDS( tsms.rf.trained,file = "results/tsms-tokens-ig-rf-train.rds")
  saveRDS( tsms.rf.cf,file = "results/tsms-tokens-ig-rf-test.rds")
}
