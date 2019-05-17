technique <- "chi"
for(technique in c("chi","ig")){
library(tm);library(pipeR);library(tokenizers);library(FSelector)
source("functions/chi2.R")
source("functions/IG.R")
emlDF <- read.csv(file = "csvs/output_spamassasin_last.csv", header = TRUE, 
                   sep = ";", dec = ".", fill = FALSE, stringsAsFactors = FALSE)

eml.corpus <- VCorpus(VectorSource(emlDF$data))
eml.corpus <- tm_map(eml.corpus, content_transformer(gsub), pattern = '[!"#$%&\'()*+,.\\/:;<=>?@\\\\^_\\{\\}|~-]+', replacement = ' ')
eml.corpus <- tm_map(eml.corpus, stripWhitespace)
eml.corpus <- tm_map(eml.corpus, removeNumbers)
removeLongWords <- content_transformer(function(x, length) {
  
  return(gsub(paste("(?:^|[[:space:]])[[:alnum:]]{", length, ",}(?=$|[[:space:]])", sep = ""), "", x, perl = T))
})
eml.corpus <- tm_map(eml.corpus, removeLongWords, 25)

#Creating Term-Document Matrices
eml.dtm <- DocumentTermMatrix(eml.corpus)
eml.matrix.dtm <- as.matrix(eml.dtm)
eml.matrix.dtm <- cbind(as.factor(emlDF$target), eml.matrix.dtm)
colnames(eml.matrix.dtm)[1] <- "targetHamSpam"
eml.data.frame.dtm <- as.data.frame(eml.matrix.dtm)

# eml.chi <- chi_squared("targetHamSpam", eml.data.frame.dtm)
# eml.ig <- information_gain("targetHamSpam", eml.data.frame.dtm)
# 
# saveRDS(eml.chi, file = "results/eml-chi.rds")
# saveRDS(eml.ig, file = "results/eml-ig.rds")

################################################################################
################################################################################
################################################################################
library("kernlab");library("caret");library("tidyverse");library("recipes");library("rlist");library("dplyr")

technique.reduce.dimensionality <- readRDS(paste("results/eml-",technique,".rds",sep=""))
order <- order(technique.reduce.dimensionality, decreasing = TRUE)
eml.dtm.cutoff <- eml.data.frame.dtm[,order[1:2000]]

eml.dtm.cutoff$X.userName <- emlDF$X.userName
eml.dtm.cutoff$hashtag <- emlDF$hashtag 
eml.dtm.cutoff$URLs <- emlDF$URLs
eml.dtm.cutoff$emoticon <- emlDF$emoticon   
eml.dtm.cutoff$emoji <- emlDF$emoji
eml.dtm.cutoff$interjection <- emlDF$interjection
eml.dtm.cutoff$extension <- as.factor(emlDF$extension)
eml.dtm.cutoff$targetHamSpam <- as.factor(emlDF$target)

source("transformColums.R")

eml.dtm.cutoff <- eml.dtm.cutoff %>%
  transformColums("X.userName") %>%
  transformColums("hashtag") %>%
  transformColums("URLs") %>%
  transformColums("emoticon") %>%
  transformColums("emoji") %>% 
  transformColums("interjection") 

def.formula <- as.formula("targetHamSpam~.")

#EML
{
  cat("Starting Random Forest EML...\n")
  set.seed(100)
  dataEml <- subset(eml.dtm.cutoff, extension == "eml")
  indexEml <- caret::createDataPartition(dataEml$target, p = .75, list = FALSE)
  eml.train <- dataEml[indexEml, ]
  eml.test <-  dataEml[-indexEml, ]
  
  eml.rf.rec <- recipes::recipe(formula = def.formula, data = eml.train) %>%
    step_zv(all_predictors()) %>% #remove zero variance
    step_nzv(all_predictors()) %>% #remove near-zero variance
    step_corr(all_predictors()) #remove high correlation filter.
  
  eml.rf.trControl <- caret::trainControl(method = "cv", #use cross-validation
                                          number = 10, #divide cross-validation into 10 folds
                                          search = "random", #"grid"
                                          savePredictions = "final", #save predictions of best model.
                                          classProbs = TRUE, #save probabilities obtained for the best model.
                                          summaryFunction = defaultSummary, #use defaultSummary function (only computes Accuracy and Kappa values)
                                          allowParallel = TRUE, #execute in parallel.
                                          seeds = set.seed(100)
  )
  cat("Training Random Forest EML...\n")
  eml.rf.trained <- caret::train(eml.rf.rec,
                                 data = eml.train,
                                 method = "rf",
                                 trControl = eml.rf.trControl,
                                 metric = "Kappa")
  
  cat("Testing Random Forest EML...\n")
  eml.rf.cf <- caret::confusionMatrix(
    predict(eml.rf.trained, newdata = eml.test, type = "raw"),
    reference = eml.test$targetHamSpam,
    positive = "spam"
  )
  
  cat("Finished Random Forest EML...\n")
  saveRDS(eml.rf.trained, file = paste("results/eml-tokens-",technique,"-rf-train.rds",sep=""))
  saveRDS(eml.rf.cf, file = paste("results/eml-tokens-",technique,"-rf-test.rds",sep=""))
}
}