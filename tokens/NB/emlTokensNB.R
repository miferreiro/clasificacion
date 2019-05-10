library(tm);library(pipeR);library(tokenizers);library(FSelector)
source("functions/chi2.R")
source("functions/IG.R")
emlDF <- read.csv(file = "csvs/output_spamassasin_last.csv", header = TRUE, 
                   sep = ";", dec = ".", fill = FALSE, stringsAsFactors = FALSE)
eml.corpus <- VCorpus(VectorSource(emlDF$data))
eml.corpus <- tm_map(eml.corpus, content_transformer(gsub), pattern = '[!"#$%&\'()*+,.\\/:;<=>?@\\[\\]\\\\^_\\{\\}|~-]+', replacement = ' ')
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

eml.chi <- chi_squared("targetHamSpam", eml.data.frame.dtm)
eml.ig <- information_gain("targetHamSpam", eml.data.frame.dtm)

saveRDS(eml.chi, file = "results/eml-chi.rds")
saveRDS(eml.ig, file = "results/eml-ig.rds")

################################################################################
################################################################################
################################################################################
library("kernlab");library("caret");library("tidyverse");library("recipes");library("rlist");library("dplyr")

percent <- 0.1
technique.reduce.dimensionality <- eml.chi
order <- order(technique.reduce.dimensionality, decreasing = TRUE)
eml.dtm.cutoff <- eml.data.frame.dtm[, order[1:round(percent * length(order))]]

eml.dtm.cutoff$X.userName <- emlDF$X.userName
eml.dtm.cutoff$hashtag <- emlDF$hashtag 
eml.dtm.cutoff$URLs <- emlDF$URLs
eml.dtm.cutoff$emoticon <- emlDF$emoticon   
eml.dtm.cutoff$emoji <- emlDF$emoji
eml.dtm.cutoff$interjection <- emlDF$interjection
eml.dtm.cutoff$language <- as.factor(emlDF$language)
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
  cat("Starting NB EML...\n")
  dataEml <- subset(eml.dtm.cutoff, extension == "eml")
  indexEml <- caret::createDataPartition(dataEml$targetHamSpam, p = .75, list = FALSE)
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
                                          allowParallel = TRUE, #execute in parallel.
                                          seeds = set.seed(100)
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
    reference = eml.test$targetHamSpam,
    positive = "spam"
  )
  
  cat("Finished NB EML...\n")
  saveRDS( eml.nb.trained,file = "results/eml-tokens-nb-train.rds")
  saveRDS( eml.nb.cf,file = "results/eml-tokens-nb-test.rds")
}

