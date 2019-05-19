library(tm);library(pipeR);library(tokenizers);library(FSelector)
source("functions/chi2.R")
source("functions/IG.R")
tsmsDF <- read.csv(file = "csvs/output_sms_last.csv", header = TRUE,
                   sep = ";", dec = ".", fill = FALSE, stringsAsFactors = FALSE)
emlDF <- read.csv(file = "csvs/output_spamassasin_last.csv", header = TRUE,
                   sep = ";", dec = ".", fill = FALSE, stringsAsFactors = FALSE)
ytbidDF <- read.csv(file = "csvs/output_youtube_last.csv", header = TRUE,
                  sep = ";", dec = ".", fill = FALSE, stringsAsFactors = FALSE)

allDF <- rbind(tsmsDF, emlDF, ytbidDF)

corpus <- VCorpus(VectorSource(allDF$data))
corpus <- tm_map(corpus, content_transformer(gsub), pattern = '[!"#$%&\'()*+,.\\/:;<=>?@\\\\^_\\{\\}|~-]+', replacement = ' ')
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, removeNumbers)
removeLongWords <- content_transformer(function(x, length) {

  return(gsub(paste("(?:^|[[:space:]])[[:alnum:]]{", length, ",}(?=$|[[:space:]])", sep = ""), "", x, perl = T))
})
corpus <- tm_map(corpus, removeLongWords, 25)

#Creating Term-Document Matrices
matrix.dtm <- as.matrix(DocumentTermMatrix(corpus))
matrix.dtm <- cbind(as.factor(allDF$target), matrix.dtm)
colnames(matrix.dtm)[1] <- "targetHamSpam"
data.frame.dtm <- as.data.frame(matrix.dtm)
rm(matrix.dtm)
# chi <- chi_squared("targetHamSpam", data.frame.dtm )
# ig <- information_gain("targetHamSpam", data.frame.dtm )
# 
# saveRDS(chi, file = "results/all-chi.rds")
# saveRDS(ig, file = "results/all-ig.rds")

################################################################################
################################################################################
################################################################################
library("kernlab");library("caret");library("tidyverse");library("recipes");library("rlist");library("dplyr")

technique.reduce.dimensionality <- readRDS("results/all-chi.rds")
order <- order(technique.reduce.dimensionality, decreasing = TRUE)
dtm.cutoff <- data.frame.dtm[,order[1:6000]]

rm(order)

dtm.cutoff$X.userName <- allDF$X.userName
dtm.cutoff$hashtag <- allDF$hashtag
dtm.cutoff$URLs <- allDF$URLs
dtm.cutoff$emoticon <- allDF$emoticon
dtm.cutoff$emoji <- allDF$emoji
dtm.cutoff$interjection <- allDF$interjection
dtm.cutoff$extension <- as.factor(allDF$extension)
dtm.cutoff$targetHamSpam <- as.factor(allDF$target)

source("transformColums.R")

dtm.cutoff <- dtm.cutoff %>%
  transformColums("X.userName") %>%
  transformColums("hashtag") %>%
  transformColums("URLs") %>%
  transformColums("emoticon") %>%
  transformColums("emoji") %>%
  transformColums("interjection")

def.formula <- as.formula("targetHamSpam~.")

#ALL
{
  cat("Starting NB ALL...\n")
  set.seed(100)
  indexAll <- caret::createDataPartition(dtm.cutoff$targetHamSpam, p = .75, list = FALSE)
  train <- dtm.cutoff[indexAll, ]
  test <-  dtm.cutoff[-indexAll, ]

  nb.rec <- recipes::recipe(formula = def.formula, data = train) %>%
    step_zv(all_predictors()) %>% #remove zero variance
    step_nzv(all_predictors()) %>% #remove near-zero variance
    step_corr(all_predictors()) #remove high correlation filter.

  nb.trControl <- caret::trainControl(method = "cv", #use cross-validation
                                      number = 10, #divide cross-validation into 10 folds
                                      search = "random", #"grid"
                                      savePredictions = "final", #save predictions of best model.
                                      classProbs = TRUE, #save probabilities obtained for the best model.
                                      summaryFunction = defaultSummary, #use defaultSummary function (only computes Accuracy and Kappa values)
                                      allowParallel = TRUE, #execute in parallel.
                                      seeds = set.seed(100)
  )
  cat("Training NB ALL...\n")
  nb.trained <- caret::train(nb.rec,
                             data = train,
                             method = "nb",
                             trControl = nb.trControl,
                             metric = "Kappa")

  cat("Testing NB ALL...\n")
  nb.cf <- caret::confusionMatrix(
    predict(nb.trained, newdata = test, type = "raw"),
    reference = test$targetHamSpam,
    positive = "spam"
  )

  cat("Finished NB ALL...\n")
  # saveRDS(nb.trained, file = "results/all-tokens-nb-train.rds")
  # saveRDS(nb.cf, file = "results/all-tokens-nb-test.rds")
}
