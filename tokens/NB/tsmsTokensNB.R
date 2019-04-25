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
tsms.matrix.dtm <- as.matrix(tsms.dtm)
tsms.matrix.dtm <- cbind(as.factor(tsmsDF$target), tsms.matrix.dtm)
colnames(tsms.matrix.dtm)[1] <- "targetHamSpam"
tsms.data.frame.dtm <- as.data.frame(tsms.matrix.dtm)

tsms.chi <- chi_squared("targetHamSpam", tsms.data.frame.dtm )
tsms.ig <- information_gain("targetHamSpam", tsms.data.frame.dtm )

saveRDS(tsms.chi, file = "results/tsms-chi.rds")
saveRDS(tsms.ig, file = "results/tsms-ig.rds")

################################################################################
################################################################################
################################################################################
library("kernlab");library("caret");library("tidyverse");library("recipes");library("rlist");library("dplyr")

percent <- 0.1
technique.reduce.dimensionality <- tsms.chi
order <- order(technique.reduce.dimensionality, decreasing = TRUE)
tsms.dtm.cutoff <- tsms.data.frame.dtm[, order[1:round(percent * length(order))]]

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
  cat("Starting NB TSMS...\n")
  dataTsms <- subset(tsms.dtm.cutoff, extension == "tsms")
  indexTsms <- caret::createDataPartition(dataTsms$targetHamSpam, p = .75, list = FALSE)
  tsms.train <- dataTsms[indexTsms, ]
  tsms.test <-  dataTsms[-indexTsms, ]
  
  tsms.nb.rec <- recipes::recipe(formula = def.formula, data = tsms.train) %>%
    step_zv(all_predictors()) %>% #remove zero variance
    step_nzv(all_predictors()) %>% #remove near-zero variance
    step_corr(all_predictors()) #remove high correlation filter.
  
  tsms.nb.trControl <- caret::trainControl(method = "cv", #use cross-validation
                                           number = 10, #divide cross-validation into 10 folds
                                           search = "random", #"grid"
                                           savePredictions = "final", #save predictions of best model.
                                           classProbs = TRUE, #save probabilities obtained for the best model.
                                           summaryFunction = defaultSummary, #use defaultSummary function (only computes Accuracy and Kappa values)
                                           allowParallel = TRUE, #execute in parallel.
                                           seeds = set.seed(100)
  )
  cat("Training NB TSMS...\n")
  tsms.nb.trained <- caret::train(tsms.nb.rec,
                                  data = tsms.train,
                                  method = "nb",
                                  trControl = tsms.nb.trControl,
                                  metric = "Accuracy")
  
  cat("Testing NB TSMS...\n")
  tsms.nb.cf <- caret::confusionMatrix(
    predict(tsms.nb.trained, newdata = tsms.test, type = "raw"),
    reference = tsms.test$targetHamSpam,
    positive = "spam"
  )
  
  cat("Finished NB TSMS...\n")
  saveRDS( tsms.nb.trained,file = "results/tsms-tokens-nb-train.rds")
  saveRDS( tsms.nb.cf,file = "results/tsms-tokens-nb-test.rds")
}


