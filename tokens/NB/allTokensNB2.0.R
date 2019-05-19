technique <- "chi"
method <- "svm"
for (technique in c("ig")) {
for (method in c("svmLinear")) {
library(tm);library(pipeR);library(tokenizers);library(FSelector);library("kernlab");library("caret");library("tidyverse");library("recipes");library("rlist");library("dplyr")
source("functions/chi2.R")
source("functions/IG.R")
source("transformColums.R")
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

technique.reduce.dimensionality <- readRDS(paste("results/tsms-",technique,".rds",sep=""))
order <- order(technique.reduce.dimensionality, decreasing = TRUE)
tsms.dtm.cutoff <- tsms.data.frame.dtm[,order[1:2000]]

tsms.dtm.cutoff$X.userName <- tsmsDF$X.userName
tsms.dtm.cutoff$hashtag <- tsmsDF$hashtag
tsms.dtm.cutoff$URLs <- tsmsDF$URLs
tsms.dtm.cutoff$emoticon <- tsmsDF$emoticon
tsms.dtm.cutoff$emoji <- tsmsDF$emoji
tsms.dtm.cutoff$interjection <- tsmsDF$interjection
tsms.dtm.cutoff$extension <- as.factor(tsmsDF$extension)
tsms.dtm.cutoff$targetHamSpam <- as.factor(tsmsDF$target)

tsms.dtm.cutoff <- tsms.dtm.cutoff %>%
  transformColums("X.userName") %>%
  transformColums("hashtag") %>%
  transformColums("URLs") %>%
  transformColums("emoticon") %>%
  transformColums("emoji") %>%
  transformColums("interjection")

def.formula <- as.formula("targetHamSpam~.")

set.seed(100)
dataTsms <- tsms.dtm.cutoff
indexTsms <- caret::createDataPartition(dataTsms$targetHamSpam, p = .75, list = FALSE)
tsms.train <- dataTsms[indexTsms, ]
tsms.test <-  dataTsms[-indexTsms, ]
rm(tsms.dtm.cutoff);rm(dataTsms);rm(indexTsms);rm(order);rm(tsmsDF);rm(tsms.data.frame.dtm);rm(tsms.matrix.dtm);rm(technique.reduce.dimensionality);rm(tsms.corpus);rm(tsms.dtm)
################################################################################
################################################################################
################################################################################
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

eml.dtm.cutoff <- eml.dtm.cutoff %>%
  transformColums("X.userName") %>%
  transformColums("hashtag") %>%
  transformColums("URLs") %>%
  transformColums("emoticon") %>%
  transformColums("emoji") %>%
  transformColums("interjection")

set.seed(100)
dataEml <- eml.dtm.cutoff
indexEml <- caret::createDataPartition(dataEml$targetHamSpam, p = .75, list = FALSE)
eml.train <- dataEml[indexEml, ]
eml.test <-  dataEml[-indexEml, ]
rm(eml.dtm.cutoff);rm(dataEml);rm(indexEml);rm(order);rm(emlDF);rm(eml.data.frame.dtm);rm(eml.matrix.dtm);rm(technique.reduce.dimensionality);rm(eml.corpus);rm(eml.dtm)
################################################################################
################################################################################
################################################################################
ytbidDF <- read.csv(file = "csvs/output_youtube_last.csv", header = TRUE, 
                    sep = ";", dec = ".", fill = FALSE, stringsAsFactors = FALSE)

ytbid.corpus <- VCorpus(VectorSource(ytbidDF$data))
ytbid.corpus <- tm_map(ytbid.corpus, content_transformer(gsub), pattern = '[!"#$%&\'()*+,.\\/:;<=>?@\\\\^_\\{\\}|~-]+', replacement = ' ')
ytbid.corpus <- tm_map(ytbid.corpus, stripWhitespace)
ytbid.corpus <- tm_map(ytbid.corpus, removeNumbers)
removeLongWords <- content_transformer(function(x, length) {
  
  return(gsub(paste("(?:^|[[:space:]])[[:alnum:]]{", length, ",}(?=$|[[:space:]])", sep = ""), "", x, perl = T))
})
ytbid.corpus <- tm_map(ytbid.corpus, removeLongWords, 25)

# Creating Term-Document Matrices
ytbid.dtm <- DocumentTermMatrix(ytbid.corpus)
ytbid.matrix.dtm <- as.matrix(ytbid.dtm)
ytbid.matrix.dtm <- cbind(as.factor(ytbidDF$target), ytbid.matrix.dtm)
colnames(ytbid.matrix.dtm)[1] <- "targetHamSpam"
ytbid.data.frame.dtm <- as.data.frame(ytbid.matrix.dtm)

technique.reduce.dimensionality <- readRDS(paste("results/ytbid-",technique,".rds",sep=""))
order <- order(technique.reduce.dimensionality, decreasing = TRUE)
ytbid.dtm.cutoff <- ytbid.data.frame.dtm[,order[1:2000]]

ytbid.dtm.cutoff$X.userName <- ytbidDF$X.userName
ytbid.dtm.cutoff$hashtag <- ytbidDF$hashtag
ytbid.dtm.cutoff$URLs <- ytbidDF$URLs
ytbid.dtm.cutoff$emoticon <- ytbidDF$emoticon
ytbid.dtm.cutoff$emoji <- ytbidDF$emoji
ytbid.dtm.cutoff$interjection <- ytbidDF$interjection
ytbid.dtm.cutoff$extension <- as.factor(ytbidDF$extension)
ytbid.dtm.cutoff$targetHamSpam <- as.factor(ytbidDF$target)

ytbid.dtm.cutoff <- ytbid.dtm.cutoff %>%
  transformColums("X.userName") %>%
  transformColums("hashtag") %>%
  transformColums("URLs") %>%
  transformColums("emoticon") %>%
  transformColums("emoji") %>%
  transformColums("interjection")

set.seed(100)
dataYtbid <- ytbid.dtm.cutoff
indexYtbid <- caret::createDataPartition(dataYtbid$targetHamSpam, p = .75, list = FALSE)
ytbid.train <- dataYtbid[indexYtbid, ]
ytbid.test <-  dataYtbid[-indexYtbid, ]
rm(ytbid.dtm.cutoff);rm(dataYtbid);rm(indexYtbid);rm(order);rm(ytbidDF);rm(ytbid.data.frame.dtm);rm(ytbid.matrix.dtm);rm(technique.reduce.dimensionality);rm(ytbid.corpus);rm(ytbid.dtm)
################################################################################
################################################################################
################################################################################
rbind.all.columns <- function(x, y) {
      
  x.diff <- setdiff(colnames(x), colnames(y))
  y.diff <- setdiff(colnames(y), colnames(x))
  x[, c(as.character(y.diff))] <- NA
   
  y[, c(as.character(x.diff))] <- NA
     
  return(rbind(x, y))
}


train <-  rbind.all.columns(ytbid.train,tsms.train)
train <- rbind.all.columns(train,eml.train)
train[is.na(train)] <- 0

test <- rbind.all.columns(ytbid.test,tsms.test)
test <- rbind.all.columns(test,eml.test)
test[is.na(test)] <- 0
rm(eml.test);rm(eml.train);rm(tsms.test);rm(tsms.train);rm(ytbid.train);rm(ytbid.test)

train$extension <- as.numeric(train$extension)
test$extension <- as.numeric(test$extension)
train$X.userName <- as.numeric(train$X.userName)
test$X.userName <- as.numeric(test$X.userName)
train$URLs <- as.numeric(train$URLs)
test$URLs <- as.numeric(test$URLs)
train$emoticon <- as.numeric(train$emoticon)
test$emoticon <- as.numeric(test$emoticon)
train$emoji <- as.numeric(train$emoji)
test$emoji <- as.numeric(test$emoji)
train$interjection <- as.numeric(train$interjection)
test$interjection <- as.numeric(test$interjection)

rec <- recipes::recipe(formula = def.formula, data = train) %>%
  step_zv(all_predictors()) %>% #remove zero variance
  step_nzv(all_predictors()) %>% #remove near-zero variance
  step_corr(all_predictors()) #remove high correlation filter.

trControl <- caret::trainControl(method = "cv", #use cross-validation
                                 number = 10, #divide cross-validation into 10 folds
                                 search = "random", #"grid"
                                 savePredictions = "final", #save predictions of best model.
                                 classProbs = TRUE, #save probabilities obtained for the best model.
                                 summaryFunction = defaultSummary, #use defaultSummary function (only computes Accuracy and Kappa values)
                                 allowParallel = TRUE, #execute in parallel.
                                 seeds = set.seed(100)
)
cat("Training ",method," ALL...\n")
trained <- caret::train(rec,
                        data = train,
                        method = method,
                        trControl = trControl,
                        metric = "Kappa")

cat("Testing ",method," ALL...\n")
cf <- caret::confusionMatrix(
  predict(trained, newdata = test, type = "raw"),
  reference = test$targetHamSpam,
  positive = "spam"
)

cat("Finished ",method," ALL...\n")
saveRDS(trained, file = paste("results/all-tokens-",technique,"-",method,"-train.rds",sep=""))
saveRDS(cf, file = paste("results/all-tokens-",technique,"-",method,"-test.rds",sep=""))

}
}
for(i in names(train)){if(!is.numeric(train[,i])){cat(i," ",class(train[,i]),"\n")}}
