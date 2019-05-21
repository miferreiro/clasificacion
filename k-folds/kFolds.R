extension <- "eml"
library(tm);library(pipeR);library(tokenizers);library(FSelector)
library("kernlab");library("caret");library("tidyverse");library("recipes");library("rlist");library("dplyr")
for(model in c("nb","svmLinear","rf")){
for(typeDimensionality in c("chi","ig")){
for(k in 3:13){
  print(model)
  print(k)
  print(typeDimensionality)

fileDimensionality <-paste("results/",extension,"-",typeDimensionality,".rds",sep="")
fileTrain <-paste("resultsWithOutSteps/",extension,"-tokens-", typeDimensionality, "-", model, "-train-", k, "-folds.rds", sep="")
fileTest <- paste("resultsWithOutSteps/",extension,"-tokens-", typeDimensionality, "-", model, "-test-", k, "-folds.rds", sep="")

source("functions/chi2.R")
source("functions/IG.R")
DF <- read.csv(file = "csvs/output_spamassasin_last.csv", header = TRUE, 
                   sep = ";", dec = ".", fill = FALSE, stringsAsFactors = FALSE)

corpus <- VCorpus(VectorSource(DF$data))
corpus <- tm_map(corpus, content_transformer(gsub), pattern = '[!"#$%&\'()*+,.\\/:;<=>?@\\\\^_\\{\\}|~-]+', replacement = ' ')
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, removeNumbers)
removeLongWords <- content_transformer(function(x, length) {
  
  return(gsub(paste("(?:^|[[:space:]])[[:alnum:]]{", length, ",}(?=$|[[:space:]])", sep = ""), "", x, perl = T))
})
corpus <- tm_map(corpus, removeLongWords, 25)

#Creating Term-Document Matrices
dtm <- DocumentTermMatrix(corpus)
matrix.dtm <- as.matrix(dtm)
matrix.dtm <- cbind(as.factor(DF$target), matrix.dtm)
colnames(matrix.dtm)[1] <- "targetHamSpam"
data.frame.dtm <- as.data.frame(matrix.dtm)

# chi <- chi_squared("targetHamSpam", data.frame.dtm )
# ig <- information_gain("targetHamSpam", data.frame.dtm )
# 
# saveRDS(chi, file = "results/",extension,"-chi.rds")
# saveRDS(ig, file = "results/",extension,"-ig.rds")

################################################################################
################################################################################
################################################################################

technique.reduce.dimensionality <- readRDS(fileDimensionality)
order <- order(technique.reduce.dimensionality, decreasing = TRUE)
dtm.cutoff <- data.frame.dtm[,order[1:2000]]

dtm.cutoff$X.userName <- DF$X.userName
dtm.cutoff$hashtag <- DF$hashtag 
dtm.cutoff$URLs <- DF$URLs
dtm.cutoff$emoticon <- DF$emoticon   
dtm.cutoff$emoji <- DF$emoji
dtm.cutoff$interjection <- DF$interjection
dtm.cutoff$extension <- as.factor(DF$extension)
dtm.cutoff$targetHamSpam <- as.factor(DF$target)

source("transformColums.R")

dtm.cutoff <- dtm.cutoff %>%
  transformColums("X.userName") %>%
  transformColums("hashtag") %>%
  transformColums("URLs") %>%
  transformColums("emoticon") %>%
  transformColums("emoji") %>% 
  transformColums("interjection") 

def.formula <- as.formula("targetHamSpam~.")

#TSMS
{
  cat("Starting ",model, extension,"...\n")
  set.seed(100)
  data <- subset(dtm.cutoff, extension == extension)
  index <- caret::createDataPartition(data$targetHamSpam, p = .75, list = FALSE)
  train <- data[index, ]
  test <-  data[-index, ]
  
  rec <- recipes::recipe(formula = def.formula, data = train) %>%
    step_zv(all_predictors()) # %>% #remove zero variance
    # step_nzv(all_predictors()) %>% #remove near-zero variance
    # step_corr(all_predictors()) #remove high correlation filter.
  
  trControl <- caret::trainControl(method = "cv", #use cross-validation
                                   number = k, #divide cross-validation into 10 folds
                                   search = "random", #"grid"
                                   savePredictions = "final", #save predictions of best model.
                                   classProbs = TRUE, #save probabilities obtained for the best model.
                                   summaryFunction = defaultSummary, #use defaultSummary function (only computes Accuracy and Kappa values)
                                   allowParallel = TRUE, #execute in parallel.
                                   seeds = set.seed(100)
  )
  cat("Training ",model,extension,"...\n")
  trained <- caret::train(rec,
                          data = train,
                          method = model,
                          trControl = trControl,
                          metric = "Kappa")
  
  cat("Testing ",model, extension,"...\n")
  cf <- caret::confusionMatrix(
    predict(trained, newdata = test, type = "raw"),
    reference = test$targetHamSpam,
    positive = "spam"
  )
  
  cat("Finished ",model, extension,"...\n")
  saveRDS(trained, file = fileTrain)
  saveRDS(cf, file = fileTest)
}
}
}
}
