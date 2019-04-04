rm(list = ls())
setwd("C:/Users/Miguel/Desktop/clasificacion")
library(rlist)
source("transformarRDataPruebas.R")
RData <- readRDS("dataFrameAllSynsets.RData")
RData <- transformarRDataPruebas(RData)

numInst <- nrow(RData)
types <- list()
percentHamTsms <- floor(nrow(subset(RData, extension == "tsms" & target == "ham")) / nrow(subset(RData, extension == "tsms")) * 100) / 100
percentHamEml <- floor(nrow(subset(RData, extension == "eml" & target == "ham")) / nrow(subset(RData, extension == "eml")) * 100) / 100
percentHamTwtid <- floor(nrow(subset(RData, extension == "twtid" & target == "ham")) / nrow(subset(RData, extension == "twtid")) * 100) / 100
percentHamYtbid <- floor(nrow(subset(RData, extension == "ytbid" & target == "ham")) / nrow(subset(RData, extension == "ytbid")) * 100) / 100
percentHamWarc <- floor(nrow(subset(RData, extension == "warc" & target == "ham")) / nrow(subset(RData, extension == "warc")) * 100) / 100

numExtTypes <- 5
namesExtTypes <- c("tsms", "eml", "twtid", "ytbid", "warc")
types <- list(list(ham = percentHamTsms, spam = 1 - percentHamTsms),
              list(ham = percentHamEml, spam = 1 - percentHamEml),
              list(ham = percentHamTwtid, spam = 1 - percentHamTwtid),
              list(ham = percentHamYtbid, spam = 1 - percentHamYtbid),
              list(ham = percentHamWarc, spam = 1 - percentHamWarc))
names(types) <- namesExtTypes
rm(percentHamTsms)
rm(percentHamEml)
rm(percentHamTwtid)
rm(percentHamYtbid)
rm(percentHamWarc)

k <- 2
numInstForExtType <- 100

numInstExtTypeInGroup <- floor(numInstForExtType / k)
#################################################
numTypes <- list(list(numHam = round(numInstExtTypeInGroup * types$tsms$ham), numSpam = round(numInstExtTypeInGroup * types$tsms$spam)),
              list(numHam = round(numInstExtTypeInGroup * types$eml$ham), numSpam = round(numInstExtTypeInGroup * types$eml$spam)),
              list(numHam = round(numInstExtTypeInGroup * types$twtid$ham), numSpam = round(numInstExtTypeInGroup * types$twtid$spam)),
              list(numHam = round(numInstExtTypeInGroup * types$ytbid$ham), numSpam = round(numInstExtTypeInGroup * types$ytbid$spam)),
              list(numHam = round(numInstExtTypeInGroup * types$warc$ham), numSpam = round(numInstExtTypeInGroup * types$warc$spam)))
names(numTypes) <- namesExtTypes

for (group in 1:length(namesExtTypes)) {
  vectorSample <- sample(numInstExtTypeInGroup * k)
  for (numGroup in 1:k) {
    vectorSampleGroup <- vectorSample[1:numInstExtTypeInGroup]

    ext <- names(numTypes[group])
    
    spam <- rownames(subset(RData, extension %in% ext & target == "spam"))
    
    for (i in 1:numTypes[[namesExtTypes[group]]]$numSpam) {
      
      x <- vectorSampleGroup[i]
      RData[spam[x],"group"] <- numGroup  
    }   
    
    ham <- rownames(subset(RData, extension %in% ext & target == "ham"))
    
    for (y in ((numTypes[[namesExtTypes[group]]]$numSpam + 1):length(vectorSampleGroup))) {
      x <- vectorSampleGroup[y]
      RData[ham[x],"group"] <- numGroup   
    }
    vectorSample <- vectorSample[(numInstExtTypeInGroup + 1):length(vectorSample)]
  }
}
rm(i)
rm(group)
rm(numGroup)
rm(x)
rm(y)
rm(ham)
rm(spam)
rm(vectorSample)
rm(vectorSampleGroup)
rm(ext)
dimnames(RData)[[2]][31:612] <- paste("V",31:612, sep = "")
def.formula <- as.formula(paste("target",
                          paste(#"length","length_after_stopwords",
                          # "userName","hashtag","URLs", "Emoticon", "Emojis",
                          # "language","langpropname","interjection",
                           paste(names(RData)[31:300], collapse = "+")
                          ,sep = "+",collapse = "+"),sep = "~"))
# def.formula <- as.formula(paste("target ",
#                           paste("length", "length_after_stopwords",
#                           "userName", "hashtag", "URLs", "Emoticon", "Emojis",
#                           "language", "langpropname", "interjection"
#                           ,sep = "+",collapse = "+"),sep = "~"))

RData <- transformColums(RData, "userName")
RData <- transformColums(RData, "hashtag")
RData <- transformColums(RData, "URLs")
RData <- transformColums(RData, "Emoticon")
RData <- transformColums(RData, "Emojis")
RData <- transformColums(RData, "langpropname")
RData <- transformColums(RData, "interjection")


library(caret)
library(klaR)
library(e1071)

typePredictions <- list()

for (type in namesExtTypes) {
  predictions <- NULL
  for (i in 1:k) {
    data1 <- subset(RData,group != i & !is.na(group) & extension == type)
    # model <- NaiveBayes(def.formula,  data = data1,fL = 1)
    model <- naiveBayes(formula = def.formula, useKernel = T ,data = RData,subset = group != i & !is.na(group) & extension == type)
    # model <- caret::train(x = def.formula, data = RData, subset = group != i & !is.na(group) & extension == type, method ="nb")
    
    predictions <- rbind(predictions, predict(model, 
                                              subset(RData, group == i & 
                                                            !is.na(group) & 
                                                            extension == type)))
  }
  typePredictions <- list.append(typePredictions, predictions)
}

names(typePredictions) <- namesExtTypes

for (i in names(typePredictions)) {
  cat("Tipos:" , i, "\n")
  
  print(table(subset(RData, extension == i & !is.na(group))$target, typePredictions[[i]]))
  cat("\n")
}


















