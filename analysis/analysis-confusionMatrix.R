source("functions/draw_confusion_matrix.R")
typeDimensionality <- "ig"
# typeDimensionality <- "chi"
model <- "nb"
class <- "ytbid"
file <- paste("results/",class,"-tokens-", typeDimensionality, "-", model, "-test",".rds", sep="")
testFile <- readRDS(file = file)



# k <- 13
# print(k)
# typeDimensionality <- "ig"
# model <- "nb"
# source("functions/draw_confusion_matrix.R")
# file <- paste("results/tsms-tokens-", typeDimensionality, "-", model, "-test-", k, "-folds.rds", sep="")
# testFile <- readRDS(file = file)
# 
# class <- "tsms"

if (model == "nb") {
  draw_confusion_matrix(testFile, paste("Confusion Matrix Naive-Bayes",class,sep = " "))
}else {
  if (model == "svmLinear") {
    draw_confusion_matrix(testFile, paste("Confusion Matrix SVM",class,sep = " "))
  }else{
    draw_confusion_matrix(testFile, paste("Confusion Matrix Random Forest",class,sep = " "))
  }
}
