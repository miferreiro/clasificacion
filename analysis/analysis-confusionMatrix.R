source("functions/draw_confusion_matrix.R")
# typeDimensionality <- "ig"
# model <- "rf"
# class <- "eml"
# file <- paste("results/",class,"-tokens-", typeDimensionality, "-", model, "-test",".rds", sep="")
# testFile <- readRDS(file = file)


k <- 3
print(k)
typeDimensionality <- "ig"
model <- "nb"
source("functions/draw_confusion_matrix.R")
file <- paste("results/tsms-tokens-", typeDimensionality, "-", model, "-test-", k, "-folds.rds", sep="")
testFile <- readRDS(file = file)

class <- "tsms"

if (model == "nb") {
  draw_confusion_matrix(testFile, paste("Confusion Matrix Naive-Bayes",class,sep = " "))
}else {
  if (model == "svmLinear") {
    draw_confusion_matrix(testFile, paste("Confusion Matrix SVM",class,sep = " "))
  }else{
    draw_confusion_matrix(testFile, paste("Confusion Matrix Random Forest",class,sep = " "))
  }
}
