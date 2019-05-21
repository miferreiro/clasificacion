source("functions/draw_confusion_matrix.R")
# typeDimensionality <- "ig"
# # typeDimensionality <- "chi"
# model <- "svmLinear"
# class <- "all"
# file <- paste("results/",class,"-tokens-", typeDimensionality, "-", model, "-test",".rds", sep="")
# testFile <- readRDS(file = file)


class <- "eml"
k <- 13
print(k)
typeDimensionality <- "chi"
model <- "rf"
source("functions/draw_confusion_matrix.R")
file <- paste("results/", class, "-tokens-", typeDimensionality, "-", model, "-test-", k, "-folds.rds", sep="")
testFile <- readRDS(file = file)



if (model == "nb") {
  draw_confusion_matrix(testFile, paste("Confusion Matrix Naive-Bayes",class,sep = " "))
}else {
  if (model == "svmLinear") {
    draw_confusion_matrix(testFile, paste("Confusion Matrix SVM",class,sep = " "))
  }else{
    draw_confusion_matrix(testFile, paste("Confusion Matrix Random Forest",class,sep = " "))
  }
}
