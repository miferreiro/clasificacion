source("functions/draw_confusion_matrix.R")
file <- "results/all-tokens-svm-test.rds"
testFile <- readRDS(file = file)

class <- "All"
# draw_confusion_matrix(testFile, paste("Confusion Matrix Naive-Bayes",class,sep = " "))
 draw_confusion_matrix(testFile, paste("Confusion Matrix SVM",class,sep = " "))
# draw_confusion_matrix(testFile, paste("Confusion Matrix Random Forest",class,sep = " "))

