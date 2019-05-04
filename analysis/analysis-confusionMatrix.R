source("functions/draw_confusion_matrix.R")
file = "results/tsms-tokens-rf-test.rds"
test <- readRDS(file = file)


draw_confusion_matrix(test, "Confusion Matrix Random Forest Tsms")
