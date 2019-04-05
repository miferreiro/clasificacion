library(tm);library(pipeR);library(tokenizers);library("FSelector")
emlDF <- read.csv(file = "csvs/output_sms_last.csv", header = TRUE, 
                   sep = ";", dec = ".", fill = FALSE, stringsAsFactors = FALSE)

eml.data <- emlDF$data
eml.corpus <- VCorpus(VectorSource(eml.data))
eml.corpus <- tm_map(eml.corpus, removePunctuation)

meta(eml.corpus, tag = "target") <- emlDF$target
#Creating Term-Document Matrices
eml.dtm <- DocumentTermMatrix(eml.corpus)
# inspect(eml.dtm)
# words_frequency <- colSums(as.matrix(eml.dtm)) 
# tsms.tfid <- DocumentTermMatrix(eml.corpus, control = list(weighting = weightTfIdf))
# inspect(eml.tfid)

eml.data.frame.dtm <- as.data.frame(as.matrix(eml.dtm))
eml.data.frame.dtm[,"target"] <- emlDF$target
eml.data.frame.dtm[,"target"] <- as.factor(eml.data.frame.dtm[,"target"])

eml.chi <- chi.squared(target~., eml.data.frame.dtm)
# Error in .jarray(x) : java.lang.OutOfMemoryError: Java heap space
eml.ig <- information.gain(target~., eml.data.frame.dtm)
# Error in .jarray(x) : java.lang.OutOfMemoryError: Java heap space

