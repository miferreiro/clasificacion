library(tm);library(pipeR);library(tokenizers);library("FSelector")
tsmsDF <- read.csv(file = "csvs/output_sms_last.csv", header = TRUE, 
                    sep = ";", dec = ".", fill = FALSE, stringsAsFactors = FALSE)

tsms.data <- tsmsDF$data
tsms.corpus <- VCorpus(VectorSource(tsms.data ))
tsms.corpus <- tm_map(tsms.corpus, removePunctuation)

meta(tsms.corpus, tag = "target") <- tsmsDF$target
#Creating Term-Document Matrices
tsms.dtm <- DocumentTermMatrix(tsms.corpus)
# inspect(tsms.dtm)
# words_frequency <- colSums(as.matrix(tsms.dtm)) 
# tsms.tfid <- DocumentTermMatrix(tsms.corpus, control = list(weighting = weightTfIdf))
# inspect(tsms.tfid)

tsms.data.frame.dtm <- as.data.frame(as.matrix(tsms.dtm))
tsms.data.frame.dtm[,"target"] <- tsmsDF$target
tsms.data.frame.dtm[,"target"] <- as.factor(tsms.data.frame.dtm[,"target"])

tsms.chi <- chi.squared(target~., tsms.data.frame.dtm)
# Error in .jarray(x) : java.lang.OutOfMemoryError: Java heap space
tsms.ig <- information.gain(target~., tsms.data.frame.dtm)
# Error in .jarray(x) : java.lang.OutOfMemoryError: Java heap space

