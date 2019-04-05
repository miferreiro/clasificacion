library(tm);library(pipeR);library(tokenizers);library("FSelector")
ytbidDF <- read.csv(file = "csvs/output_ytbid.csv", header = TRUE, 
                          sep = ";", dec = ".", fill = FALSE, stringsAsFactors = FALSE)
  
ytbid.data <- ytbidDF$data
ytbid.corpus <- VCorpus(VectorSource(ytbid.data))
ytbid.corpus <- tm_map(ytbid.corpus, removePunctuation)

meta(ytbid.corpus, tag = "target") <- ytbidDF$target
#Creating Term-Document Matrices
ytbid.dtm <- DocumentTermMatrix(ytbid.corpus)
# inspect(ytbid.dtm)
# words_frequency <- colSums(as.matrix(ytbid.dtm)) 
# ytbid.tfid <- DocumentTermMatrix(ytbid.corpus, control = list(weighting = weightTfIdf))
# inspect(ytbid.tfid)

ytbid.data.frame.dtm <- as.data.frame(as.matrix(ytbid.dtm))
ytbid.data.frame.dtm[,"target"] <- ytbidDF$target
ytbid.data.frame.dtm[,"target"] <- as.factor(ytbid.data.frame.dtm[,"target"])

ytbid.chi <- chi.squared(target~., ytbid.data.frame.dtm)
ytbid.ig <- information.gain(target~., ytbid.data.frame.dtm)

