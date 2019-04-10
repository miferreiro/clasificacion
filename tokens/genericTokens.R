library(tm);library(pipeR);library(tokenizers);library(FSelector)
source("functions/IG-chi2.R")
genericTokens <- function(DF, methodCHI2ORIG = T, fileSave = NULL) {
  # DF <- read.csv(file = file, header = TRUE, 
  #                    sep = ";", dec = ".", fill = FALSE, stringsAsFactors = FALSE)
  
  corpus <- VCorpus(VectorSource(DF$data))
  corpus <- tm_map(corpus, removePunctuation)
  
  #Creating Term-Document Matrices
  dtm <- DocumentTermMatrix(corpus)
  data.frame.dtm <- as.data.frame(as.matrix(dtm))
  data.frame.dtm[,"target"] <- as.factor(DF$target)
  
  if (methodCHI2ORIG) {
    toret <- chi_squared(target~., data.frame.dtm)
  } else {
    toret <- information.gain(target~., tsms.data.frame.dtm)
  }
  if (!is.null(fileSave)) {
    saveRDS(toret, file = fileSave)
  }
  
  return(toret)
}
