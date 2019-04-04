transformarRDataPruebas = function(RData) {
  
  RData <- rbind(RData,RData)
  RData <- rbind(RData,RData)
  RData <- rbind(RData,RData)
  
  x <- 1
  while (x  < (dim(RData)[1])) {
    RData[x, ]$target <- "spam"
    x <- x + 3
  }
  
  for (i in 1:floor((dim(RData)[1])/5)) {
    RData[i, ]$extension <- "tsms"
  }
  
  for (i in (dim(RData)[1]/5):floor( 2 * (dim(RData)[1])/5)) {
    RData[i, ]$extension <- "eml"
  }
  
  for (i in (2 * dim(RData)[1]/5):floor( 3 * (dim(RData)[1])/5)) {
    RData[i, ]$extension <- "twtid"
  }
  
  for (i in (3 * dim(RData)[1]/5):floor( 4 * (dim(RData)[1])/5)) {
    RData[i, ]$extension <- "ytbid"
  }
  
  for (i in (4 * dim(RData)[1]/5):floor( 5 * (dim(RData)[1])/5)) {
    RData[i, ]$extension <- "warc"
  }
  
  RData$target <- as.factor(RData$target)
  RData$extension <- as.factor(RData$extension)
  RData$language <- as.factor(RData$language)
  RData$length <- as.numeric(RData$length)
  RData$length_after_stopwords <- as.numeric(RData$length_after_stopwords)
  RData[is.na(RData)] <- 0
  return(RData)
}


transformColums <- function(RData, column) {
  RData[, column]  <- sapply(RData[, column] , function(col) {
    if (is.null(col) |
        is.na(col) |
        col == "") {
      col <- 0
      
    } else {
      col <- 1
    }
  })
  
  RData[, column] <- as.numeric(RData[, column])
  return(RData)
}
