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
