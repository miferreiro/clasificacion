chi_squared <- function(target, new_data)
{
  i <- which(names(new_data) == target)
  class_data = new_data[[i]]
  new_data = new_data[-i]
  results = sapply(new_data, function(w) {
    cont = table(class_data, w)
    row_sums = apply(cont, 1, sum)
    col_sums = apply(cont, 2, sum)
    all_sum = sum(col_sums)
    expected_matrix = t(as.matrix(col_sums) %*% 
                          t(as.matrix(row_sums)))/all_sum
    chis = sum((cont - expected_matrix)^2/expected_matrix)
    if (chis == 0 || length(col_sums) < 2 || length(row_sums) <
        2) {
      return(0)
    }
    else {
      return(sqrt(chis/(all_sum * min(length(col_sums) -
                                        1, length(row_sums) - 1))))
    }
  })
  attr_names = dimnames(new_data)[[2]]
  return(data.frame(attr_importance = results, row.names = attr_names))
}

