information_gain <- function(target, data) 
{
  i <- which(names(data) == target)
  attr_entropies = sapply(data, entropyHelper)
  class_entropy = attr_entropies[i]
  attr_entropies = attr_entropies[-i]
  joint_entropies = sapply(data[-i], function(t) {
    entropyHelper(data.frame(cbind(data[[i]], t)))
  })
  results = class_entropy + attr_entropies - joint_entropies
  
  attr_names = dimnames(data)[[2]][-1]
  return(data.frame(attr_importance = results, row.names = attr_names))
}

entropyHelper <- function(x) 
{
  return(entropy(table(x, useNA = "always")))
}

entropy <- function(freqs, method = "ML")
{
  freqs = freqs/sum(freqs)
  if (method == "ML")  H = -sum(ifelse(freqs > 0, freqs * log(freqs), 0))
}
