information_gain <- function(formula, data, unit = "log") 
{
  information.gain.body(formula, data, type = "infogain", unit)
}

information.gain.body <- function (formula, data, type = c("infogain", "gainratio", "symuncert"), 
                                   unit) 
{
  type = match.arg(type)
  new_data = get.data.frame.from.formula(formula, data)
  attr_entropies = sapply(new_data, entropyHelper, unit)
  class_entropy = attr_entropies[1]
  attr_entropies = attr_entropies[-1]
  joint_entropies = sapply(new_data[-1], function(t) {
    entropyHelper(data.frame(cbind(new_data[[1]], t)), unit)
  })
  results = class_entropy + attr_entropies - joint_entropies
  if (type == "gainratio") {
    results = ifelse(attr_entropies == 0, 0, results/attr_entropies)
  }
  else if (type == "symuncert") {
    results = 2 * results/(attr_entropies + class_entropy)
  }
  attr_names = dimnames(new_data)[[2]][-1]
  return(data.frame(attr_importance = results, row.names = attr_names))
}

entropyHelper <- function (x, unit = "log") 
{
  return(entropy(table(x, useNA = "always"), unit = unit))
}

entropy <- function (y, lambda.freqs, method = c("ML", "MM", "Jeffreys", 
                                                 "Laplace", "SG", "minimax", "CS", "NSB", "shrink"), unit = c("log", 
                                                                                                              "log2", "log10"), verbose = TRUE, ...) 
{
  method = match.arg(method)
  if (method == "ML") 
    H = entropy.empirical(y, unit = unit)
  if (method == "MM") 
    H = entropy.MillerMadow(y, unit = unit)
  if (method == "NSB") 
    H = entropy.NSB(y, unit = unit, ...)
  if (method == "CS") 
    H = entropy.ChaoShen(y, unit = unit)
  if (method == "Jeffreys") 
    H = entropy.Dirichlet(y, a = 1/2, unit = unit)
  if (method == "Laplace") 
    H = entropy.Dirichlet(y, a = 1, unit = unit)
  if (method == "SG") 
    H = entropy.Dirichlet(y, a = 1/length(y), unit = unit)
  if (method == "minimax") 
    H = entropy.Dirichlet(y, a = sqrt(sum(y))/length(y), 
                          unit = unit)
  if (method == "shrink") 
    H = entropy.shrink(y, lambda.freqs = lambda.freqs, unit = unit, 
                       verbose = verbose)
  return(H)
}