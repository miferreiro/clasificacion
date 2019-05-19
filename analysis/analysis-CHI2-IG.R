library(dplyr)
file = "results/eml-chi.rds"
chi2 <- readRDS(file = file)
chi2 <- arrange(chi2,-attr_importance)
plot(type = "l", density(chi2$attr_importance), xlim=c(0,0.3),main = "Eml Chi square", xlab = "importance", ylab = "Density")
cut.off.chi2 = 2000
abline(v = chi2$attr_importance[cut.off.chi2], col = "red", lty = 2)

density <- density(chi2$attr_importance)
min <- min(which(density$x >= chi2$attr_importance[cut.off.chi2]))
max <- max(which(density$x <= chi2$attr_importance[1]))
with(density, polygon(x = c(x[c(min, min:max, max)]), y = c(0,y[min:max], 0), col = rgb(0, 206/255, 209/255, 0.5)))



