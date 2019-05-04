file = "results/tsms-ig.rds"
chi2 <- readRDS(file = file)

plot(type = "l", density(chi2$attr_importance), main = "Tsms", xlab = paste("N =  ", length(ig$attr_importance)) , ylab = "Densidad")
punto.corte.chi2 = 0.001
abline(v = punto.corte.chi2, col = "blue", lty = 2)

# text(x = 9.5,y =  0.1,"1000 instancias")
# text(x = 11.5,y = 0.05,"100 instancias")

densidade <- density(chi2$attr_importance)
minimo <- min(which(densidade$x >= punto.corte.chi2))
maximo <- max(which(densidade$x <= 0.08)) 
with(densidade, polygon(x = c(x[c(minimo, minimo:maximo, maximo)]), y = c(0,y[minimo:maximo], 0), col = rgb(0, 206/255, 209/255, 0.5)))

