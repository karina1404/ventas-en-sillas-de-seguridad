library("randomForest")
library("MASS")
datos = read.csv('C:/Users/Jonathan/OneDrive/Documentos/UNAB/MINERIA DE DATOS/SOLEMNE 2/Child Car Safety Seats.csv',sep=',',header=T)

set.seed(42)
particiones = sample(1:nrow(datos), size = 0.8 * nrow(datos))
datos_entrenamiento = datos[particiones, ] 
datos_prueba = datos[-particiones, ] 

# Entrenamos el modelo
modelo = randomForest(sales~., data = datos_entrenamiento, ntree = 100, mtry = 2, importance = TRUE)  


# Gráfico que muestra la importancia de las variables:
varImpPlot(modelo) 

# %IncMSE (Porcentaje de incremento en Error Cuadrático Medio):
# Un valor alto indica que la variable es crucial para la precisión del modelo. 

# IncNodePurity (Increase in Node Purity)
# Un mayor valor para una variable indica que esa variable es útil para crear 
# nodos más puros, dando como resultado una mejor agrupación para obtener el valor objetivo. 


# Validamos el modelo
predicciones = predict(modelo, datos_prueba)

# Un modelo en un problema de regresión puede ser evaluado con: 
# MSE: indica la discrepancia entre los valores observados y los obtenidos 
# con el modelo. Mientras menor sea, mejor será el modelo. 

mse = mean((datos_prueba$sales - predicciones)^2)

# r2: 

# Calcular la suma total de cuadrados (TSS)
tss = sum((datos_prueba$sales - mean(datos_prueba$sales))^2)
# Calcular la suma de los cuadrados de los residuos (RSS)
rss = sum((datos_prueba$sales - predicciones)^2)
# Calcular el coeficiente de determinación (R^2)
r2 = 1 - (rss / tss)

# Un r2 más cercano a 1 indica un mejor ajuste

# 2. Problema Regresión un sólo árbol --------------------------------------

# Comparemos con un sólo árbol
library(rpart)
modelo = rpart(sales ~., data = datos_entrenamiento, method = "anova", cp=0.001)
predicciones = predict(modelo, datos_prueba)
tss = sum((datos_prueba$sales - mean(datos_prueba$sales))^2)
rss = sum((datos_prueba$sales - predicciones)^2)
r2 = 1 - (rss / tss)







