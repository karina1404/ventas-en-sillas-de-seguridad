# Cargar librerías necesarias
library(neuralnet)
library(caret)
library(ggplot2)

# Cargar datos
datos <- read.csv("Child Car Safety Seats.csv", sep = ",", header = TRUE)

# Convertir variables categóricas a factores y luego a numérico para la normalización
datos$shelveLoc <- as.numeric(factor(datos$shelveLoc))
datos$urban <- as.numeric(factor(datos$urban))
datos$us <- as.numeric(factor(datos$us))

# Separar variables numéricas y categóricas
datos_numericos <- datos[, sapply(datos, is.numeric)]
datos_categoricos <- datos[, !sapply(datos, is.numeric)]

# Normalizar los datos numéricos
maxs <- apply(datos_numericos, 2, max)
mins <- apply(datos_numericos, 2, min)
datos_numericos_n <- as.data.frame(scale(datos_numericos, center = mins, scale = maxs - mins))

# Combinar las variables normalizadas con las categóricas (sin modificar)
datos_n <- cbind(datos_numericos_n, datos_categoricos)

# Verificar la estructura de los datos escalados
str(datos_n)

# Dividir los datos en entrenamiento (80%) y prueba (20%)
set.seed(123)
particiones <- createDataPartition(datos_n$sales, p = 0.8, list = FALSE)
datos_entrenamiento <- datos_n[particiones, ]
datos_prueba <- datos_n[-particiones, ]

# Convertir variables categóricas a factores y luego a numérico para datos_entrenamiento
datos_entrenamiento$shelveLoc <- as.numeric(factor(datos_entrenamiento$shelveLoc))
datos_entrenamiento$urban <- as.numeric(factor(datos_entrenamiento$urban))
datos_entrenamiento$us <- as.numeric(factor(datos_entrenamiento$us))

# Definir la fórmula
formula <- sales ~ comp_price + income + advertising + population + price + shelveLoc + age + education + urban + us

# Crear la red neuronal con una capa oculta y 5 neuronas
set.seed(123)
nn1 <- neuralnet(formula, data = datos_entrenamiento, hidden = c(5), linear.output = TRUE)

# Visualizar la red neuronal
plot(nn1)

# Hacer predicciones
predicciones1 <- compute(nn1, datos_prueba[, names(datos_prueba) != "sales"])$net.result

# Desnormalizar las predicciones para comparar con los valores originales
predicciones1 <- predicciones1 * (max(datos$sales) - min(datos$sales)) + min(datos$sales)

# Calcular el R^2 para evaluar el primer modelo
valores_reales1 <- datos_prueba$sales * (max(datos$sales) - min(datos$sales)) + min(datos$sales)
SSE1 <- sum((valores_reales1 - predicciones1)^2)
SST1 <- sum((valores_reales1 - mean(valores_reales1))^2)
R2_1 <- 1 - SSE1/SST1

print(paste("R^2 del primer modelo: ", R2_1))

# Calcular el MSE para el primer modelo
MSE1 <- mean((valores_reales1 - predicciones1)^2)
print(paste("R^2 del primer modelo: ", R2_1))
print(paste("MSE del primer modelo: ", MSE1))

# Crear la red neuronal con dos capas ocultas y 10 neuronas en la primera capa y 5 en la segunda
set.seed(123)
nn2 <- neuralnet(formula, data = datos_entrenamiento, hidden = c(10, 5), linear.output = TRUE)

# Visualizar la red neuronal
plot(nn2)

# Hacer predicciones
predicciones2 <- compute(nn2, datos_prueba[, names(datos_prueba) != "sales"])$net.result

# Desnormalizar las predicciones para comparar con los valores originales
predicciones2 <- predicciones2 * (max(datos$sales) - min(datos$sales)) + min(datos$sales)

# Calcular el R^2 para evaluar el segundo modelo
valores_reales2 <- datos_prueba$sales * (max(datos$sales) - min(datos$sales)) + min(datos$sales)
SSE2 <- sum((valores_reales2 - predicciones2)^2)
SST2 <- sum((valores_reales2 - mean(valores_reales2))^2)
R2_2 <- 1 - SSE2/SST2

print(paste("R^2 del segundo modelo: ", R2_2))

# Calcular el MSE para el segundo modelo
MSE2 <- mean((valores_reales2 - predicciones2)^2)
print(paste("R^2 del segundo modelo: ", R2_2))
print(paste("MSE del segundo modelo: ", MSE2))

# Comparación de R^2 de ambos modelos
print(paste("R^2 del primer modelo (5 neuronas, 1 capa): ", R2_1))
print(paste("R^2 del segundo modelo (10 neuronas, 2 capas): ", R2_2))

if(R2_2 > R2_1) {
  print("El segundo modelo es mejor.")
} else {
  print("El primer modelo es mejor.")
}

# Comparación de predicciones vs datos reales
datos_prueba_real <- datos_prueba$sales * (max(datos$sales) - min(datos$sales)) + min(datos$sales)

ggplot() +
  geom_point(aes(x = datos_prueba_real, y = predicciones1), color = 'blue', alpha = 0.5) +
  geom_point(aes(x = datos_prueba_real, y = predicciones2), color = 'red', alpha = 0.5) +
  labs(title = "Comparación de Predicciones de los Modelos", x = "Valores Reales", y = "Predicciones") +
  theme_minimal() +
  scale_color_manual(values = c('blue', 'red'), labels = c('Modelo 1', 'Modelo 2')) +
  guides(color = guide_legend(title = "Modelos"))

