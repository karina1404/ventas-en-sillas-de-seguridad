# Instalar y cargar bibliotecas necesarias
install.packages("e1071")
install.packages("tidyverse")
install.packages("caret")
install.packages("kernlab")

library(e1071)
library(tidyverse)
library(caret)
library(kernlab)

#conjunto de datos de asientos de seguridad para niños en automovil (Child Car Safety Seats)
#sales (ventas): Ventas unitarias en miles
#comp_price (precio_comp): Precio cobrado por el competidor en cada ubicación
#income (ingreso):	Nivel de ingresos de la comunidad en miles de dólares
#adversiting (publicidad):	Presupuesto publicitario local en cada ubicación en miles de dólares
#population (población):	El pop regional en miles
#price (precio):	Precio de las sillas de coche en cada sitio
#shelveloc (estanteLoc):	Malo, Bueno o Medio indica la calidad de la ubicación de las estanterías.
#age (edad):	Nivel de edad de la población
#education (educación):	Nivel educativo en la ubicación
#urban (urbano):	Los niveles de factor 'Sí' o 'No' se utilizan para indicar si una tienda está en una ubicación urbana o rural.
#us (a nosotros):	Los niveles de factor 'Sí' o 'No' se utilizan para indicar si una tienda está en Estados Unidos o no.


# Leer los datos
data <- read.csv("D:/Descargas/Child Car Safety Seats.csv")

# Revisar dimensiones, estructura y resumen estadístico de los datos
dim(data)
str(data)
summary(data)

#revisar datos faltantes
any(is.na(data))
# en este caso no hay datos faltantes

# Visualización: Usa gráficos para explorar la relación entre las variables predictoras y las ventas.
ggplot(data, aes(x = income, y = sales)) + 
  geom_point() + 
  labs(title = "Ventas vs. Ingreso")

ggplot(data, aes(x = comp_price, y = sales)) + 
  geom_point() + 
  labs(title = "Ventas vs. Precio Competidor")

ggplot(data, aes(x = advertising, y = sales)) + 
  geom_point() + 
  labs(title = "Ventas vs. Publicidad")

ggplot(data, aes(x = population, y = sales)) + 
  geom_point() + 
  labs(title = "Ventas vs. Población")

ggplot(data, aes(x = price, y = sales)) + 
  geom_point() + 
  labs(title = "Ventas vs. Precio")

ggplot(data, aes(x = age, y = sales)) + 
  geom_point() + 
  labs(title = "Ventas vs. Edad")

# Gráfico de Ventas vs. Ubicación de estanterías
ggplot(data, aes(x = shelveLoc, y = sales)) + 
  geom_boxplot() + 
  labs(title = "Ventas vs. Ubicación de estanterías")

# Gráfico de Ventas vs. Nivel educativo
ggplot(data, aes(x = education, y = sales)) + 
  geom_point() + 
  labs(title = "Ventas vs. Nivel educativo")

# Gráfico de Ventas vs. Ubicación urbana
ggplot(data, aes(x = urban, y = sales)) + 
  geom_boxplot() + 
  labs(title = "Ventas vs. Ubicación urbana")

# Gráfico de Ventas vs. Ubicación en EE.UU.
ggplot(data, aes(x = us, y = sales)) + 
  geom_boxplot() + 
  labs(title = "Ventas vs. Ubicación en EE.UU.")

# Convertir variables categóricas a factores
data$shelveLoc <- as.factor(data$shelveLoc)
data$urban <- as.factor(data$urban)
data$us <- as.factor(data$us)

# Dividir los datos en conjuntos de entrenamiento y prueba
set.seed(123)
#createDataPartition() es una función del paquete caret en R 
#que se utiliza para dividir un conjunto de datos en subconjuntos,
#generalmente para separar los datos en conjuntos de entrenamiento y prueba
#en este caso se entrena el 80% de los datos.
training_samples <- createDataPartition(data$sales, p = 0.8, list = FALSE)
# Aquí se está creando el conjunto de entrenamiento.
#se usa los índices almacenados en training_samples,
#se selecciona las filas correspondientes del conjunto de datos original data
#para formar el conjunto de entrenamiento train_data.
train_data <- data[training_samples, ]
#Aquí se está creando el conjunto de prueba.
#se utiliza los índices negativos [-training_samples] 
#para seleccionar las filas que no fueron seleccionadas para el conjunto de entrenamiento,
#y así crear el conjunto de prueba test_data.
test_data <- data[-training_samples, ]

# Convertir factores a numéricos y normalizar las variables predictoras
convert_to_numeric <- function(df) {
  df_numeric <- df
  for (col in names(df)) {
    if (is.factor(df[[col]])) {
      df_numeric[[col]] <- as.numeric(df[[col]])
    }
  }
  return(df_numeric)
}

# Aplicar la conversión a los datos de entrenamiento y prueba
train_data_numeric <- convert_to_numeric(train_data)
test_data_numeric <- convert_to_numeric(test_data)

# Identificar columnas predictoras (todas excepto 'sales')
predictor_cols <- setdiff(names(train_data_numeric), "sales")

# Normalizar las variables predictoras
train_data_scaled <- train_data_numeric
train_data_scaled[, predictor_cols] <- scale(train_data_numeric[, predictor_cols])

test_data_scaled <- test_data_numeric
test_data_scaled[, predictor_cols] <- scale(test_data_numeric[, predictor_cols])

# Entrenar el modelo SVM lineal
modelo_SVM <- svm(sales ~ ., data = train_data, kernel = "linear", cost = 1, scale = TRUE)

# Realizar predicciones con el modelo
predicciones <- predict(modelo_SVM, test_data)

# Calcular MSE y RMSE
mse <- mean((predicciones - test_data$sales)^2)
rmse <- sqrt(mse)
print(paste("Raíz del error cuadrático medio (RMSE): ", rmse))

# Calcular R-cuadrado
suma_total_cuadrados <- sum((test_data$sales - mean(test_data$sales))^2)
suma_cuadrado_residual <- sum((predicciones - test_data$sales)^2)
r_cuadrado <- 1 - (suma_cuadrado_residual / suma_total_cuadrados)
print(paste("R-cuadrado: ", r_cuadrado))

#RMSE de 1.09: La desviación promedio de las predicciones 
#con respecto a los valores reales es de 1.09 unidades.
#R-cuadrado de 0.852: El modelo explica aproximadamente el 85.2% de la variabilidad en las ventas,
#lo que indica una buena capacidad predictiva.
#Ambos resultados sugieren que el modelo está funcionando bastante bien,
#captura una buena parte de la variabilidad en las ventas
#y con errores de predicción relativamente bajos.

# Definir la matriz de combinaciones de hiperparámetros
matriz_combinaciones <- expand.grid(C = 2^(-5:2), sigma = 2^(-5:2))

# Entrenar con validación cruzada usando svmRadial
modelo_ajustado <- train(sales ~ ., data = train_data_scaled, method = "svmRadial",
                         tuneGrid = matriz_combinaciones, trControl = trainControl(method = "cv"))

# Ver el mejor modelo
print(modelo_ajustado)

#Se utilizó RMSE para seleccionar el modelo óptimo utilizando el valor más pequeño.
#Los valores finales utilizados para el modelo fueron sigma = 0,0625 y C = 4.

# Realizar predicciones con el modelo ajustado
predicciones_ajustadas <- predict(modelo_ajustado, newdata = test_data_scaled)

# Calcular MSE y RMSE para el modelo ajustado
mse_ajustado <- mean((predicciones_ajustadas - test_data_scaled$sales)^2)
rmse_ajustado <- sqrt(mse_ajustado)
print(paste("Raíz del error cuadrático medio (RMSE) del modelo ajustado: ", rmse_ajustado))

# Calcular R-cuadrado para el modelo ajustado
suma_total_cuadrados_ajustado <- sum((test_data_scaled$sales - mean(test_data_scaled$sales))^2)
suma_cuadrado_residual_ajustado <- sum((predicciones_ajustadas - test_data_scaled$sales)^2)
r_cuadrado_ajustado <- 1 - (suma_cuadrado_residual_ajustado / suma_total_cuadrados_ajustado)
print(paste("R-cuadrado del modelo ajustado: ", r_cuadrado_ajustado))

#RMSE de 1.4: La desviación promedio de las predicciones 
#con respecto a los valores reales es de 1.4 unidades.
#R-cuadrado de 0.755: El modelo explica aproximadamente el 75.5% de la variabilidad en las ventas,
#lo que indica una buena capacidad predictiva.
#Ambos resultados sugieren que el modelo está funcionando bastante bien,
#captura una buena parte de la variabilidad en las ventas
#y con errores de predicción relativamente bajos
#pero el juste anterior es mejor.

# Crear la gráfica
ggplot(modelo_ajustado, aes(x = C, y = sigma, fill = RMSE)) +
  geom_tile() +
  scale_fill_viridis_c(option = "plasma") +
  labs(title = "RMSE en función de Sigma y C",
       x = "C",
       y = "Sigma",
       fill = "RMSE") +
  theme_minimal()

# Graficar las predicciones del modelo ajustado vs valores reales
library(ggplot2)

ggplot() +
  geom_point(aes(x = test_data_scaled$sales, y = predicciones_ajustadas), color = "blue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Predicciones del Modelo Ajustado vs Valores Reales",
       x = "Valores Reales",
       y = "Predicciones del Modelo Ajustado") +
  theme_minimal()

#El modelo SVM ha demostrado un buen rendimiento en la predicción de ventas,
#con un RMSE de 1.09, indicando errores de predicción relativamente bajos,
#y un R-cuadrado de 0.852, que muestra que el modelo explica el 85.2% de la variabilidad en las ventas.
#Estos resultados sugieren que el modelo es preciso y eficaz. 
#Los hiperparámetros óptimos seleccionados fueron sigma = 0.0625 y C = 4,
# los cuales optimizaron el rendimiento del modelo.