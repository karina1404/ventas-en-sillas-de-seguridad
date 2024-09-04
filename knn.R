install.packages("kknn")
library(kknn)

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

# Revisar datos faltantes
any(is.na(data))

# Convertir variables categóricas a factores
data$shelveLoc <- as.factor(data$shelveLoc)
data$urban <- as.factor(data$urban)
data$us <- as.factor(data$us)


# Dividir los datos en conjuntos de entrenamiento y prueba
set.seed(123)
particiones <- sample(1:nrow(data), size = 0.8 * nrow(data))
datos_entrenamiento <- data[particiones, ]
datos_prueba <- data[-particiones, ]

# Definir los valores de k y tipos de distancia a experimentar
valores_k <- c(1,2,3,4, 5,6, 7,8, 9,10,11,12,13,15)
distancias <- c(1, 2)  # 1: Manhattan, 2: Euclidiana

# Crear un dataframe para almacenar los resultados
resultados <- data.frame(k = integer(), distancia = character(), MSE = numeric(), R2 = numeric())

# Iterar sobre los valores de k y distancias
for (k in valores_k) {
  for (dist in distancias) {
    
    # Aplicar el modelo KNN
    modelo_knn <- kknn(sales ~ ., 
                       datos_entrenamiento, 
                       datos_prueba[,-1],  # Excluyendo la columna de ventas
                       k = k, 
                       distance = dist)  
    
    # Obtener las predicciones
    predicciones_knn <- fitted(modelo_knn)
    
    # Calcular el MSE
    mse_knn <- mean((datos_prueba$sales - predicciones_knn)^2)
    
    # Calcular el coeficiente de determinación (R^2)
    sum_total_cuadrado_knn <- sum((datos_prueba$sales - mean(datos_prueba$sales))^2)
    rss_knn <- sum((datos_prueba$sales - predicciones_knn)^2)
    r2_knn <- 1 - (rss_knn / sum_total_cuadrado_knn)
    
    # Almacenar los resultados
    dist_name <- ifelse(dist == 1, "Manhattan", "Euclidiana")
    resultados <- rbind(resultados, data.frame(k = k, distancia = dist_name, MSE = mse_knn, R2 = r2_knn))
  }
}

# Mostrar los resultados
print(resultados)

# Identificar la configuración con el menor MSE
mejor_mse <- resultados[which.min(resultados$MSE), ]
print(paste("Mejor configuración basada en MSE:"))
print(mejor_mse)

# Identificar la configuración con el mayor R^2
mejor_r2 <- resultados[which.max(resultados$R2), ]
print(paste("Mejor configuración basada en R^2:"))
print(mejor_r2)

#Aunque el modelo con 𝑘=8 y la distancia Euclidiana ofrece el mejor resultado
#en comparación con otras combinaciones, el valor de 𝑅^2= 0.49 
#sugiere que el modelo no es particularmente robusto
#en su capacidad para predecir las ventas. 
#Esto indica que podría ser necesario mejorar el modelo,
#ya sea ajustando más parámetros, explorando diferentes métodos de modelado,
#o revisando las características del conjunto de datos.
