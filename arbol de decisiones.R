#Decision Tree - Regresión

library(lattice)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(pROC)

datos<-read.csv("D:/Descargas/Child Car Safety Seats.csv")

# columnas:
#sales: Ventas (posiblemente en miles de unidades).
#comp_price: Precio de la competencia.
#income: Ingreso promedio (posiblemente en miles de dólares).
#advertising: Gasto en publicidad (posiblemente en miles de dólares).
#population: Población del área (en miles).
#price: Precio del asiento de seguridad.
#shelveLoc: Ubicación en la estantería (puede ser "Bad", "Medium" o "Good").
#age: Edad promedio de la población.
#education: Nivel de educación promedio (en años).
#urban: Si el área es urbana (Sí o No).
#us: Si el área está en Estados Unidos (Sí o No).

#Variable dependiente: sales

# Se dividen los datos en datos de entrenamiento y datos de prueba
# entrenaremos el modelo sólo con el 80% de los datos
set.seed(123)
particiones = createDataPartition(datos$sales, p = 0.8, list = FALSE)
datos_entrenamiento = datos[particiones, ] # Datos para entrenar el modelo
datos_prueba = datos[-particiones, ] # Datos que luego usaremos para probar el modelo


# Generaremos un modelo utilizando los datos de entrenamiento

arbol = rpart(sales ~., 
              data = datos_entrenamiento, 
              method = "anova", # ANOfVArience.
              cp=0.001) # cp especifica el valor del costo de complejidad mínimo.

# se dibuja el arbol
rpart.plot(arbol,fallen.leaves = FALSE)

# Poda del árbol:
# Nos ayudara a encontrar el modelo optimo

mejor_cp = arbol$cptable[which.min(arbol$cptable[,"xerror"]),"CP"]

# Podamos el árbol
arbol_podado = prune(arbol, cp = mejor_cp)

# dibujamos el arbol podado
rpart.plot(arbol_podado, fallen.leaves = FALSE)

#ahora se validaran los modelos y se compararán
#Se harán predicciones y se utilizará el coeficiente de determinación (R^2)
val_pred1 = predict(arbol, datos_prueba) 
val_pred2 = predict(arbol_podado, datos_prueba) 

# Cálculo de R^2 arbol no podado
# Calcular la suma total de cuadrados (TSS)
tss = sum((datos_prueba$sales - mean(datos_prueba$sales))^2)
# Calcular la suma de los cuadrados de los residuos (RSS)
rss = sum((datos_prueba$sales - val_pred1)^2)
# Calcular el coeficiente de determinación (R^2)
r_cuadrado1 = 1 - (rss / tss)

# Cálculo de MSE para el árbol sin podar
MSE1 <- mean((datos_prueba$sales - val_pred1)^2)

# Cálculo de R^2 arbol podado
# Calcular la suma total de cuadrados (TSS)
tss = sum((datos_prueba$sales - mean(datos_prueba$sales))^2)
# Calcular la suma de los cuadrados de los residuos (RSS)
rss = sum((datos_prueba$sales - val_pred2)^2)
# Calcular el coeficiente de determinación (R^2)
r_cuadrado2 = 1 - (rss / tss)

# Cálculo de MSE para el árbol podado
MSE2 <- mean((datos_prueba$sales - val_pred2)^2)

if (r_cuadrado1 > r_cuadrado2){
  print("Módelo Árbol sin poda es mejor")
}else{
  print("Módelo Árbol con poda es mejor")
}

#R^2 del arbol sin podar es de 0.5562676 
#mientras que el valor de R^2 del arbol podado es de 0.5358276
# por lo tanto el Módelo Árbol sin poda es mejor, ya que es mas cercano a 1.