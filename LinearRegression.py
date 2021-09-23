#importando datos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importando datos

dataset = pd.read_csv('C:/Users/Dell/Documents/DataScience/ProcesamientoDatos/salarios.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# Separando datos en sets de entrenamiento y prueba

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x,y, test_size=1/3, random_state=0)

# Para hacer la regresión entre los datos usamos

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Prediciendo resultados del set de prueba

y_pred = regressor.predict(x_test)

# visualizando los resultados del set de entrenamiento  
plt.scatter(x_train, y_train, color='green')
plt.plot(x_train, regressor.predict(x_train), color="yellow")
plt.title('Salario contra años de experienciencia(Set de Entrenamiento)')
plt.xlabel('Años de Experiencia')
plt.ylabel('Salario')
plt.show()

# visualizando los resultados del set de prueba 
plt.scatter(x_test, y_test, color='green')
plt.plot(x_train, regressor.predict(x_train), color="yellow")
plt.title('Salario contra años de experienciencia(Set de Entrenamiento)')
plt.xlabel('Años de Experiencia')
plt.ylabel('Salario')
plt.show()
