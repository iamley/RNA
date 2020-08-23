# -*- coding: utf-8 -*-
"""
Created on Tue JulLst 25 00:37:45 2020

@author: Leidy Pulido
"""
import numpy as np
 
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
 
def sigmoid_derivada(x):
    return sigmoid(x)*(1.0-sigmoid(x))
 
def tanh(x):
    return np.tanh(x)
 
def tanh_derivada(x):
    return 1.0 - x**2
 
 
class NNetwork:
 
    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_derivada
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_derivada
 
        # Inicializamos los pesos #
        self.weights = []
        self.deltas = []
        # Capas -> [1,3,2] #
        # Random de pesos varia entre (-1,1)
        # Se asigna valores aleatorios a capa de entrada y capa oculta
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        # Se asigna valores aleatorios a la capa de salida
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)
 
    def fit(self, X, y, learning_rate=0.2, epochs=100000):
        # Agrego columna de unos a las entradas X
        # Con esto agregamos la unidad de Bias a la capa de entrada
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]
 
            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    # Proceso de activación #
                    activation = self.activation(dot_value)
                    a.append(activation)
            # Calculo la diferencia en la capa de salida y el valor obtenido
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]
            
            # Empezamos en el segundo layer hasta el ultimo #
            # (Una capa anterior a la de salida)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))
            self.deltas.append(deltas)
 
            # Proceso de invertir #
            deltas.reverse()
 
            # Backpropagation #
            # 1. Multiplico los delta de salida con las activaciones de entrada para obtener el gradiente del peso #
            # 2. Actualizo el peso restandole un porcentaje del gradiente #
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
 
            if k % 10000 == 0: print('# Iteraciones:', k)
 
    def predict(self, x): 
        # Aqui genero las predicciones y activación de las redes #
        ones = np.atleast_2d(np.ones(x.shape[0]))
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a
 
    def print_weights(self):
        print("Listado de pesos de conexiones:")
        for i in range(len(self.weights)):
            print(self.weights[i])
 
    def get_deltas(self):
        return self.deltas
    
    
    nn = NNetwork([1,3,2],activation ='tanh')
    X = np.array([[6],   # frecuencia baja
                  [9],   # frecuencia baja
                  [10],  # frecuencia baja
                  [18],  # frecuencia estable
                  [20],  # frecuencia alta
                  [12]]) # frecuencia estable
     
    y = np.array([[1,0],   # incremento
                  [1,0],   # incremento
                  [1,0],   # incremento
                  [0,0],   # estable
                  [0,1],   # decremento
                  [0,0]])  # estable
    
    nn.fit(X, y, learning_rate=0.03,epochs=15001)
 
    index=0
    for e in X:
        print("X:",e,"y:",y[index],"Red:",nn.predict(e))
        index=index+1
    
    