#!/usr/bin/env python
# algoritmo de selección clonal (sistema inmune artificial)
from random import random
from matplotlib import pyplot as plt
import numpy as np

def optimization(x) -> float:
    """Función benchmark (absolute)"""
    return abs(x)

def generate_pop(min: float, max: float, n: int):
    """Genera la el vector de anticuerpos (soluciones)"""
    # fila 1: anticuerpos (su valor en x)
    # fila 2: su afinidad
    x = np.zeros(shape=(n, 2))
    offset = abs(min) + abs(max)
    for i in range(n):
        x[i, 0] = ((random() * 100) % offset) + min
    return x

def mutate(x, mut, min, max):
    if mut >= random():
        offset = abs(min) + abs(max)
        x = ((random() * 100) % offset) + min
    return x

def clone(x, best, f):
    """Clona los mejores anticuerpos para generar a la nueva población"""
    tmp_arr = np.array([])
    for i in range(best):
        tmp = np.full((1, f), x[i])
        tmp_arr = np.append(tmp_arr, tmp)
    return tmp_arr

# Parámetros del CLONALG (clonal selection algorithm)
min = -10
max = 10
pop_size = 100
gen = 30
# Población de anticuerpos (soluciones) y sus afinidades
ab = generate_pop(min, max, pop_size)
# Mejores anticuerpos de la generación
best = 50
best_ab = np.zeros(shape=(best, 1))
# Probabilidad de mutación
mut = 0.02
# Factor de multiplicación
f = int(pop_size / best)
# Array con la media de los datos para realizar las gráficas
gen_mean = []
gen_best = abs(max)
it_best = 0
# Impresión de datos iniciales
print(f"Datos iniciales:")
print(f"Campo de búsqueda (Dominio de la función): [{min}, {max}]")
print(f"Generaciones: {gen}")
print(f"Cantidad de anticuerpos (tamaño de la población): {pop_size}")
print(f"Probabilidad de mutación: {mut}")
print(f"Cantidad de mejores anticuerpos por generación: {best}")

# Algoritmo
for g in range(gen):
    for p in range(pop_size):
        # Se calcula la afinidad de cada posible solución (su resultado en la
        # función de optimización) y se almacena el valor en tmp_ab
        ab[p, 1] = optimization(np.transpose(ab[p, 0]))
    # Se ordena el vector de afinidades de menor a mayor
    ab = ab[ab[:, 1].argsort()]
    # Se clonan los mejores anticuerpos de la solución para generar la siguiente gen
    ab[:, 0] = clone(np.transpose(ab[:, 0]), best, f)
    # Se realiza la mutación
    for p in range(pop_size):
        ab[p, 0] = mutate(ab[p, 0], mut, min, max)
    mean = np.mean(ab[:, 0])
    gen_mean.append(mean)
    if abs(mean) < abs(gen_best):
        gen_best = mean
        it_best = g

# Impresión de los resultados en consola
print(f"\nMejor solución obtenida (promedio): {gen_best}")
print(f"Obtenido en la iteración: {it_best}")

# Graficación
fig = plt.figure()
ax = fig.add_subplot()
fig.suptitle("Optimización por Sistemas Inmunes Artificiales (Selección clonal)")
ax.set_xlabel("Generaciones")
ax.set_ylabel("Costo")
plt.plot(gen_mean, 'b-', label='Promedio de los anticuerpos')
plt.plot(gen - 1, np.mean(ab[:, 0]), 'r+', label='Últimos anticuerpos encontrados (promedio)')
plt.plot(it_best, gen_best, 'r*', label='Mejores anticuerpos encontrados (promedio)')
plt.plot([ 0 for _ in range(gen + 1) ], 'g--', label='Solución óptima (mínimo global)')
ax.legend()
plt.show()
