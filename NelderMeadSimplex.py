import numpy as np

def nelder_mead(f, x_start, gamma=2.0, beta=0.5, tol=1e-5, max_iter=1000):

    N = len(x_start)
    
    def create_initial_simplex(x0, alpha=1.0):
        delta1 = (np.sqrt(N + 1) + N - 1) / (N * np.sqrt(2)) * alpha
        delta2 = (np.sqrt(N + 1) - 1) / (N * np.sqrt(2)) * alpha
        simplex = np.zeros((N + 1, N))
        simplex[0] = x0
        for i in range(1, N + 1):
            point = x0.copy()
            point[i - 1] += delta1
            for j in range(N):
                if j != i - 1:
                    point[j] += delta2
            simplex[i] = point
        return simplex

    simplex = create_initial_simplex(x_start)
    
    for iteration in range(max_iter):
        simplex = sorted(simplex, key=lambda x: f(x))
        xc = np.mean(simplex[:-1], axis=0) 
        xr = 2 * xc - simplex[-1]
        
        if f(xr) < f(simplex[0]):
            # Expansión
            xe = xc + gamma * (xc - simplex[-1])
            x_new = xe if f(xe) < f(xr) else xr
        elif f(xr) >= f(simplex[-2]):
            if f(xr) < f(simplex[-1]):
                # Contracción fuera
                x_new = xc + beta * (xr - xc)
            else:
                # Contracción dentro
                x_new = xc - beta * (xc - simplex[-1])
        else:
            x_new = xr
        
        simplex[-1] = x_new  
        
        
        if np.sqrt(np.mean([(f(x) - f(xc))**2 for x in simplex])) <= tol:
            break


    simplex = sorted(simplex, key=lambda x: f(x))
    return simplex[0], f(simplex[0]), iteration


def objective_function(x):
    # return x[0]**2 + x[1]**2 
    # return np.sum(np.square(x))

    # Funcion Himmelblau
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    # Función Rastrigin
    # n = len(x)
    # return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))  

    # Funcion Rosenbrock
    # return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))





# x_start = np.array([-1.0, 1.5])

# Funcion Himmelblau
x_start = np.array([3.0, 2.0])

# Función Rastrigin
# x_start = np.array([-2,-2,-2])

# Funcion Rosenbrock
# x_start = np.array([2,1.5,3,-1.5,-2])

x_opt, f_opt, n_iter = nelder_mead(objective_function, x_start)

print(f"Punto óptimo: {x_opt}")
print(f"Valor óptimo de la función: {f_opt}")
print(f"Número de iteraciones: {n_iter}")
