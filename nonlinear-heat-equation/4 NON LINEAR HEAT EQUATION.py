import numpy as np
import matplotlib.pyplot as plt

# Constants
M = 10
N = 200
TN = 15
k = TN / (M-1)
a = 1
b = 3
h = (b-a) / (N-1)
alpha = 1
beta = 1

# Spatial and initial condition setup
x = np.linspace(a, b, N)
u0 = np.zeros(N)
for i in range(N):
    u0[i] = 2 - (x[i]-1)/2 + (x[i]-1)*(x[i]-3)

# Time array setup
t = np.linspace(0, TN, M)

# Initialize arrays for the solution
U = np.zeros((N, M))
U[:, 0] = u0
print(U)
# Set up L matrix
L = np.zeros((N, N))
L[0, 0] = 1
L[N-1, N-1] = 1

# Time stepping
for j in range(1, M):
    u = U[:, j-1]
    G = np.zeros(N)
    print(u)
    G[0] = u[0] - alpha
    G[N-1] = u[N-1] - beta
    
    for i in range(1, N-1):
        G[i] = u[i+1] - 2*u[i] + u[i-1] - h*h
        L[i, i-1] = 1 + 0.5 * h
        L[i, i] = -2 - h*h
        L[i, i+1] = 1 - 0.5 * h
    
#     # Solve the system of equations
    u_next = np.linalg.solve(L, G)
    
#     # Update the solution matrix
    U[:, j] = u_next
    print(U)
# Plotting
X, T = np.meshgrid(x, t)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, U.T, cmap='viridis')

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x, t)')
plt.show()
