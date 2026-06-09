import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

M = 20
N = 20
tM = 15
dt = tM / (M-1)
a = 1
b = 3 
h = (b-a) / (N-1)
alpha = 2
beta = 1
u0 = np.zeros(N)
x = np.linspace(a, b, N)
for i in range(0, N):
    u0[i] = 2 - (x[i] - 1) / 2 + (x[i] - 1) * (x[i] - 3)

u_l = np.zeros(N)
u = np.zeros(N)
uNext = np.zeros(N)
U = np.zeros([N, M])
U[:, 0] = u0

for n in range(1, M):
    u = U[:, n-1]
    u_l = U[:, n-1]
        
    J = np.zeros([N, N])
    J[0, 0] = 1
    J[N-1, N-1] = 1
    F = np.zeros(N)
    F[0] = u[0] - alpha
    F[N-1] = u[N-1] - beta 
    for i in range(1, N-1):
        v = (u[i+1] - u[i-1]) / (2*h)
        phi = ((u[i] - u[i-1]) / dt) - np.e**(u[i]) * v**2
        f = phi / np.e**(u[i])
        q = -f + (1 / (dt * np.e**(u[i]))) - v**2
        p = -2*v
        F[i] = u[i+1] - 2*u[i] + u[i-1] - h**2*f
        J[i, i] = -2 - (h**2) * q
        J[i, i-1] = 1 + 0.5*h*p
        J[i, i+1] = 1 - 0.5*h*p
        J[0, 1] =  1 - 0.5*h*p
        J[N-1, N-2] = 1 + 0.5*h*p

    Jinv = np.linalg.inv(J)
    uNext = u - Jinv@F
    
    U[:,n] = uNext

# 3D Plot
X, T = np.meshgrid(x, np.linspace(0, tM, M))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, U.T, cmap='viridis')

ax.set_xlabel('x')
ax.set_ylabel('time')
ax.set_zlabel('u')
ax.set_title('3D Surface Plot of Solution')

plt.show()

# Plotting the results
for n in range(M):
    plt.plot(x, U[:,n],)

plt.xlabel('x')
plt.ylabel('u')
plt.title('Solution of the PDE over time')
plt.legend()
plt.show()
