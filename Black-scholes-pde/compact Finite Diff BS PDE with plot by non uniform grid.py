import numpy as np
import sympy as smp
from sympy import symbols, sin, exp, diff, lambdify
from scipy.sparse import diags
import scipy
import matplotlib.pyplot as plt

# Parameters
m = 3
n = 3
x0 = -2
xm = 2
t0 = 0
tn = 1
T = 1
c = 0.1
h = (xm - x0) / m
print(h)


    
dt = T / n

# Define symbolic variables
t1 = symbols('t1')
x1 = symbols('x1')

# Define arrays x and t
x = np.zeros(m+1)
x[0] = -2
x[m] = 2
for i in range(0,m+1):
    x[i] = x0+i*h
# print('x ',x)

# ____________________________________________________________
#non Uniform grid 

mu = np.zeros(n+1)
for i in range(0,n+1):
    mu[i] = ((1+c)**(x[i])-1)/(c)
    # mu[i] = x[i]
    
h = np.zeros(n+1)
# for i in range(1,n+1):
#     hb[i] = mu[i-1]-mu[i]
for i in range(0,n):
    h[i+1] = mu[i+1]-mu[i]
h[0]=h[1]
# ____________________________________________________________


t = np.zeros(n+1)
for i in range(0,n+1):
    t[i] = i*dt
# print('t ',t)
th = np.zeros(n)
for i in range(0,n):
    th[i] = t[i]+dt/2

# print('t(1/2)',th)

# Define the functions symbolically
def a(x1, t1):
    return 0.08 * (2 + (1 - t1) * sin(exp(x1)))**2

def b(x1, t1):
    return 0.06 * (1 + t1 * exp(-exp(x1))) - 0.02 * exp(-t1 - exp(x1)) - 0.08 * (2 + (1 - t1) * sin(exp(x1)))**2

def r(x1, t1):
    return -0.06 * (1 + t1 * exp(-exp(x1)))

def f(x1, t1):
    return 0.02 * exp(x1 - exp(x1) - 2 * t1) - exp(x1 - t1)
# Initialize solution matrix u
u = np.zeros([m+1,n+1])

for i in range(0,m+1):
    u[i,0] =np.e**(x[i])
for j in range(0,n+1):        
    u[0,j] = np.e**(-2-t[j])
    u[m,j] = np.e**(2-t[j])
# print("u\n",u)

# Compute symbolic derivatives
a1_sym = diff(a(x1, t1), x1)
a2_sym = diff(a1_sym, x1)
b1_sym = diff(b(x1, t1), x1)
b2_sym = diff(b1_sym, x1)
r1_sym = diff(r(x1, t1), x1)
r2_sym = diff(r1_sym, x1)

# Convert symbolic derivatives to numerical functions
a1_func = lambdify((x1, t1), a1_sym, modules="numpy")
a2_func = lambdify((x1, t1), a2_sym, modules="numpy")
b1_func = lambdify((x1, t1), b1_sym, modules="numpy")
b2_func = lambdify((x1, t1), b2_sym, modules="numpy")
r1_func = lambdify((x1, t1), r1_sym, modules="numpy")
r2_func = lambdify((x1, t1), r2_sym, modules="numpy")

# Initialize arrays for numerical computations
fj = np.zeros([m+1,n])
aj = np.zeros([m+1,n])
bj = np.zeros([m+1,n])
rj = np.zeros([m+1,n])

# Compute values using symbolic functions
for i in range(0,m+1):
    for j in range(0,n):
        fj[i,j] = f(x[i],th[j])
        aj[i,j] = (a(x[i], th[j]))*(-1/2)
        bj[i,j] = (b(x[i], th[j]))*(-1/2)
        rj[i,j] = r(x[i], th[j])

gx = np.zeros([m+1,n])
gx[0,0] = fj[0,0] - aj[0,0] * (exp(-2)) - bj[0,0] * (exp(-2)) + (1 / dt + rj[0,0]/2) * u[0,0]
gx[m,0] = fj[m,0] - aj[m,0] * (exp(2)) - bj[m,0] * (exp(2)) + (1 / dt + rj[m,0]/2) * u[m,0]

for i in range(1,m):
    gx[i,0] = fj[i,0] - aj[i,0] * ((u[i - 1,0] - 2 * u[i,0] + u[i + 1,0]) / h[i]**2) - bj[i,0] * ((u[i + 1,0] - u[i-1,0]) / (2 * h[i])) + (1 / dt +rj[i,0]/2) * u[i,0]

ai = np.zeros([m-1,n])
bi = np.zeros([m-1,n])
ci = np.zeros([m-1,n])
pi = np.zeros([m-1,n])
qi = np.zeros([m-1,n])
ri = np.zeros([m-1,n])
a1i = np.zeros([m-1,n])
a2i = np.zeros([m-1,n])
b1i = np.zeros([m-1,n])
b2i = np.zeros([m-1,n])
c1i = np.zeros([m-1,n])
c2i = np.zeros([m-1,n])
E = np.zeros([m-1,n])
F = np.zeros([m-1,n])
G = np.zeros([m-1,n])
H = np.zeros([m-1,n])

# # Evaluate symbolic derivatives at specific points
for j in range(0,n):
    for i in range(0,m-1):
        ai[i,j] =aj[i+1,j]
        bi[i,j] = bj[i+1,j]
        ci[i,j] = rj[i+1,j]
        a1i[i,j] = (a1_func(x[i+1], th[j]))*(-1/2)
        a2i[i,j] = (a2_func(x[i+1], th[j]))*(-1/2)
        b1i[i,j] = (b1_func(x[i+1], th[j]))*(-1/2)
        b2i[i,j] = (b2_func(x[i+1], th[j]))*(-1/2)
        c1i[i,j] = (r1_func(x[i+1], th[j]))*(-1/2)
        c2i[i,j] = (r2_func(x[i+1], th[j]))*(-1/2)

        pi[i,j] = (bi[i,j]-2*a1i[i,j])*(a1i[i,j]+bi[i,j])+ai[i,j]*(a2i[i,j]+2*b1i[i,j]+1/dt-ci[i,j]/2)
        qi[i,j] = (bi[i,j]-2*a1i[i,j])*(b1i[i,j]+1/dt-ci[i,j])+ai[i,j]*(b2i[i,j]+2*c1i[i,j])
        ri[i,j] = (bi[i,j]-2*a1i[i,j])*c1i[i,j]+ai[i,j]*c2i[i,j]

        E[i,j] = -24*ai[i,j]**2-2*h[i]**2*pi[i,j]+12*ai[i,j]*bi[i,j]*h[i]+h[i]**3*qi[i,j]
        F[i,j] = 48*ai[i,j]**2+4*h[i]**2*pi[i,j]-24*ai[i,j]*(1/dt-ci[i,j]/2)*h[i]**2-2*h[i]**4*ri[i,j]
        G[i,j] = -24*ai[i,j]**2-2*h[i]**2*pi[i,j]-12*ai[i,j]*bi[i,j]*h[i]-h[i]**3*qi[i,j]

for i in range(0,m-1):
    # H[i-1] = h**3*(2*a1i[i-1]-bi[i-1])*(gx[i+1]-gx[i-1])-2*h**2*ai[i-1]*(gx[i-1]-2*gx[i]+gx[i+1])-24*ai[i-1]*h**2*gx[i]
    H[i,0] = h[i]**3*(2*a1i[i,0]-bi[i,0])*(gx[i+2,0]-gx[i,0])-2*h[i]**2*ai[i,0]*(gx[i,0]-2*gx[i+1,0]+gx[i+2,0])-24*ai[i,0]*h[i]**2*gx[i+1,0]
    
for j in range(0,n-1):
       
    P = np.zeros([m-1,m-1])
    for i in range(1,m-2):
        P[i,i]=F[i,j]
        P[i,i-1]=E[i,j]
        P[i,i+1]=G[i,j]
    P[0,0]=F[0,j]
    P[0,1]=G[0,j]
    P[m-2,m-2]=F[m-2,j]
    P[m-2,m-3]=E[m-2,j]
        
    R = np.zeros(m-1)
    for i in range(1,m-2):
        R[i] = H[i,j]
    R[0] = H[0,j]-E[0,j]*(u[0,j+1])
    R[m-2] = H[m-2,j]-G[m-2,j]*(u[m,j+1])
        
    uj = np.linalg.solve(P,R)
        
    for i in range(1,m):
        u[i,j+1]=np.take(uj,i-1)
    
    gx[0,j+1] = fj[0,j+1] - aj[0,j+1] * (exp(-2-t[j+1])) - bj[0,j+1] * (exp(-2-t[j+1])) + (1 / dt + rj[0,j+1]/2) * u[0,j+1]
    gx[m,j+1] = fj[m,j+1] - aj[m,j+1] * (exp(2-t[j+1])) - bj[m,j+1] * (exp(2-t[j+1])) + (1 / dt + rj[m,j+1]/2) * u[m,j+1]
   
    for i in range(1,m):
        gx[i,j+1] = fj[i,j+1] - aj[i,j+1] * ((u[i - 1,j+1] - 2 * u[i,j+1] + u[i + 1,j+1]) / h[i]**2) - bj[i,j+1] * ((u[i + 1,j+1] - u[i-1,j+1]) / (2 * h[i])) + (1 / dt + rj[i,j+1]/2) * u[i,j+1]
    for i in range(0,m-1):  
        H[i,j+1] = h[i]**3*(2*a1i[i,j+1]-bi[i,j+1])*(gx[i+2,j+1]-gx[i,j+1])-2*h[i]**2*ai[i,j+1]*(gx[i,j+1]-2*gx[i+1,j+1]+gx[i+2,j+1])-24*ai[i,j+1]*h[i]**2*gx[i+1,j+1]

P = np.zeros([m-1,m-1])
for i in range(1,m-2):
    P[i,i]=F[i,n-1]
    P[i,i-1]=E[i,n-1]
    P[i,i+1]=G[i,n-1]
P[0,0]=F[0,n-1]
P[0,1]=G[0,n-1]
P[m-2,m-2]=F[m-2,n-1]
P[m-2,m-3]=E[m-2,n-1]    
R = np.zeros(m-1)
for i in range(1,m-2):
    R[i] = H[i,n-1]
R[0] = H[0,n-1]-E[0,n-1]*(u[0,n])
R[m-2] = H[m-2,n-1]-G[m-2,n-1]*(u[m,n])
        
uj = np.linalg.solve(P,R)
        
for i in range(1,m):
    u[i,n]=np.take(uj,i-1)
    

v = np.zeros([m+1,n+1])
err = np.zeros([m+1,n+1])
for i in range(m+1):
    for j in range(n+1):
        v[i,j] = np.e**(x[i]-t[j])
        err[i,j]=np.abs(u[i,j]-v[i,j])
        
error = err.max()
print(error)





# from mpl_toolkits.mplot3d import Axes3D

# Create meshgrid for plotting
X, T = np.meshgrid(x, t)

# Transpose the matrix for correct orientation
U = u.T
V = v.T

# Plot the exact solution
fig = plt.figure(figsize=(14, 7))

# Exact solution
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, T, V, cmap='viridis')
ax1.set_title('Exact Solution')
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_zlabel('v(x,t)')
plt.savefig('1.png',dpi=500)

# Approximate solution
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, T, U, cmap='inferno')
ax2.set_title('Approximate Solution')
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_zlabel('u(x,t)')
plt.savefig('2.png',dpi=500)
plt.show()


# Plot the graph of Exact solution vs Approximate solution
plt.figure(figsize=(10, 6))

# Exact solution (v)
plt.subplot(1, 2, 1)
plt.title("Exact Solution")
plt.contourf(t, x, v, 20, cmap='viridis')
plt.colorbar()
plt.xlabel('Time (t)')
plt.ylabel('Space (x)')
plt.grid()

# Approximate solution (u)
plt.subplot(1, 2, 2)
plt.title("Approximate Solution")
plt.contourf(t, x, u, 20, cmap='viridis')
plt.colorbar()
plt.xlabel('Time (t)')
plt.ylabel('Space (x)')
plt.grid()

plt.tight_layout()
plt.show()
