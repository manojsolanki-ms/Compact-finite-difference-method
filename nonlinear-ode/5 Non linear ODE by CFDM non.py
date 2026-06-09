import numpy as np
import sympy as smp
from sympy import symbols, sin, exp, diff, lambdify
from scipy.sparse import diags
import scipy
import matplotlib.pyplot as plt
import sympy as smp
from sympy import symbols
import math
    
y = symbols("y")
x = symbols("x")
alpha = 0.1
a = 0
b = 1
yL = 1
yR =1 
n = 4
h=(b-a)/n-1
Ht2 = 0.5
# print(h)
c = 0.1
X=np.linspace(a,b,n+1)
# print("x",X)

f = -Ht2*(1-(y/(1-alpha*y)))
# print(f)
f_fun = smp.lambdify(y,f)
u0y = f/2
uxy = f



# intial guess for Quassilinirization 

# y0x = 
Y = np.zeros(n+1)
Yi = np.zeros(n+1)
mu = np.zeros(n+1)
for i in range(0,n+1):
    mu[i] = ((1+c)**(X[i])-1)/(c)
    # w = 1
    # mu[i] = math.asin((w * math.cos(math.pi * i / n)) / math.asin(w))

# print("mu",mu)
# hb = np.zeros(n+1)
hf = np.zeros(n+1)
# for i in range(1,n+1):
#     hb[i] = mu[i-1]-mu[i]
for i in range(0,n):
    hf[i+1] = mu[i+1]-mu[i]
hf[0]=hf[1]
    # hf[n] = 0.17A
# hf = np.zeros(n+1)
# for i in range(n+1):
#     hf[i] = h
hb = hf
# print("hb",hb)
# print("hf",hf)

M1 = 1/x
#numerical values of M 
M_fun = smp.lambdify(x,M1)
M = np.zeros(n+1)

for i in range(n+1):
    if X[i]!=0:
        M[i] = M_fun(X[i])

        # M[i] = 1/X[i]
    else:
        M[i]=0
print("M",M)    
  
# Frist Derivative of M = Md1
Md1 = smp.diff(M1,x)
Md1_fun =smp.lambdify(x,Md1)
# print(Md1)
Md2 = smp.diff(Md1,x)
Md2_fun =smp.lambdify(x,Md2)

Md1 = np.zeros(n+1)
Md2 = np.zeros(n+1)
# print(Md2)
for i in range(n+1):
    if X[i]!=0:
            Md1[i] = Md1_fun(X[i])
            Md2[i] = Md2_fun(X[i])
            # M[i] = 1/X[i]
    else:
        Md1[i]=0
        Md2[i]=0

    

Y = np.zeros(n)
Y0 = np.zeros(n)
# print(Y)
Y1 = np.zeros(n+1)
Y2 = np.zeros(n+1)
uy1 = np.zeros(n+1)
uy2 = np.zeros(n+1)
u1 = np.zeros(n+1)
Vt1 = np.zeros(n+1)
Vt2 = np.zeros(n+1)
Ut1 = np.zeros(n+1)
Ut2 = np.zeros(n+1)

for i in range(n):
    Y0[i] = 1/(1+alpha)

for l in range(5):
    for i in range(n): 
        # Y0[i] = 1/(1+alpha)
        # Y0 = np.array([0.0090162 ,0.01068306 ,0.01205378 ,0.00906355])
        
        Y[i] = -Ht2*(1-(Y0[i]/1-alpha*Y0[i]))
    
        uy1[i] = Ht2/(1-alpha*Y[i]**2)
        uy2[i] = 2*alpha*((1-alpha*Y[i])*Y2[i]+3*alpha*Y1[i]**2)/(1-alpha*Y[i])**4
        u1[i] = Ht2*(Y1[i])/(1-alpha*Y[i])**2
        # 
    # Y = Yi
    # Y = np.array([0.14206078,0.13517144,0.39992203,0.90909091])
    # print("1",Y)
    Ut = np.zeros(n)
    Vt = np.zeros(n)
    for i in range(1,n):
        Ut[i] = Ht2/(1-alpha*Y[i])**2
        Ut[0] = Ht2/(1-alpha*Y[0])**2*(0.5)
        Ut1[i-1] = Ut[i]-Ut[i-1]
        Ut2[i-1] = Ut1[i]-Ut1[i-1]
        # Ut1 = np.array([0.2455157,0,0,0,0,0])
        # Ut2 = np.array([-0.2455157,0,0,0,0,0])
        
        Vt[i] = f_fun(Y[i]) -Y[i]*Ht2/(1-alpha*Y[i])**2
        Vt[0] = (f_fun(Y[0]))/2 -Y[0]*Ht2/(1-alpha*Y[0])**2
        Vt1[i-1] = Vt[i]-Vt[i-1]
        Vt2[i-1] =  Vt1[i]-Vt1[i-1]
        # Vt1 = np.array([-0.27252252,0,0,0,0,0])
        # Vt2 = np.array([0.27252252,0,0,0,0,0])
        
    # values of ai,bi,ci,di
    ai = np.zeros(n)
    bi= np.zeros(n)
    ci= np.zeros(n)
    di= np.zeros(n)
    for i in range (0,n):
       ai[i] = -1 + (((hb[i] - hf[i+1]) / 3) * M[i]) + (((hb[i])**2 + (hf[i+1])**2 - hb[i] * hf[i+1]) / 12) * ((M[i])**2 +Ut[i]- Md1[i]) - ((hb[i] * hf[i+1]) / 6) * (M[i])**2 + ((hb[i] * (hf[i+1])**2 - (hb[i])**2 * hf[i+1]) / 24) * ((M[i])**3 + M[i]*Ut[i] - 2 * M[i] * Md1[i])
       bi[i] = ((hf[i+1] - hb[i]) / 3) * (Ut[i] - Md1[i])+ ((hb[i]**2 + hf[i+1]**2 - hb[i] * hf[i+1]) / 12) * (M[i] * Md1[i] - M[i] * Ut[i] + 2 * Ut1[i] - Md2[i]) - M[i]+ (hb[i] * hf[i+1] / 6) * (M[i] * Ut[i] - M[i] * Md1[i])+ ((hb[i] * hf[i+1]**2 - hf[i+1] * hb[i]**2) / 24) * (M[i]**2 * Md1[i] - M[i]**2 * Ut[i] + 2 * M[i] * Ut1[i] - M[i] * Md2[i])
       ci[i] = ((hf[i+1] - hb[i]) / 3) * (Ut1[i]) + ((hb[i]**2 + hf[i+1]**2 - hb[i] * hf[i+1]) / 12) * (Ut2[i] - M[i] * Ut1[i]) + (hb[i] * hf[i+1] / 6) * (M[i] * Ut1[i]) + ((hb[i] * hf[i+1]**2 - hf[i+1] * hb[i]**2) / 24) * (M[i] * Ut2[i] - M[i]**2 * Ut1[i]) + Ut[i]
       di[i] = ((hf[i+1] - hb[i]) / 3) * Vt1[i]+ ((hb[i]**2 + hf[i+1]**2 - hb[i] * hf[i+1]) / 12) * (Vt2[i] - M[i] * Vt1[i])+ (hb[i] * hf[i+1] / 6) * M[i] * Vt1[i]+ ((hb[i] * hf[i+1]**2 - hf[i+1] * hb[i]**2) / 24) * (M[i] * Vt2[i] - M[i]**2 * Vt1[i]) + Vt[i]
    
    # for tridiagonal system 
    p = np.zeros(n)
    q = np.zeros(n)
    r = np.zeros(n)
    s = np.zeros(n)
    rt = np.zeros(n)
    st = np.zeros(n)
    
    for i in range(0,n):
        p[i] = -2*ai[i]/(hb[i]*hf[i+1])
        q[i] = -bi[i]*((hb[i])**2 * (hf[i+1]**2))/(hb[i]*hf[i+1])
        
    # for i in range(1,n-1):
        r[i] = -2*ai[i]/(hb[i]*(hb[i]+hf[i+1]))
        rt[i] = 2*ai[i]/(hb[i]*(hb[i]+hf[i+1]))
        s[i] = bi[i]*hf[i+1]/(hb[i]*(hb[i]+hf[i+1]))
        st[i] = bi[i]*hb[i]/(hf[i+1]*(hb[i]+hf[i+1]))
    
    vi = np.zeros(n)
    wi = np.zeros(n)
    wti = np.zeros(n)
    for i in range(n):
        vi[i] = p[i]+q[i]+ci[i]
        wti[i] = rt[i]-st[i]
    for i in range(1,n):
        wi[i-1] = r[i]-s[i]
        
    
    # Sparse matrix A setup
    A = diags([wi, vi, wti], [-1,0,1], shape=(n, n)).toarray()
    A[0,1] = (2*ai[0]/hf[0]*(hf[0]+hf[1])-bi[0]*hf[1]/hf[0]*(hf[0]+hf[1]))+(2*ai[0]/hf[1]*(hf[0]+hf[1])-bi[0]*hf[0]/hf[1]*(hf[0]+hf[1]))
# print(A)

    R= np.zeros(n)
    for i in range(n):
        R[i] = -di[i]    
    S = np.linalg.solve(A,R)
    Y0 = S
    print("S",Y0)
# Adjust X to match the dimensions of S
plt.figure(figsize=(8, 6))
plt.plot(X[:n], S, marker='o', linestyle='-', color='b', label="Solution S")
plt.xlabel("x")
plt.ylabel("S (Solution)")
plt.title("Numerical Solution")
plt.legend()
plt.grid()
plt.show()

