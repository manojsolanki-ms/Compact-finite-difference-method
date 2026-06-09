import numpy as np
import sympy as smp
from sympy import symbols, sin, exp, diff, lambdify
from scipy.sparse import diags
import scipy
import matplotlib.pyplot as plt
import sympy as smp
from sympy import symbols
    
y = symbols("y")
x = symbols("x")
alpha = 0.1
a = 0
b = 1
yL = 1
yR =1 
n = 4              
h=(b-a)/n
Ht2 = 0.5
print(h)
c = 0.1
X=np.linspace(a,b,n+1)
print("x",X)

f = -Ht2*(1-y/(1-alpha*y))
print(f)
f_fun = smp.lambdify(y,f)
u0y = f/2
uxy = f



# intial guess for Quassilinirization 

# y0x = 
# Y = np.zeros(n+1)
# Yi = np.zeros(n+1)
mu = np.zeros(n+1)
for i in range(0,n+1):
    mu[i] = ((1+c)**(X[i])-1)/(c)
# print("mu",mu)
# hb = np.zeros(n+1)
hf = np.zeros(n+1)
# for i in range(1,n+1):
#     hb[i] = mu[i-1]-mu[i]
for i in range(0,n):
    hf[i] = mu[i+1]-mu[i]
hb = hf
# print("hb",hb)
# print("hf",hf)

M1 = 1/x
#numerical values of M 
M_fun =smp.lambdify(x,M1)
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
# print(Md2)


    

Y = np.zeros(n)
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
Md1 = np.zeros(n+1)
Md2 = np.zeros(n+1)

for l in range(5):
    for i in range(n): 
        Y[i] = 1/(1+alpha)
        # Y = np.array([0.16715123, 0.15978877, 0.48189228, 1.10202849])
        
        Y[i] = -Ht2*(1-(Y[i]/1-alpha*Y[i]))
        uy1[i] = Ht2/(1-alpha*Y[i]**2)
        uy2[i] = 2*alpha*((1-alpha*Y[i])*Y2[i]+3*alpha*Y1[i]**2)/(1-alpha*Y[i])**4
        u1[i] = Ht2*(Y1[i])/(1-alpha*Y[i])**2
        # 
    # Y = Yi
    # Y = np.array([0.14206078,0.13517144,0.39992203,0.90909091])
    # print("1",Y)
    Ut = np.zeros(n)
    Vt = np.zeros(n)
    for i in range(n):
        Ut[i] = Ht2*(1-alpha*Y[i])**2
        Ut[0] = Ht2*(1-alpha*Y[0])**2*(0.5)
        Ut1[i] = uy1[i]
        Ut2[i] = uy2[i]
        
        Vt[i] = f_fun(Y[i]) -Y[i]*Ht2*(1-alpha*Y[i])**2
        Vt[0] = (f_fun(Y[0]))/2 -Y[0]*Ht2*(1-alpha*Y[0])**2
        Vt1[i] = -Y[i]*uy1[i]
        Vt2[i] = -Y1[i]*uy1[i]-Y[i]*uy2[i]
        
        for i in range(n+1):
            if X[i]!=0:
                Md1[i] = Md1_fun(X[i])
                Md2[i] = Md2_fun(X[i])
                # M[i] = 1/X[i]
            else:
                Md1[i]=0
                Md2[i]=0
        

    # values of ai,bi,ci,di
    ai = np.zeros(n)
    bi= np.zeros(n)
    ci= np.zeros(n)
    di= np.zeros(n)
    for i in range (0,n):
       ai[i] = -1 + (((hb[i] - hf[i+1]) / 3) * M[i]) + (((hb[i])**2 + (hf[i+1])**2 - hb[i] * hf[i+1]) / 12) * ((M[i])**2 - Md1[i]) - ((hb[i] * hf[i+1]) / 6) * (M[i])**2 + ((hb[i] * (hf[i+1])**2 - (hb[i])**2 * hf[i+1]) / 24) * ((M[i])**3 + M[i] - 2 * M[i] * Md1[i])
       bi[i] = ((hf[i+1] - hb[i]) / 3) * (Ut[i] - Md1[i])+ ((hb[i]**2 + hf[i+1]**2 - hb[i] * hf[i+1]) / 12) * (M[i] * Md1[i] - M[i] * Ut1[i] + 2 * Ut[i] - Md2[i] - M[i])+ (hb[i] * hf[i+1] / 6) * (M[i] * Ut[i] - M[i] * Md1[i])+ ((hb[i] * hf[i+1]**2 - hf[i+1] * hb[i]**2) / 24) * (M[i]**2 * Md1[i] - M[i]**2 * Ut[i] + 2 * M[i] * Ut1[i] - M[i] * Md1[i])
       ci[i] = ((hf[i+1] - hb[i]) / 3) * (Ut1[i]) + ((hb[i]**2 + hf[i+1]**2 - hb[i] * hf[i+1]) / 12) * (Ut2[i] - M[i] * Ut1[i]) + (hb[i] * hf[i+1] / 6) * (M[i] * Ut1[i]) + ((hb[i] * hf[i+1]**2 - hf[i+1] * hb[i]**2) / 24) * (M[i] * Ut2[i] - M[i]**2 * Ut1[i]) + Ut[i]
       di[i] = ((hf[i+1] - hb[i]) / 3) * Vt1[i]+ ((hb[i]**2 + hf[i+1]**2 - hb[i] * hf[i+1]) / 12) * (Vt2[i] - M[i] * Vt1[i])+ (hb[i] * hf[i+1] / 6) * M[i] * Vt1[i]+ ((hb[i] * hf[i+1]**2 - hf[i+1] * hb[i]**2) / 24) * (M[i] * Vt2[i] - M[i]**2 * Vt1[i])+ Vt[i]
    
    # for tridiagonal system 
    p = np.zeros(n)
    q = np.zeros(n)
    r = np.zeros(n)
    s = np.zeros(n)
    rt = np.zeros(n)
    st = np.zeros(n)
    
    for i in range(0,n-1):
        p[i] = - 2*ai[i]/(hb[i]*hf[i+1])
        q[i] = -bi[i]*((hb[i])**2 * (hf[i+1]**2))/(hb[i]*hf[i+1])
        r[i] = -2*ai[i]/(hb[i]*(hb[i]+hf[i+1]))
        rt[i] = 2*ai[i]/(hb[i]*(hb[i]+hf[i+1]))
        s[i] = bi[i]*hf[i+1]/(hb[i]*(hb[i]+hf[i+1]))
        st[i] = bi[i]*hb[i]/(hf[i+1]*(hb[i]+hf[i+1]))
    
    vi = np.zeros(n)
    wi = np.zeros(n-1)
    wti = np.zeros(n)
    for i in range(n):
        vi[i] = p[i]+q[i]+ci[i]
    for i in range(1,n):
        wi[i-1] = r[i]-s[i]
        wti[i] = rt[i]-st[i]
    
    # Sparse matrix A setup
    A = diags([wi, vi, wti], [-1,0,1], shape=(n, n)).toarray()
    A[0,1] = -p[0]+q[0]
    R= np.zeros(n)
    for i in range(n):
        R[i] = -di[i]    
    S = np.linalg.solve(A,R)
    Y = S
    print(S)
    # Adjust X to match the dimensions of S
    plt.figure(figsize=(8, 6))
    plt.plot(X[:n], S, marker='o', linestyle='-', color='b', label="Solution S")
    plt.xlabel("x")
    plt.ylabel("S (Solution)")
    plt.title("Numerical Solution")
    plt.legend()
    plt.grid()
    plt.show()

