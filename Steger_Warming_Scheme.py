#A program for the standard Sod shock tube problem using Steger Warming Scheme

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gamma=1.4

# Importing and extracting analytic data
std_data=pd.read_csv("StandardSod_0_15.txt",sep=' ')
x_std=np.array(std_data['x'])
rho_std=np.array(std_data['rho'])
u_std=np.array(std_data['u'])
p_std=np.array(std_data['p'])
ie_std=np.array(std_data['ie'])

#Giving user input for the left and right-side conditions of Riemann problem
Wl=np.zeros(3)                                #[density  velocity  pressure]   
Wr=np.zeros(3)
Wl[0]=float(input("Input left density :"))
Wl[1]=float(input("Input left velocity :"))
Wl[2]=float(input("Input left pressure :"))
Wr[0]=float(input("Input right density :"))
Wr[1]=float(input("Input right velocity :"))
Wr[2]=float(input("Input right pressure :"))

Ul=np.zeros(3)                  #[rho    rhoU    rhoE]
Ur=np.zeros(3)
Ul[0]=Wl[0]
Ur[0]=Wr[0]
Ul[1]=Wl[0]*Wl[1]
Ur[1]=Wr[0]*Wr[1]
Ul[2]=Wl[0]*(Wl[1]**2)*0.5 + Wl[2]/(gamma-1)
Ur[2]=Wr[0]*(Wr[1]**2)*0.5 + Wr[2]/(gamma-1)

#Giving user input for no. of cells(N), length(L), total simulation time(t)
N=int(input("enter the no. of cells:"))
L=float(input("enter the length:"))
t=float(input("enter the total simulation time:"))
dx=L/N                  #Calculating delta x
CFL=0.8

#Formulation of grid point
xl=0
xr=1
x=np.zeros(N)
xnodes=np.linspace(xl,xr,N+1)
for i in range(N):
    x[i]=(xnodes[i]+xnodes[i+1])/2
dx=xnodes[2]-xnodes[1]

#Initializing the domain at zero time step
U = np.zeros((3,N))
for i in range(N):
    if x[i]<0.5:
        for j in range(3):
            U[j][i]=Ul[j]           
    else:
        for j in range(3):
            U[j][i] = Ur[j]         


a=np.zeros(N)
H=np.zeros(N)
flux = np.zeros((3,N))
fluxP = np.zeros((3,N))
fluxM = np.zeros((3,N))
fluxFace = np.zeros((3,N + 1))
lam=np.zeros((3,N))
lamP=np.zeros((3,N))
lamM=np.zeros((3,N))
time=0
#Iterative solving       
while(time<=t):
    for i in range(N):
        a[i] = (((U[2][i]/U[0][i])-(0.5*(U[1][i]/U[0][i])**2))*gamma*(gamma-1))**0.5                  #Computing the speed of sound
        H[i] = ((gamma*U[2][i]) - 0.5 * (gamma-1)*((U[1][i] **2 ) / U[0][i])) / U[0][i]               #Computing the H value at each iteration
        #Computing the eigenvalues
        lam[0][i] = U[1][i]/U[0][i] - a[i]
        lam[1][i] = U[1][i]/U[0][i] 
        lam[2][i] = U[1][i]/U[0][i] + a[i]
        #Computing the positive eigenvalues
        lamP[0][i] = 0.5*(lam[0][i]+abs(lam[0][i]))
        lamP[1][i] = 0.5*(lam[1][i]+abs(lam[1][i]))
        lamP[2][i] = 0.5*(lam[2][i]+abs(lam[2][i]))
        #Computing the negative eigenvalues
        lamM[0][i]  = 0.5*(lam[0][i]-abs(lam[0][i]))
        lamM[1][i]  = 0.5*(lam[1][i]-abs(lam[1][i]))
        lamM[2][i]  = 0.5*(lam[2][i]-abs(lam[2][i]))
    
    #Computing the flux at each interface
    for i in range(N):
        fluxP[0][i] = U[0][i]*(lamP[0][i]+2*(gamma-1)*lamP[1][i]+lamP[2][i])/(2*gamma) 
        fluxP[1][i] = U[0][i]*(lamP[0][i]*(U[1][i]/U[0][i]-a[i])+2*(gamma-1)*lamP[1][i]*U[1][i]/U[0][i]+(U[1][i]/U[0][i]+a[i])*lamP[2][i])/(2*gamma) 
        fluxP[2][i] = U[0][i]*(lamP[0][i]*(H[i]-a[i]*U[1][i]/U[0][i])+(gamma-1)*(U[1][i]/U[0][i])**2*lamP[1][i]+(H[i]+a[i]*U[1][i]/U[0][i])*lamP[2][i])/(2*gamma)
        fluxM[0][i] = U[0][i]*(lamM[0][i]+2*(gamma-1)*lamM[1][i]+lamM[2][i])/(2*gamma) 
        fluxM[1][i] = U[0][i]*(lamM[0][i]*(U[1][i]/U[0][i]-a[i])+2*(gamma-1)*lamM[1][i]*U[1][i]/U[0][i]+(U[1][i]/U[0][i]+a[i])*lamM[2][i])/(2*gamma) 
        fluxM[2][i] = U[0][i]*(lamM[0][i]*(H[i]-a[i]*U[1][i]/U[0][i])+(gamma-1)*(U[1][i]/U[0][i])**2*lamM[1][i]+(H[i]+a[i]*U[1][i]/U[0][i])*lamM[2][i])/(2*gamma) 

    for i in range(1,N):
        fluxFace[:, i] = fluxP[:, i-1] + fluxM[:, i]                         #Computing the F(i+(1/2)) = Fi(plus) + Fi+1(minus)
    fluxFace[:,0]=fluxFace[:,1]                                              #Giving the boundary condition at the both sides
    fluxFace[:,N] = fluxFace[:,N-1]  

    maxlam = np.max(abs(lam))            #Computing the maximum eigenvalue
    for i in range(N):
        U[:, i] = U[:, i] - (CFL / maxlam) * (fluxFace[:, i+1] - fluxFace[:, i])
    dt = CFL * dx / maxlam           #Computing the time step for each iteartion
    time = time + dt      

'''print("x\trho\tu\tp\tie\n")
for i in range(N):
    print(x[i],U[0][i],U[1][i]/U[0][i],(gamma-1)*(U[2][i]-0.5*U[1][i]**2/U[0][i]),(2*U[2][i]*U[0][i]-U[1][i]**2)/(2*U[0][i]**2))

#Ploting the numerical and analytical density, velocity, pressure and energy profiles
plt.plot(x,U[0,:],color='Red',linestyle='',marker='o',markersize=1.2,label='Density')
plt.plot(x_std,rho_std,color='Black',label='Density')
plt.title("Density")
plt.show()
plt.close()
plt.plot(x,U[1,:]/U[0,:],color='Blue',linestyle='',marker='o',markersize=1.2,label='Velocity')
plt.plot(x_std,u_std,color='Black',label='velocity')
plt.title("Velocity")
plt.show()
plt.close()
plt.plot(x,(gamma-1)*(U[2,:]-(0.5*(U[1,:]**2)/U[0,:])),color='Green',linestyle='',marker='o',markersize=1.2,label='Pressure')
plt.plot(x_std,p_std,color='Black',label='Pressure')
plt.title("Pressure")
plt.show()
plt.close()
plt.plot(x,(2*U[2,:]*U[0,:]-U[1,:]**2)/(2*U[0,:]**2),color='Orange',linestyle='',marker='o',markersize=1.2,label='Energy')
plt.plot(x_std,ie_std,color='Black',label='Energy')
plt.title("Energy")
plt.show()
plt.close()''' 




        
        
