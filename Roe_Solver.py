#A program for the standard Sod shock tube problem using Roe Solver(with entropy fix)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gamma=1.4
epsilon = 10**(-6)

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
CFL = 0.8

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

time=0
#Iterative solving       
while(time<=t):
    aTelda=0
    uTelda=0
    HTelda=0
    lam =np.zeros((3,N))
    eigenK = np.zeros((3,3,N))
    delU1 = np.zeros(N)
    delU2 = np.zeros(N)
    delU3 = np.zeros(N)
    delTelda = np.zeros((3, N))
    
    for i in range(N-1):
        Hr = ((gamma*U[2][i + 1]) - 0.5 * (gamma-1)*((U[1][i + 1] **2 ) / U[0][i + 1])) / U[0][i + 1]
        Hl = ((gamma*U[2][i]) - 0.5 * (gamma-1)*((U[1][i] **2)/U[0][i])) / U[0][i]
        uTelda = ((U[0][i+1]**.5 * U[1][i + 1] / U[0][i + 1]) + (U[0][i]**.5 * U[1][i] / U[0][i]))/(U[0][i+1]**.5 + U[0][i]**.5)
        HTelda=(U[0][i+1]**.5 * Hr + U[0][i]**.5 * Hl)/(U[0][i+1]**.5 + U[0][i]**.5)
        aTelda = ((gamma - 1) * (HTelda - 0.5*(uTelda**2)))**0.5

        #Computing the eigenvalues
        lam[0][i+1] = uTelda - aTelda                        
        lam[1][i+1] = uTelda
        lam[2][i+1] = uTelda + aTelda

        #Computing the eigenvectors
        eigenK[0][0] = 1
        eigenK[0][1] = 1
        eigenK[0][2] = 1
        eigenK[1][0] = uTelda - aTelda
        eigenK[1][1] = uTelda
        eigenK[1][2] = uTelda + aTelda
        eigenK[2][0] = HTelda - (uTelda * aTelda)
        eigenK[2][1] = 0.5 * (uTelda ** 2)
        eigenK[2][2] = HTelda + (uTelda * aTelda)

        delU1[i+1] = U[0][i + 1] - U[0][i]
        delU2[i+1] = U[1][i + 1] - U[1][i]
        delU3[i+1] = U[2][i + 1] - U[2][i]

        delTelda[1][i+1] = (gamma - 1) * ((delU1[i+1] * (HTelda - (uTelda ** 2))) + (uTelda * delU2[i+1]) - delU3[i+1]) / (aTelda ** 2)
        delTelda[0][i+1] = ((delU1[i+1] * (uTelda + aTelda)) - delU2[i+1] - (aTelda * delTelda[1][i+1])) / (2 * aTelda)
        delTelda[2][i+1] = delU1[i+1] - delTelda[0][i+1] - delTelda[1][i+1]
    
    #Computing the flux at each interface
    flux = np.zeros((3, N ))
    fluxFace = np.zeros((3, N + 1))
    sum = np.zeros((3, N + 1))

    for i in range(N):
        flux[0][i] = U[1][i]
        flux[1][i] = ((3 - gamma)*0.5*(U[1][i]**2)/U[0][i]) + (gamma-1)*U[2][i]
        flux[2][i] = (gamma * U[1][i] * U[2][i] / U[0][i]) - ((gamma - 1) *0.5* (U[1][i] ** 3) / (U[0][i] ** 2))

    #Entropy Fix
    for i in range(N):
        for j in range(3):
            if (abs(lam[j][i])<epsilon):
                if(lam[j][i] !=0):
                    lam[j][i] = 0.5*((lam[j][i]**2/epsilon) + epsilon)
    
    for i in range(1,N):
        for j in range(3):
            if(lam[j][i]<0):
                sum[:, i] = sum[:, i] + (lam[j][i]) * delTelda[j][i] * eigenK[:, j, i]
                fluxFace[:, i] = flux[:, i-1] + sum[:, i]               #Computing the F(i+(1/2)) = Fi + sum
                fluxFace[:,0]=fluxFace[:,1]                             #Giving the boundary condition at the both sides
                fluxFace[:,N] = fluxFace[:,N-1]   
    
    maxlam = np.max(abs(lam))                 #Computing the maximum eigenvalue
    for i in range(N):
        U[:, i] = U[:, i] - (CFL / maxlam) * (fluxFace[:, i+1] - fluxFace[:, i])
    dt = CFL * dx / maxlam           #Computing the time step for each iteartion
    time = time + dt      
'''print("x\trho\tu\tp\tIE\n")
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


