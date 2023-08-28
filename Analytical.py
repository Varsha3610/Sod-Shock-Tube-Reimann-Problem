#A program to calculate p∗, u∗, ρ∗L and ρ∗R for a given Riemann problem for the Euler equations using Newton-Raphson iterative scheme

import numpy as np

gamma =1.4
#Giving user input for the left and right-side conditions of Riemann problem
Wl=np.zeros(3)
Wr=np.zeros(3)
Wl[0]=float(input("Input left density :"))
Wl[1]=float(input("Input left velocity :"))
Wl[2]=float(input("Input left pressure :"))
Wr[0]=float(input("Input right density :"))
Wr[1]=float(input("Input right velocity :"))
Wr[2]=float(input("Input right pressure :"))

fp=open('Result.txt','w')               #opening a file to write the results

#Calculating the speed of sound
al= (gamma*Wl[2]/Wl[0])**(0.5)
ar= (gamma*Wr[2]/Wr[0])**(0.5)

#Computing pressure value on left and right hand side of contact discontinuity and also the speed of contact discontinuity
#Setting initial guess for pstar
if(Wl[2]!=Wr[2]):
    pstar=(Wl[2]+Wr[2])/2
else:
    pstar = 0.000001

iteration=0
error=1
Fl=0.0
Fr=0.0
Flprime=0.0
Frprime=0.0
#print("Iteration         pstar                              Fr                                  Fl")
#Using Newton-Raphson iterative scheme
while(error>.000001):               #convergence criteria
    if(pstar<=Wl[2]):
        Fl= (2*al/(gamma -1))*((pstar/Wl[2])**((gamma-1)*0.5/gamma)-1)
        Flprime = (1/(Wl[0]*al))*(pstar/Wl[2])**(-(gamma+1)*0.5/gamma)
    else:
        Fl= (pstar - Wl[2])*((2/((gamma+1)*Wl[0]))/(pstar +((gamma-1)*Wl[2]/(gamma+1))))**0.5
        Flprime = (1-((pstar-Wl[2])/(2*(pstar+((gamma-1)*Wl[2]/(gamma+1))))))*((2/((gamma+1)*Wl[0]))/(pstar +((gamma-1)*Wl[2]/(gamma+1))))**0.5
    if(pstar<=Wr[2]):
        Fr= (2*ar/(gamma -1))*((pstar/Wr[2])**((gamma-1)*0.5/gamma)-1)
        Frprime = (1/(Wr[0]*ar))*(pstar/Wr[2])**(-(gamma+1)*0.5/gamma)
    else:
        Fr= (pstar - Wr[2])*((2/((gamma+1)*Wr[0]))/(pstar +((gamma-1)*Wr[2]/(gamma+1))))**0.5
        Frprime = (1-((pstar-Wr[2])/(2*(pstar+((gamma-1)*Wr[2]/(gamma+1))))))*((2/((gamma+1)*Wr[0]))/(pstar +((gamma-1)*Wr[2]/(gamma+1))))**0.5

    F = Fr +Fl +Wr[1] - Wl[1]          #function to calculate the value of pstar            
    Fprime = Frprime + Flprime         #taking derivative of the function
    pstarnew = pstar - F/Fprime        #calculating pstar using Newton Raphson Method
    error = abs(pstarnew-pstar)        #calculating absolute error
    iteration=iteration+1
    pstar=pstarnew
    #print(iteration,"          ",pstar,"          ",Fr,"          ",Fl)
    ustar=((Wl[1]+Wr[1]) + (Fr-Fl))/2           #calculating ustar           
print('The value of pstar is=',pstar, '\nThe value of ustar is=',ustar,file=fp)

#Wave Structure
if(pstar>Wl[2]):
    print('Shock on left hand side of contact discontinuity',file=fp)
    rhostarL = Wl[0]*((pstar/Wl[2] + (gamma-1)/(gamma+1))/((pstar/Wl[2])*(gamma-1)/(gamma+1) + 1))
    SL = Wl[1] - al*(((gamma+1)*(pstar/(2*gamma*Wl[2]))+(gamma-1)/(2*gamma))**0.5)
    print('Rho Star Left=',rhostarL,file=fp)                     
    print('The speed of left moving shock is=',SL,file=fp)

else:
    print('Rarefaction on left hand side of contact discontinuity',file=fp)
    rhostarL = Wl[0]*(pstar/Wl[2])**(1/gamma)
    astarL = al*(pstar/Wl[2])**((gamma-1)/(2*gamma))
    SHL = Wl[1] - al
    STL = ustar -astarL
    print('Rho Star Left=',rhostarL,file=fp)
    print('The speeds of left moving rarefaction head and tail are=',SHL,STL,file=fp)

if(pstar>Wr[2]):
    print('Shock on right hand side of contact discontinuity',file=fp)
    rhostarR = Wr[0]*((pstar/Wr[2] + (gamma-1)/(gamma+1))/((pstar/Wr[2])*(gamma-1)/(gamma+1) + 1))
    SR = Wr[1] + ar * (((gamma + 1) * (pstar / (2 * gamma * Wr[2])) + (gamma - 1) / (2 * gamma)) ** 0.5)
    print('Rho Star Right=',rhostarR,file=fp)
    print('The speed of right moving shock is=',SR,file=fp)

else:
    print('Rarefaction on right hand side of contact discontinuity',file=fp)
    rhostarR = Wr[0]*(pstar/Wr[2])**(1/gamma)
    astarR = ar*(pstar/Wr[2])**((gamma-1)/(2*gamma))
    SHR = Wr[1] + ar
    STR = ustar + astarR
    print('Rho Star Right=',rhostarR,file=fp)
    print('The speeds of right moving rarefaction head and tail are=',SHR,STR,file=fp)
print("value of pstar: ",pstar)
print("value of ustar: ",ustar)
print("Rho Star Left: ",rhostarL)
print("Rho Star Right:",rhostarR)
fp.close()


