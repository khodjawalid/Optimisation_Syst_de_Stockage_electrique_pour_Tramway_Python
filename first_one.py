import numpy as np
from matplotlib import pyplot as plt

# Ouvrir le fichier en mode lecture
with open("marche.txt", "r") as f:
    data = f.readlines()  # Liste contenant chaque ligne du fichier

# Optionnel : Supprimer les sauts de ligne
data = [ligne.strip().split('\t') for ligne in data]

temps = [t[0] for t in data]    
position = [x[1] for x in data]

position = np.array(list(map(float, position)))
temps = np.array(list(map(int, temps)))

vitesse =[position[i+1]-position[i] for i in range(len(position)-1)]
vitesse.append(vitesse[-1])
vitesse = np.array(vitesse)

acc =[vitesse[i+1]-vitesse[i] for i in range(len(vitesse)-1)]
acc.append(acc[-1])
acc = np.array(acc)


#calcul de ptrain
##calcul  de fmotrice 
M=70*1000  # conversion tonne/kg
A0 = 780
A1 = 6.4/1000   # conversion tonne/kg
B0 = 0
B1 = 3.6*0.14/1000  # conversion tonne/kg et kmh/ms
C0 = 0.3634*(3.6)**2 #conversion kmh/ms
C1=0
Fr= (A0 + A1*M) + (B0 + B1*M)*vitesse +(C0 + C1*M)*vitesse**2
Fmot = M*acc + Fr
Pmec = Fmot*vitesse
rend =0.8


Pelec = np.array([Pmec[i]/rend if Pmec[i] >=0 else Pmec[i]*rend for i in range(len(vitesse))])

Ptrain = Pelec + 35*1000


plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.plot(temps,position) 
plt.title("position en fonction du temps") 
plt.grid()

plt.subplot(2,2,2)
plt.plot(temps,vitesse)
plt.title("vitesse en fonction du temps") 
plt.grid()

plt.subplot(2,2,3)
plt.plot(temps,acc)
plt.title("accélération en fonction du temps") 
plt.grid()

plt.subplot(2,2,4)
plt.plot(temps,Ptrain)
plt.title("Ptrain en fonction du temps") 
plt.grid()
plt.show() 


Rsst = 33e-3
rolac = 131e-6
rorail = 18e-6
Rlac1 = rolac*position 
Rlac2 = rolac*(position[-1]-position)
Rrail1 = rorail*position
Rrail2 = rorail*(position[-1]-position)
Vsst=790

Req =  (Rsst + Rrail1 + Rlac1)*(Rsst + Rrail2 + Rlac2)/(2*Rsst + Rrail1 + Rlac1 + Rrail2 + Rlac2)

Vtrain = 0.5*Vsst + 0.5*np.sqrt(Vsst**2 - 4*Req*Ptrain)

chute = [abs(Vsst - v) for v in Vtrain]

plt.figure()
plt.plot(position,Vtrain)
plt.plot(position,chute)
#plt.xlim([0 , 5000])
plt.grid()
plt.show()


## calcul de la tension avec batterie 

# Calcul de l'intégrale cumulée avec numpy (methode des trapezes)
Etrain = np.cumsum(np.diff(temps) * (Ptrain[:-1] + Ptrain[1:]) / 2)
# On ajoute un 0 initial pour que l'intégrale commence à 0
Etrain = np.insert(Etrain, 0, 0)

plt.figure()
plt.plot(temps,Etrain)
#plt.xlim([0 , 5000])
plt.title("Energie du train en fonction du temps")
plt.grid()
plt.show()

capa = 4e6
Ebatt = 0
Pbatt = np.zeros(len(Etrain))
Prhe = np.zeros(len(Etrain))
Plac = np.zeros(len(Etrain))

flag1 = True  #pour charger la batterie
flag2 = True  #pour decharger la batterie 

for t in temps[:-2] :
    if Ptrain[t] <= 0 : #durant le freinage
        if flag1 :   #fixer le point du debut de freinage afin de soustraire l'énergie commulée avant 
            t0 = t 
            flag1 = False
        if Ebatt < capa :  #si la batterie n'est pas complètement chargée 
            Pbatt[t] = Ptrain[t]  #Gestion de la batterie 
            Ebatt -= (Etrain[t+1] - Etrain[t0])  #mettre à jour l'energie de la batterie 
            
        elif Ebatt == capa :
            Prhe[t] = Ptrain[t] 
        else :
            print("Erreur énergie a depassé la capacit  dans la batterie")
            Ebatt = capa

    flag1 =True # on met le flag à True afn de détecter le prochaain début d'un cycle de freinage 

    if Ptrain[t] > 0 :
        if chute[t] > 100 and Ebatt > 0 :
            if flag2 :
                t1 = t
                flag2 = False
            Pbatt[t] = Ptrain[t]
            Ebatt -= (Etrain[t+1] - Etrain[t1])

        flag2 = True

        if chute[t] <= 100 or Ebatt <= 0 :
            Plac[t] = Ptrain[t] 

        


plt.figure(figsize=(8,10))
plt.subplot(2,2,1)
plt.title("Ptrain")
plt.plot(position,Ptrain)
plt.grid()
plt.subplot(2,2,2)
plt.title("Pbatt")
plt.plot(position,Pbatt)
plt.grid()
plt.subplot(2,2,3)
plt.title("Prheostat")
plt.plot(position,Prhe)
plt.grid()
plt.subplot(2,2,4)
plt.title("Plac")
plt.plot(position,Plac)
plt.grid()
plt.show()




Vtrainbatt = 0.5*Vsst + 0.5*np.sqrt(Vsst**2 - 4*Req*Plac)

plt.figure()
plt.plot(position,Vtrain,label="tension sans batterie")
plt.plot(position,Vtrainbatt,label="tension avec batterie")
plt.legend()
plt.grid()
plt.show()
