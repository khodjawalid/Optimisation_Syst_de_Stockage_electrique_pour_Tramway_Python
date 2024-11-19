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
temps = np.array(list(map(float, temps)))

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

print(Ptrain)


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

