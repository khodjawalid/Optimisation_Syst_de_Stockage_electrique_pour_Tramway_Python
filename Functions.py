import numpy as np 




# Fonction pour simuler le système et calculer la chute de tension maximale
def simulation_sys(capacite, seuil_puissance , Ptrain, Etrain, temps, Vsst, Req) : 
    """
    Simule le comportement du système et calcule la chute de tension maximale.

    Inputs:
    - capacite (float): Capacité maximale de la batterie (en kWh).
    - seuil_puissance (float): Seuil de puissance admissible (W).
    - Ptrain (list[float]): Puissance consommée ou produite par le train (en W).
    - Etrain (list[float]): Énergie consommée ou produite cumulée par le train (en kWh).
    - temps (list[int]): Liste des indices correspondant aux pas de temps.
    - Vsst (float): Tension de la sous-station (en volts).
    - Req (float): Résistance équivalente du système (en ohms).

    Outputs:
    - float: Chute de tension maximale observée (en volts).
    """

    # Initialisation des paramètres
    capa = capacite  # Capacité maximale de la batterie (en énergie)
    seuil = seuil_puissance  # Seuil de chute de tension admissible
    rend_batt = 0.9

    # Variables initiales
    Ebatt = 0  # Énergie actuelle dans la batterie
    Pbatt = np.zeros(len(Etrain))  # Puissance fournie ou absorbée par la batterie
    Prhe = np.zeros(len(Etrain))  # Puissance dissipée par le rhéostat
    Plac = np.zeros(len(Etrain))  # Puissance fournie par la ligne aérienne de contact (LAC)

    # Simulation sur la durée du trajet
    for t in temps[:-2] :
        if Ptrain[t] <= 0 : #durant le freinage

            if Ebatt < capa :  #si la batterie n'est pas complètement chargée 
                Pbatt[t] = Ptrain[t]  #Gestion de la batterie 
                Ebatt -= (Etrain[t+1] - Etrain[t])*rend_batt  #mettre à jour l'energie de la batterie avec un rendement 
                
            elif Ebatt == capa :  #La batterie est complètement chargée 
                Prhe[t] = Ptrain[t] 

            else :   #Remettre la bonne valeur de l'énergie 
                #print("L'énnergie de la batterie a depassé la capacité, remise à niveau")
                Ebatt = capa
                Prhe[t] = Ptrain[t] 


        if Ptrain[t] > 0 : #Durant l'accélération 
            if Ptrain[t] > seuil:
                if Ebatt > 0 :
                
                    Pbatt[t] = Ptrain[t] - seuil  #Gestion de la batterie 
                    Plac[t] = seuil
                    
                    E_nec = (Etrain[t+1] - Etrain[t])*(Pbatt[t]/Ptrain[t]) #Energie nécéssaire
                    E_dep = E_nec*rend_batt  #Energie dépensée par la batterie pour fournir de l'énergie 
                    Ebatt -= E_dep  #mettre à jour l'energie de la batterie avec un rendement 
                else :
                    Plac[t] = Ptrain[t] #On utilise la sst 

            if Ptrain[t] <= seuil : #Si on respecte le seuil ou bien la batterie est déchargée 
                Plac[t] = Ptrain[t] #On utilise la sst 

    
    # Calcul de la tension aux bornes du train avec la batterie
    #Vtrainbatt = 0.5 * Vsst + 0.5 * np.sqrt(Vsst**2 - 4 * Req * Plac)

    # Calcul de l'expression sous la racine
    expression = Vsst**2 - 4 * Req * Plac

    # Initialiser Vtrainbatt avec Vsst 
    Vtrainbatt = np.ones_like(expression)*Vsst

    # Calculer uniquement pour les indices valides (expression >= 0)
    valid_indices = expression >= 0
    Vtrainbatt[valid_indices] = 0.5 * Vsst + 0.5 * np.sqrt(expression[valid_indices])


    # Calcul de la chute de tension à chaque instant
    chutebatt = np.array([abs(Vsst - v) for v in Vtrainbatt])

    # Retourner la chute de tension maximale
    return np.max(chutebatt)