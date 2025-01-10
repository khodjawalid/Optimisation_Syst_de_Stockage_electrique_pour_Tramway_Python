import numpy as np 
import random



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



################### Partie algorrithme génétique #######################
#Fxtraction des  fronts de pareto, rang 1,2....
def pareto_sort(cost_matrix):
    """
    Cette fonction trie les solutions en fonction des fronts de Pareto.
    
    cost_matrix : Matrice contenant les critères des solutions (une solution par ligne)
    Retourne : Liste des fronts de Pareto (chaque front est une liste d'indices)
    """
    # Nombre de solutions dans la matrice
    num_solutions = cost_matrix.shape[0]
    if num_solutions == 0:  # Si aucune solution n'est présente
        return []

    # Initialisation
    pareto_fronts = [[]]  # Liste pour stocker les fronts de Pareto
    domination_counts = np.zeros(num_solutions, dtype=int)  # Compteur de domination pour chaque solution
    dominates = [[] for _ in range(num_solutions)]  # Liste des solutions dominées par chaque solution

    # Comparer chaque solution avec toutes les autres
    for i in range(num_solutions):
        for j in range(num_solutions):
            
            # Si la solution i domine la solution j
            if np.all(cost_matrix[i] <= cost_matrix[j]) and np.any(cost_matrix[i] < cost_matrix[j]):
                dominates[i].append(j)
            # Si la solution j domine la solution i
            elif np.all(cost_matrix[j] <= cost_matrix[i]) and np.any(cost_matrix[j] < cost_matrix[i]):
                domination_counts[i] += 1

        # Si la solution i n'est dominée par aucune autre
        if domination_counts[i] == 0:
            pareto_fronts[0].append(i)


    # Identifier les fronts suivants
    current_front = 0
    while len(pareto_fronts[current_front]) > 0:
        next_front = []  # Front suivant
        for i in pareto_fronts[current_front]:
            for j in dominates[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        current_front += 1
        pareto_fronts.append(next_front)

    # Supprimer le dernier front vide
    if len(pareto_fronts[-1]) == 0:
        pareto_fronts.pop()

    return pareto_fronts


#**Opérations génétiques**
def croisement(parent1, parent2):
    """
    Effectue un croisement linéaire entre deux parents.
    """
    random_boolean = random.choice([True, False])

    if random_boolean :
        return [parent1[0],parent2[1]]
    else : 
        return [parent2[0],parent1[1]]


def mutation(solution, mutation_rate=0.1):
    """
    Applique une mutation aléatoire à une solution.
    """
    alpha = random.uniform(-0.1, 1.1)
    gamma = random.uniform(-0.1, 1.1)

    beta = random.choice([True, False])

    if beta : 
        return solution
    else : 
        return [alpha*solution[0], gamma*solution[1]]





def algo_genetique(batt_capa,p_seuil,chutte_max,N) : 
    # Chaque solution est une ligne contenant [capacité de batterie, chute de tension max]
    Liste_solutions = np.vstack((batt_capa, chutte_max)).T


    # **Appliquer le tri par domination pour trouver les fronts de Pareto**
    pareto_fronts = pareto_sort(Liste_solutions)


    # **Sélection d'une sous-population pour l'évolution**
    population_size = N // 2  # Taille de la sous-population (50 % de la population initiale)
    selected_population = []  # Liste pour stocker les indices des solutions sélectionnées
    current_population_count = 0  # Compteur pour suivre la taille actuelle de la sous-population
    remaining_indices = list(range(len(batt_capa)))  # Indices des solutions restantes

    # Remplir la sous-population en priorisant les fronts de Pareto
    while current_population_count < population_size:
        # Sélectionner les objectifs des solutions restantes
        remaining_objectives = Liste_solutions[remaining_indices]

        # Ré-appliquer le tri de Pareto
        sorted_fronts = pareto_sort(remaining_objectives)

        # Ajouter les solutions des fronts dans la sous-population
        for front in sorted_fronts:
            if current_population_count + len(front) <= population_size:
                # Ajouter toutes les solutions du front
                selected_population.extend([remaining_indices[i] for i in front])
                current_population_count += len(front)
            else:
                # Ajouter juste assez pour atteindre la taille désirée
                remaining_count = population_size - current_population_count
                selected_population.extend([remaining_indices[i] for i in front[:remaining_count]])
                current_population_count = population_size

        # Retirer les solutions sélectionnées des indices restants
        remaining_indices = [i for i in remaining_indices if i not in selected_population]


    # **Créer la prochaine génération**
    selected_solutions = Liste_solutions[selected_population]
    next_generation = []

    while len(next_generation) < N:
        # Sélectionner deux parents au hasard
        indexs = np.random.choice(selected_population, 2, replace=False)
        parent_a, parent_b = [p_seuil[indexs[0]],batt_capa[indexs[0]]], [p_seuil[indexs[1]],batt_capa[indexs[1]]]

        # Appliquer le croisement
        croi_a = croisement(parent_a, parent_b)
        croi_a = mutation(croi_a)


        #Critère de distance 
        seuil_distance_y = 200000/5 #Longueur d'un segment 
        seuil_distance_x = 1800000/5

        #Facteuur de pondération
        ponderation = seuil_distance_y/seuil_distance_x

        #Le point pondéré afin de le comparer 
        croi_weighted = np.array(croi_a)*[1,ponderation]


        if np.array(next_generation).size == 0 : 
            distances = 2*seuil_distance_y
        else : 
            gen_weighted = np.array(next_generation)*[1,ponderation]
            # Calcul des distances euclidiennes
            distances = np.linalg.norm(gen_weighted - croi_weighted, axis=1)

        # Trouver la distance minimale
        min_distance = np.min(distances)

        if min_distance < seuil_distance_y : 
            continue  
        else : 
            # Ajouter les enfants à la prochaine génération
            next_generation.append(croi_a)

    next_generation = np.array(next_generation)

    return next_generation