import numpy as np 
import random
from matplotlib import pyplot as plt 


########################### Parti construction système ###############################
######################################################################################

def Affichage_puissance(Ptrain, Pbatt, Etrain, Ebatt, Prhe, Plac, position, temps, capa, rend_batt, seuil_puissance):
    """
    Cette fonction simule et affiche les puissances consommées et échangées par un train
    en tenant compte de la batterie, de la rhéostat et de la Ligne d'Alimentation en Courant (LAC).
    
    Paramètres :
    ------------
    - Ptrain : tableau des puissances consommées par le train à chaque instant.
    - Pbatt : tableau des puissances échangées avec la batterie.
    - Etrain : tableau de l'énergie consommée par le train à chaque instant.
    - Ebatt : énergie actuelle stockée dans la batterie.
    - Prhe : tableau des puissances dissipées par la rhéostat.
    - Plac : tableau des puissances fournies par la LAC.
    - position : tableau des positions du train à chaque instant.
    - temps : tableau des instants de simulation.
    - capa : capacité maximale de la batterie (en énergie).
    - rend_batt : rendement de la batterie (efficacité énergétique).
    - seuil_puissance : puissance maximale que la LAC peut fournir.

    Sorties :
    ---------
    La fonction produit des graphiques montrant :
    - La puissance consommée par le train.
    - La puissance échangée avec la batterie.
    - La puissance dissipée par la rhéostat.
    - La puissance fournie par la LAC.
    - Une comparaison entre la puissance consommée et la somme des puissances calculées.
    """

    # Parcourir tous les instants de simulation sauf les deux derniers
    for t in temps[:-2]:
        # **Durant le freinage**
        if Ptrain[t] <= 0:
            if Ebatt < capa:  # Si la batterie n'est pas complètement chargée
                Pbatt[t] = Ptrain[t]  # La batterie récupère la puissance
                Ebatt -= (Etrain[t + 1] - Etrain[t]) * rend_batt  # Mise à jour de l'énergie stockée avec rendement

            elif Ebatt == capa:  # Si la batterie est pleine
                Prhe[t] = Ptrain[t]  # Toute la puissance est dissipée dans la rhéostat

            else:  # Cas improbable où l'énergie dépasse la capacité maximale
                Ebatt = capa  # Correction en fixant l'énergie à la capacité maximale
                Prhe[t] = Ptrain[t]

        # **Durant l'accélération**
        if Ptrain[t] > 0:
            if Ptrain[t] > seuil_puissance:  # Lorsque la puissance dépasse le seuil de la LAC
                if Ebatt > 0:  # Si la batterie a de l'énergie
                    Pbatt[t] = Ptrain[t] - seuil_puissance  # Complément fourni par la batterie
                    Plac[t] = seuil_puissance  # La LAC fournit le seuil maximum

                    # Calcul de l'énergie nécessaire et dépensée
                    E_nec = (Etrain[t + 1] - Etrain[t]) * (Pbatt[t] / Ptrain[t])
                    E_dep = E_nec * rend_batt
                    Ebatt -= E_dep  # Mise à jour de l'énergie de la batterie

                else:  # Si la batterie est vide
                    Plac[t] = Ptrain[t]  # Toute la puissance vient de la LAC

            elif Ptrain[t] <= seuil_puissance:  # Si la puissance demandée est inférieure au seuil
                Plac[t] = Ptrain[t]  # La LAC fournit directement la puissance

    # **Affichage des graphiques**
    # 1. Puissance consommée par le train
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.title("Puissance consommée par le train")
    plt.plot(position, Ptrain * 1e-6)
    plt.ylabel('Puissance [MW]')
    plt.xlabel('Position [m]')
    plt.grid()
    plt.ylim(-1, 1.2)

    # 2. Puissance échangée avec la batterie
    plt.subplot(2, 2, 2)
    plt.title("Puissance consommée par la batterie")
    plt.plot(position, Pbatt * 1e-6)
    plt.ylabel('Puissance [MW]')
    plt.xlabel('Position [m]')
    plt.grid()
    plt.ylim(-1, 1.2)

    # 3. Puissance dissipée par la rhéostat
    plt.subplot(2, 2, 3)
    plt.title("Puissance consommée par la rhéostat")
    plt.plot(position, Prhe * 1e-6)
    plt.ylabel('Puissance [MW]')
    plt.xlabel('Position [m]')
    plt.grid()
    plt.ylim(-1, 1.2)

    # 4. Puissance fournie par la LAC
    plt.subplot(2, 2, 4)
    plt.title("Puissance fournie par la LAC")
    plt.plot(position, Plac * 1e-6)
    plt.ylabel('Puissance [MW]')
    plt.xlabel('Position [m]')
    plt.grid()
    plt.ylim(-1, 1.2)
    plt.show()

    # **Comparaison entre Ptrain et la somme des puissances**
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(position, (Plac + Prhe + Pbatt) * 1e-6)
    plt.title("Ptrain à partir de la somme des graphes de puissance")
    plt.ylabel('Puissance [MW]')
    plt.xlabel('Position [m]')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(position, Ptrain * 1e-6)
    plt.title("Ptrain à partir des calculs")
    plt.ylabel('Puissance [MW]')
    plt.xlabel('Position [m]')
    plt.grid()
    plt.show()




def simulation_sys(capacite, seuil_puissance, Ptrain, Etrain, temps, Vsst, Req):
    """
    Simule le comportement du système et calcule la chute de tension maximale.

    Paramètres :
    ------------
    - capacite (float) : Capacité maximale de la batterie (en kWh).
    - seuil_puissance (float) : Seuil de puissance admissible (en W).
    - Ptrain (list[float]) : Puissance consommée ou produite par le train à chaque instant (en W).
    - Etrain (list[float]) : Énergie cumulée consommée ou produite par le train (en kWh).
    - temps (list[int]) : Indices correspondant aux instants de temps de simulation.
    - Vsst (float) : Tension fournie par la sous-station (en volts).
    - Req (float) : Résistance équivalente du système (en ohms).

    Retour :
    --------
    - float : Chute de tension maximale observée (en volts).
    """

    # Initialisation des paramètres et variables
    capa = capacite  # Capacité maximale de la batterie
    seuil = seuil_puissance  # Seuil de puissance admissible
    rend_batt = 0.9  # Rendement de la batterie

    # Variables d'état initiales
    Ebatt = 0  # Énergie initiale de la batterie
    Pbatt = np.zeros(len(Etrain))  # Puissance échangée avec la batterie
    Prhe = np.zeros(len(Etrain))  # Puissance dissipée par le rhéostat
    Plac = np.zeros(len(Etrain))  # Puissance fournie par la LAC

    # Boucle de simulation pour chaque pas de temps
    for t in temps[:-2]:  # Exclut les deux derniers indices
        # Cas : Freinage (Ptrain[t] <= 0)
        if Ptrain[t] <= 0:
            if Ebatt < capa:  # Batterie non pleine
                Pbatt[t] = Ptrain[t]  # Stocker l'énergie du freinage dans la batterie
                Ebatt -= (Etrain[t + 1] - Etrain[t]) * rend_batt  # Mise à jour de l'énergie de la batterie

            elif Ebatt == capa:  # Batterie pleine
                Prhe[t] = Ptrain[t]  # Dissiper l'énergie excédentaire dans la rhéostat

            else:  # Cas anormal : énergie de la batterie dépasse sa capacité
                Ebatt = capa  # Correction en fixant l'énergie à la capacité maximale
                Prhe[t] = Ptrain[t]

        # Cas : Accélération (Ptrain[t] > 0)
        if Ptrain[t] > 0:
            if Ptrain[t] > seuil:  # Si la puissance dépasse le seuil admissible
                if Ebatt > 0:  # La batterie est utilisée pour fournir de l'énergie
                    Pbatt[t] = Ptrain[t] - seuil  # Batterie compense le surplus
                    Plac[t] = seuil  # La LAC fournit le seuil maximum admissible

                    # Calcul de l'énergie nécessaire et dépensée
                    E_nec = (Etrain[t + 1] - Etrain[t]) * (Pbatt[t] / Ptrain[t])
                    E_dep = E_nec * rend_batt
                    Ebatt -= E_dep  # Mise à jour de l'énergie de la batterie

                else:  # Batterie vide
                    Plac[t] = Ptrain[t]  # La LAC fournit toute la puissance demandée

            elif Ptrain[t] <= seuil:  # Si la puissance est inférieure au seuil
                Plac[t] = Ptrain[t]  # Toute la puissance est fournie par la LAC

    # Calcul de la tension aux bornes du train avec la batterie
    expression = Vsst**2 - 4 * Req * Plac  # Expression sous la racine
    Vtrainbatt = np.ones_like(expression) * Vsst  # Initialisation par la tension de la sous-station

    # Calcul uniquement pour les indices où l'expression est valide (>= 0)
    valid_indices = expression >= 0
    Vtrainbatt[valid_indices] = 0.5 * Vsst + 0.5 * np.sqrt(expression[valid_indices])

    # Calcul de la chute de tension à chaque instant
    chutebatt = np.array([abs(Vsst - v) for v in Vtrainbatt])

    # Retourner la chute de tension maximale
    return np.max(chutebatt)



########################### Parti Monte Carlo ##~####################################
######################################################################################

def pareto(capacite, chute, rang):
    """
    Détermine le rang de Pareto (jusqu'à un rang spécifié) pour un ensemble de solutions
    caractérisées par deux objectifs : 'capacite' et 'chute'.

    Paramètres :
    ------------
    - capacite : Liste des capacités .
    - chute : Liste des chutes de tension (deuxième objectif).
    - rang : int, Nombre maximal de fronts de Pareto à extraire.

    Retour :
    --------
    - list_rangs : list[list[int]]
        Liste de listes d'indices :
        - list_rangs[0] contient les indices des solutions de rang 0 (front de Pareto),
        - list_rangs[1] contient ceux du rang 1, etc.
        Chaque sous-liste correspond à un rang particulier.
    """

    # Création d'une liste de solutions (chaque solution est [capacite, chute])
    solutions = [np.array([i, j]) for i, j in zip(capacite, chute)]
    
    # nb_dom[i] stocke le nombre de solutions qui dominent la solution i
    nb_dom = np.zeros(len(solutions))
    
    # list_rangs[r] stockera la liste d'indices des solutions de rang r
    list_rangs = [[] for _ in range(rang)]
    
    # Calcul du nombre de dominations pour chaque solution
    for i, sol1 in enumerate(solutions):
        for j, sol2 in enumerate(solutions):
            # Condition de domination : sol2 <= sol1 sur tous les critères
            # ET sol2 < sol1 sur au moins un critère.
            # NB : On suppose ici qu'on veut minimiser capacite et chute.
            if np.all(sol2 <= sol1) and np.any(sol2 < sol1):
                nb_dom[i] += 1

    # Regroupement des solutions par rang
    # Rang d'une solution = nombre de solutions qui la dominent (ici, rang = nb_dom).
    for r in range(rang):
        for i in range(len(solutions)):
            if nb_dom[i] == r:
                list_rangs[r].append(i)

    return list_rangs





def affichage_Monte_Carlo(batt_capa, rangs, p_seuil, optimal, chute_sys):
    """
    Affiche les résultats de l'algorithme de Monte Carlo sous forme de graphiques :
    - Espace des solutions (capacité de la batterie vs seuil de puissance).
    - Espace des objectifs (capacité de la batterie vs chute de tension).

    Paramètres :
    ------------
    - batt_capa : list ou array-like
        Liste des capacités de la batterie (en joules).
    - rangs : list[list[int]]
        Liste de listes contenant les indices des solutions pour chaque rang de Pareto.
        - rangs[0] : indices des solutions du front de Pareto (rang 0).
        - rangs[1] : indices des solutions du rang 1, etc.
    - p_seuil : list ou array-like
        Liste des seuils de puissance correspondants (en watts).
    - optimal : int
        Indice de la solution optimale dans les données.
    - chute_sys : list ou array-like
        Liste des chutes de tension maximales associées à chaque solution (en volts).

    Retour :
    --------
    Aucun retour. La fonction génère deux graphiques :
    - Espace des solutions.
    - Espace des objectifs.
    """

    # **Affichage : Espace des solutions**
    list_labels = []
    plt.figure(figsize=(10, 4))
    for i in range(len(batt_capa)):
        # Attribuer une couleur selon le rang de la solution
        if i in rangs[0]:  # Front de Pareto (rang 0)
            couleur,label = 'red','Front de Parelo rang 1'
        elif i in rangs[1]:  # Rang 1
            couleur,label = 'green','Front de Parelo rang 2'
        else:  # Autres rangs
            couleur,label = 'blue','Population'

        if label in list_labels:
            plt.scatter(batt_capa[i]*1e-3/3600,p_seuil[i]*1e-6,color=couleur)
        else :
            plt.scatter(batt_capa[i]*1e-3/3600,p_seuil[i]*1e-6,color=couleur,label=label)
            list_labels.append(label)

    # Afficher la solution optimale en jaune
    plt.scatter(batt_capa[optimal] * 1e-3 / 3600, p_seuil[optimal] * 1e-6, color='yellow', label="Solution optimale")
    plt.title("Espace des solutions")
    plt.xlabel("Capacité de la batterie [kWh]")  # Conversion de Joules -> kWh
    plt.ylabel("Seuil de la puissance [MW]")  # Conversion de Watts -> MW
    plt.grid()
    plt.legend()
    plt.show()

    #Affichage espace objectifs
    list_labels = []
    plt.figure(figsize=(10,4))
    for i in range(len(batt_capa)):
        if i in rangs[0]:
            couleur,label = 'red','Front de Pareto rang 1'
        elif i in rangs[1]:
            couleur,label ='green', 'Front de Pareto rang 2'
        else :
            couleur,label = 'blue', 'Population'
        
        if label in list_labels:
            plt.scatter(batt_capa[i]*1e-3/3600,chute_sys[i],color=couleur)
        else:
            plt.scatter(batt_capa[i]*1e-3/3600,chute_sys[i],color=couleur,label=label)
            list_labels.append(label)


    plt.scatter(batt_capa[optimal]*1e-3/3600,chute_sys[optimal],color='yellow',label='Solution optimale')
    plt.ylabel("Chute maximale de tension (V)")
    plt.xlabel("Capacité de la batterie (kWs)")
    plt.grid()
    plt.legend(loc='right')




########################### Parti Algorithme génétique  ###############################
######################################################################################


def pareto_sort(cost_matrix):
    """
    Trie les solutions en fonction des fronts de Pareto.

    Paramètres :
    ------------
    - cost_matrix : Matrice où chaque ligne correspond à une solution

    Retour :
    --------
    - pareto_fronts : list[list[int]]
        Une liste de listes, où chaque sous-liste contient les indices des solutions
        appartenant à un front de Pareto donné. 
        - pareto_fronts[0] : indices des solutions non dominées (rang 0).
        - pareto_fronts[1] : indices des solutions du deuxième front, etc.
    """

    # Nombre total de solutions
    num_solutions = cost_matrix.shape[0]
    
    # Cas spécial : Pas de solutions
    if num_solutions == 0:
        return []

    # Initialisation des structures
    pareto_fronts = [[]]  # Liste pour stocker les fronts de Pareto
    domination_counts = np.zeros(num_solutions, dtype=int)  # Compteur de domination pour chaque solution
    dominates = [[] for _ in range(num_solutions)]  # Liste des solutions dominées par chaque solution

    # **Étape 1 : Comparer chaque solution avec toutes les autres**
    for i in range(num_solutions):
        for j in range(num_solutions):
            # Si la solution i domine la solution j
            if np.all(cost_matrix[i] <= cost_matrix[j]) and np.any(cost_matrix[i] < cost_matrix[j]):
                dominates[i].append(j)  # Ajouter j à la liste des solutions dominées par i

            # Si la solution j domine la solution i
            elif np.all(cost_matrix[j] <= cost_matrix[i]) and np.any(cost_matrix[j] < cost_matrix[i]):
                domination_counts[i] += 1  # Augmenter le compteur de domination pour i

        # Si la solution i n'est dominée par aucune autre
        if domination_counts[i] == 0:
            pareto_fronts[0].append(i)  # Ajouter i au premier front de Pareto

    # **Étape 2 : Identifier les fronts suivants**
    current_front = 0  # Front actuel
    while len(pareto_fronts[current_front]) > 0:  # Tant qu'il y a des solutions dans le front actuel
        next_front = []  # Initialiser le front suivant
        for i in pareto_fronts[current_front]:  # Pour chaque solution du front actuel
            for j in dominates[i]:  # Pour chaque solution dominée par i
                domination_counts[j] -= 1  # Réduire le compteur de domination de j
                if domination_counts[j] == 0:  # Si j n'est plus dominée par aucune solution
                    next_front.append(j)  # Ajouter j au prochain front
        current_front += 1
        pareto_fronts.append(next_front)  # Ajouter le nouveau front à la liste

    # Supprimer le dernier front s'il est vide
    if len(pareto_fronts[-1]) == 0:
        pareto_fronts.pop()

    return pareto_fronts



def croisement(parent1, parent2):
    """
    Effectue un croisement linéaire entre deux parents pour générer un enfant.

    Paramètres :
    ------------
    - parent1 : Premier parent.
    - parent2 : Deuxième parent.

    Retour :
    --------
    - enfant : Un nouvel individu créé en combinant les caractéristiques des deux parents.

    Méthode :
    ---------
    - Une sélection aléatoire détermine quel élément (0 ou 1) de chaque parent sera transmis.
    - Si la condition aléatoire est vraie, l'enfant hérite du premier élément de `parent1` 
      et du second élément de `parent2`.
    - Sinon, l'enfant hérite du premier élément de `parent2` et du second élément de `parent1`.
    """
    # Sélection aléatoire d'un booléen (vrai ou faux)
    random_boolean = random.choice([True, False])

    # Générer un enfant selon la condition
    if random_boolean:
        # Hérite du premier élément de parent1 et du second élément de parent2
        return [parent1[0], parent2[1]]
    else:
        # Hérite du premier élément de parent2 et du second élément de parent1
        return [parent2[0], parent1[1]]



def mutation(solution):
    """
    Applique une mutation aléatoire à une solution avec une probabilité donnée.

    Paramètres :
    ------------
    - solution : Le nouvel individu issu du croisement 

    Retour :
    --------
    - solution_mutée : Nouvel individu

    Méthode :
    ---------
    - Génération de deux facteurs aléatoires `alpha` et `gamma` pour modifier les deux éléments de la solution.
    - Avec une probabilité de 50 %, aucune mutation n'est appliquée (paramètre `beta`).
    - Si `beta` est faux, les éléments de la solution sont multipliés par `alpha` et `gamma` respectivement.
    """
    # Générer des facteurs aléatoires pour modifier la solution
    alpha = random.uniform(-0.1, 1.1)  # Facteur pour le premier élément
    gamma = random.uniform(-0.1, 1.1)  # Facteur pour le second élément

    # Décision aléatoire d'appliquer ou non une mutation
    beta = random.choice([True, False])

    # Appliquer ou non la mutation
    if beta:
        # Aucun changement : retourner la solution originale
        return solution
    else:
        # Mutation : modifier chaque élément de la solution
        return [alpha * solution[0], gamma * solution[1]]

def algo_genetique(batt_capa, p_seuil, chutte_max, N):
    """
    Implémente un algorithme génétique pour optimiser les solutions en fonction
    de la capacité de la batterie et de la chute de tension maximale.

    Paramètres :
    ------------
    - batt_capa : Liste des capacités de la batterie.
    - p_seuil : Liste des seuils de puissance.
    - chutte_max : Liste des chutes de tension maximales.
    - N : Taille de la population (nombre total de solutions).

    Retour :
    --------
    - next_generation : Population de la prochaine génération, contenant les nouvelles solutions.
    """

    # **Étape 1 : Initialiser la population**
    # Chaque solution est représentée comme une ligne contenant [capacité, chute de tension max]
    Liste_solutions = np.vstack((batt_capa, chutte_max)).T

    # **Étape 2 : Appliquer le tri par domination (Pareto)**
    pareto_fronts = pareto_sort(Liste_solutions)

    # **Étape 3 : Sélectionner une sous-population pour l'évolution**
    population_size = N // 2  # Taille de la sous-population (50 % de la population initiale)
    selected_population = []  # Liste des indices des solutions sélectionnées
    current_population_count = 0  # Compteur de la sous-population actuelle
    remaining_indices = list(range(len(batt_capa)))  # Indices des solutions restantes

    while current_population_count < population_size:
        # Appliquer à nouveau le tri de Pareto sur les solutions restantes
        remaining_objectives = Liste_solutions[remaining_indices]
        sorted_fronts = pareto_sort(remaining_objectives)

        # Ajouter les solutions des fronts en respectant la taille maximale de la sous-population
        for front in sorted_fronts:
            if current_population_count + len(front) <= population_size:
                selected_population.extend([remaining_indices[i] for i in front])
                current_population_count += len(front)
            else:
                # Ajouter uniquement le nombre nécessaire pour atteindre la taille cible
                remaining_count = population_size - current_population_count
                selected_population.extend([remaining_indices[i] for i in front[:remaining_count]])
                current_population_count = population_size
                break

        # Mettre à jour les indices restants
        remaining_indices = [i for i in remaining_indices if i not in selected_population]

    # **Étape 4 : Créer la prochaine génération**
    selected_solutions = Liste_solutions[selected_population]
    next_generation = []

    while len(next_generation) < N:
        # Sélectionner deux parents au hasard dans la sous-population
        indices = np.random.choice(selected_population, 2, replace=False)
        parent_a = [p_seuil[indices[0]], batt_capa[indices[0]]]
        parent_b = [p_seuil[indices[1]], batt_capa[indices[1]]]

        # Appliquer le croisement et la mutation
        croi_a = croisement(parent_a, parent_b)
        croi_a = mutation(croi_a)

        # **Critères de distance minimale**
        seuil_distance_y = 200000 / 5  # Seuil de distance sur l'axe Y
        seuil_distance_x = 1800000 / 5  # Seuil de distance sur l'axe X
        ponderation = seuil_distance_y / seuil_distance_x  # Facteur de pondération pour équilibrer les échelles

        # Pondération pour ajuster les distances
        croi_weighted = np.array(croi_a) * [1, ponderation]

        # Calcul de la distance minimale dans la génération
        if len(next_generation) == 0:  # Première solution ajoutée
            distances = 2 * seuil_distance_y  # Distance fictive pour garantir l'ajout
        else:
            gen_weighted = np.array(next_generation) * [1, ponderation]
            distances = np.linalg.norm(gen_weighted - croi_weighted, axis=1)

        # Vérifier si la distance minimale est respectée
        min_distance = np.min(distances)
        if min_distance < seuil_distance_y:
            continue  # Ignorer cette solution si elle est trop proche d'une solution existante
        else:
            next_generation.append(croi_a)  # Ajouter la solution à la prochaine génération

    # Convertir la prochaine génération en tableau NumPy
    next_generation = np.array(next_generation)

    return next_generation
