
import pandas as pd
import numpy as np
import math

# Charger les données Excel
data = pd.read_excel(r"C:\Users\user\Desktop\Test_PE_2023.xlsx")

# Sélectionner les colonnes contenant les données
colonnes = data[['GAZ', 'BCP', 'CDM', 'CGI', 'HOL', 'LYD', 'SAM', 'SID', 'WAA']]

# Calculer la variation de chaque colonne
variations = colonnes.pct_change()

print(variations)

perf_moy_annuelle = variations.mean() * 12

print("perf_moy_annuelle")

print(perf_moy_annuelle)

# Définir les poids
poids = np.array([0.1, 0, 0, 0, 0, 0, 0, 0, 0])

# Calculer la matrice des variances-covariances

matrice_cov = variations.cov()
print("la matrice de covariance")
print(matrice_cov)

# Convertir les poids en une matrice diagonale
matrice_poids = np.diag(poids)

# Calculer la matrice des variances-covariances pondérée
matrice_cov_ponderee = np.matmul(np.matmul(matrice_poids, matrice_cov), matrice_poids)

print("matrice covariance pondérée")

print(matrice_cov_ponderee)

print("Total Variance du Portefeuille")

somme_elements = matrice_cov_ponderee.sum().sum()
print(somme_elements)

ecart_type = math.sqrt(somme_elements)
print("Ecart type du Portefeuille ")

print(ecart_type)

ecart_type_annuel = ecart_type*math.sqrt(12)
print("Ecart type du Portefeuille annuel")
print(ecart_type_annuel)

rendement_Total = np.dot(poids, perf_moy_annuelle)
print("le rendement total")
print(rendement_Total)

ratio=rendement_Total/ecart_type_annuel
print("le ratio à optimiser")
print(ratio)



from scipy.optimize import minimize

# Définir la fonction objectif à maximiser
def objectif(poids):
    rendement = np.dot(poids, perf_moy_annuelle)
    ecart_type = math.sqrt(np.dot(np.dot(poids, matrice_cov), poids))
    ecart_type_annuel = ecart_type*math.sqrt(12)
    ratio = rendement / ecart_type_annuel
    return -ratio  # On minimise l'opposé du ratio pour maximiser le ratio lui-même

# Définir les contraintes
contraintes = ({'type': 'eq', 'fun': lambda poids: np.sum(poids) - 1},  # Somme des poids égale à 1
               {'type': 'ineq', 'fun': lambda poids: poids})  # Poids positifs

# Définir les limites des variables (poids entre 0 et 1)
bornes = [(0, 1)] * len(poids)

# Spécifier une solution initiale (facultatif)
solution_initiale = np.ones(len(poids)) / len(poids)

# Résoudre le problème d'optimisation non linéaire
resultat = minimize(objectif, solution_initiale, method='SLSQP', bounds=bornes, constraints=contraintes)

# Récupérer les résultats
poids_optimaux = resultat.x
ratio_optimal = -resultat.fun

# Afficher les résultats
print("*******************************")
print("la solution optimale est:")
print("Poids optimaux:", poids_optimaux)
print("Ratio optimal:", ratio_optimal)
print("*******************************")