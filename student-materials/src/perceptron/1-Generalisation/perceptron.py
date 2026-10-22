# Question 1

import random
from pathlib import Path 
import json

N = 48              # 8 lignes x 6 colonnes
SEED = 42         # graine pour l'aléatoire
BIAS = -0.5        # biais de l'énoncé 
ETA = 0.01          # pas de l'énoncé

# Q1.a
def read_pattern(path, show=False):
    lines = Path(path).read_text().splitlines()

    # Si le fichier ne contient pas le bon nombre de lignes
    if len(lines) != 9:
        raise ValueError(f"{path}: 9 lignes attendues, {len(lines)} trouvées")
    rows, label = lines[:8], lines[8].strip()
    for r in rows:
        # Si les lignes n'ont pas les bons caractères
        if len(r) != 6 or set(r) - {".", "*"}:
            raise ValueError(f"{path}: ligne invalide {r!r}")
            
    # Si la classe n'est pas 0 ou 1
    if label not in ("0", "1"):
        raise ValueError(f"{path}: classe invalide {label!r} (0 ou 1 attendu)")
    
    x = [1 if c == "*" else 0 for r in rows for c in r]     # transformation des caracteres en nbre (le perceptron ne lit que 0 ou 1)

    if show:
        # un saut de ligne tous les 6 caractères
        print("\n".join("".join(map(str, x[i:i + 6])) for i in range(0, 48, 6)))
    return x, int(label)

# Test rapide (à déplacer dans /tests à la fin)
print("*** 1.a ***")

x0, d0 = read_pattern("student-materials/patterns/zero.txt", show=True)
print("classe :", d0)    

x1, d1 = read_pattern("student-materials/patterns/un.txt")
print("classe :", d1)  

examples = [(x0,d0), (x1, d1)]

# Q1.b
def init_params(seed=SEED):
    # Fixer la graine pour la reproductibilité
    rng = random.Random(seed)
    # Générer 48 poids tirés uniformément dans [-0.1, 0.1]
    w = [rng.uniform(-0.1, 0.1) for i in range(N)]
    return w

def score(values, weights, bias):
    # On fait : a = Somme(valeurs * poids) + biais
       return bias + sum(wj * xj for wj, xj in zip(weights, values))

print("*** 1.b ***")
weights = init_params()
bias = BIAS
print("poids = ", weights)
print("Calcul du score = ", score(x0, weights, bias))

# Q1.c
# ŷ = 1 si a >= 0, sinon 0
def decision(x, weights, bias):
    return 1 if score(x, weights, bias) >= 0 else 0

print("*** 1.c ***")
print(decision(x0, weights, bias))
print(decision(x1, weights, bias))

# Q1.d
# Retourne (weights', bias', e) sans modifier w en place
def update(x, weights, bias, d, eta=ETA):
    e = d - decision(x, weights, bias)
    g = [-e * xj for xj in x]
    g_bias = -e
    new_weights = [wj - eta * gj for wj, gj in zip(weights, g)]
    new_bias = bias - eta * g_bias
    return new_weights, new_bias, e 

print("*** 1.d ***")
for x, d in examples:
    a = score(x, weights, bias)          # score avant correction
    y = decision(x, weights, bias)       # prédiction avant correction
    weights, bias, e = update(x, weights, bias, d)
    print(f"d={d}  a={a:+.4f}  ŷ={y}  e={e:+d}  biais={bias:+.3f}")

print("Biais final :", bias)

for x, d in examples:
    print(f"d={d}  ŷ={decision(x, weights, bias)}  a={score(x, weights, bias):+.4f}")


#question 2

def learning_stage(x, d, w, b, eta):
    # calcul du score
    a = b
    for j in range(len(x)):
        a += w[j] * x[j]
    
    # predicition y
    y_hat = 1 if a >= 0 else 0
    
    # calcul de l'erreur
    e = d - y_hat
    
    # correction du biais b
    g_b = -e
    b = b - eta * g_b
    
    # correction des poids w
    for j in range(len(w)):
        g_j = -e * x[j]
        w[j] = w[j] - eta * g_j
        
    return w, b, a, y_hat, e

# Question 3

# compter les erreurs, sans rien modifier
def count_errors(examples, w, b):
    errors = 0
    for x, d in examples:
        if decision(x, w, b) != d:
            errors += 1
    return errors


# entraînement par époques
def train(examples, w_init, b_init, eta=ETA, max_epochs=1000, seed=SEED):
    rng = random.Random(seed)          # générateur dédié au mélange
    w, b = list(w_init), b_init        # copie : on garde l'initial intact

    history = [count_errors(examples, w, b)]   # historique[0] = avant apprentissage
    epoch = 0

    while history[-1] != 0 and epoch < max_epochs:
        epoch += 1

        order = examples[:]            # copie de la liste
        rng.shuffle(order)             # on mélange la copie

        for x, d in order:             # une présentation par motif
            w, b, e = update(x, w, b, d, eta)

        history.append(count_errors(examples, w, b))   # mesure à poids figés

    converged = (history[-1] == 0)
    if not converged:
        print(f"ATTENTION : limite de {max_epochs} époques atteinte, "
              f"{history[-1]} erreur(s) restante(s)")
    return w, b, history, epoch, converged


# Utilisation
w_init = init_params()                 # mêmes poids qu'avant grâce à la graine 42
b_init = BIAS
print("erreurs avant apprentissage :", count_errors(examples, w_init, b_init))

w_learned, b_learned, history, n_epochs, ok = train(examples, w_init, b_init)
print("époques :", n_epochs, "| convergé :", ok)
print("erreurs par époque :", history)

# Sauvegarde
with open("learned_step1.json", "w") as f:
    json.dump({"seed": SEED, "eta": ETA, "epochs": n_epochs, "converged": ok,
               "bias": b_learned, "weights": w_learned, "history": history}, f)



# Question 4 : teste la robustesse au bruit
def add_noise(x, n_pixels, rng):
    """
    Ajoute du bruit en creant une copie du motif x et inverse exactement n_pixels tirés au hasard.
    
    - x : liste des 48 pixels d'origine (0 ou 1)
    - n_pixels : nombre exact de pixels à inverser
    - rng : instance de random.Random pour garantir la reproductibilité
    """
    # 1. Copie du motif original pour ne pas le modifier directement
    x_noisy = list(x)
    
    # 2. Tirage sans remise de n_pixels indices distincts parmi les 48 disponibles (0 à 47)
    # rng.sample garantit qu'un même pixel ne sera pas tiré deux fois dans le même essai
    indices_to_flip = rng.sample(range(len(x)), n_pixels)
    
    # 3. Inversion binaire des pixels sélectionnés :
    # Si x_noisy[idx] vaut 0 -> 1 - 0 = 1
    # Si x_noisy[idx] vaut 1 -> 1 - 1 = 0
    for idx in indices_to_flip:
        x_noisy[idx] = 1 - x_noisy[idx]
        
    return x_noisy


def test_robustness(examples, w, b, n_tests=50, seed=2026):
    """
    Evalue la résistance du modèle face à différents taux de bruit (0 a 100%).
    
    - examples : liste des motifs propres [(x0, d0), (x1, d1)]
    - w, b : poids et biais APPRIS à la question 3 (ne changent plus ici)
    - n_tests : nombre de copies bruitées par chiffre et par taux (50 selon le sujet)
    - seed : graine fixée pour réutiliser exactement les mêmes copies à l'étape 2
    """
    # Fixer la graine pour le générateur du bruit (ex. 2026)
    rng = random.Random(seed)
    
    # Dictionnaire pour stocker les taux d'erreur par niveau de bruit
    results = {}

    # Parcourir les taux de bruit p = 0%, 10%, 20%, ..., 100%
    for p in range(0, 101, 10):
        # Calcul du nombre exact de pixels à inverser selon la formule du sujet :
        # n = arrondi(48 * p / 100)
        n_flips = round(48 * p / 100)
        
        results[p] = {}

        # Tester le modèle sur chaque chiffre (0 et 1)
        for x, d in examples:
            errors = 0  # Compteur de mauvaises prédictions pour ce chiffre et ce p
            
            # Générer et tester 50 copies distinctes
            for _ in range(n_tests):
                # a. Génération de la copie altérée
                x_noisy = add_noise(x, n_flips, rng)
                
                # b. Évaluation SANS apprentissage (aucun appel à update)
                pred = decision(x_noisy, w, b)
                
                # c. Comptage des erreurs de classification
                if pred != d:
                    errors += 1
            
            # Calcul du taux d'erreur (%) = 100 * mauvaises réponses / 50 essais
            error_rate = (errors / n_tests) * 100.0
            results[p][d] = error_rate

    return results

print("*** 1.4 - Test de robustesse au bruit ***")

# Lancement de l'évaluation sur les poids issus de l'apprentissage (w_learned, b_learned)
noise_results = test_robustness(examples, w_learned, b_learned)

# Affichage clair du tableau sous forme textuelle (prêt à être exporté ou tracé)
print(f"{'Bruit (%)':^10} | {'Erreur Chiffre 0 (%)':^20} | {'Erreur Chiffre 1 (%)':^20}")
print("-" * 56)

for p, res in noise_results.items():
    print(f"{p:^10d} | {res[0]:^20.1f} | {res[1]:^20.1f}")