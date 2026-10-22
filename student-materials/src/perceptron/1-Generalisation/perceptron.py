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