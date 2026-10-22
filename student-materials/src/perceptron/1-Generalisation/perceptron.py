# Question 1

import random
from pathlib import Path 

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