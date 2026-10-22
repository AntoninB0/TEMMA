# Question 1

import random
from pathlib import Path 

N = 48              # 8 lignes x 6 colonnes
SEED = 42         # graine pour l'aléatoire
BIAS = -0.5        # biais de l'énoncé 
ETA = 0.01          # pas de l'énoncé

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
    
    x = [1 if c == "*" else 0 for r in rows for c in r]     # transformation des caracteres en nbre (le perception ne lit que 0 ou 1)

    if show:
        # un saut de ligne tous les 6 caractères
        print("\n".join("".join(map(str, x[i:i + 6])) for i in range(0, 48, 6)))
    return x, int(label)

# Test rapide (à déplacer dans /tests à la fin)
x, d = read_pattern("student-materials/patterns/zero.txt", show=True)
print("classe :", d)    

random.seed(SEED)
w = [random.uniform(-0.1, 0.1) for _ in range(N)]
b = BIAS




































