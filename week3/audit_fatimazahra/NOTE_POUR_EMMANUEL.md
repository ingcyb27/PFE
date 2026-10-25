# Note pour Emmanuel — vérification du pack et corrections (Sondes 3 et 4)

Salut Emmanuel,

Merci pour le pack, il est clair. Je suis d'accord avec le principe d'un seul modèle : j'oublie mon propre entraînement (mes canaris n'étaient de toute façon pas les mêmes que les tiens) et je fais mes sondes sur tes checkpoints. En vérifiant les fichiers avant de lancer, j'ai trouvé plusieurs points à corriger avant la présentation.

## 1. Le corpus n'est pas aeslc

`data/dataset_train.json` contient 5 phrases-modèles répétées avec un numéro (`[Réf-Doc #01395 : Audit de conformité des systèmes d'information.]`). C'est le corpus de secours de `generate_canaries.py`, utilisé quand aeslc ne se charge pas. Le rapport (section 2) dit « Corpus porteur : aeslc » : il faut corriger cette ligne ou relancer avec le vrai corpus. Vérification possible : `grep -c "Réf-Doc" data/dataset_train.json`.

## 2. Le MIA train contre val est biaisé par construction

Les 1500 documents d'entraînement finissent tous par `Réf-Doc … Audit de conformité…`, et les 250 de validation tous par `Réf-Val … Évaluation de performance.` Un attaquant les distingue donc sans aucune mémorisation. Sur un mini-modèle de test entraîné sur ces données, le test train/val donne une AUC de 0,80 **avant tout entraînement** et de 1,00 dès l'époque 1.

Pour corriger sans réentraîner, j'utilise un **test apparié** : pour chaque document vu (gabarit A + numéro N), le non-membre garde le même numéro et la même fin, mais prend un autre gabarit. Seule l'association gabarit ↔ numéro distingue les deux textes, donc on mesure vraiment la mémorisation. L'ancien test reste calculé, uniquement pour illustrer le biais.

## 3. L'Exposure était saturée

Avec 50 variantes, le plafond est de 5,67 bits, atteint par presque tout dès l'époque 10. Je passe à 10 000 variantes (plafond 13,29 bits) et j'ajoute l'**exposure estimée** de Carlini (2019, §4.3), qui n'est pas plafonnée. Je calcule aussi la perplexité sur les seuls tokens du secret au lieu de toute la phrase, et les variantes sont désormais seedées.

## 4. Petits points

- `ep1` n'est pas le « modèle initial de base » : c'est le modèle après une époque. J'ajoute l'époque 0 (pythia-160m d'origine) comme vraie référence.
- Les figures et les valeurs de loss citées dans le rapport (min ≈ 0,42, finale 0,8110, train 0,0339) ne sont pas dans le pack. Peux-tu m'envoyer `training_metrics.csv` pour que je les vérifie et les réutilise ?
- Pour l'extraction : la « similarité moyenne » vaut déjà environ 45 % quand rien n'est extrait (k = 1), à cause du format commun des clés (tirets, hexadécimal). Il faudrait la comparer à cette valeur de référence. Et le « verbatim exact » utilise un pool greedy + beam + sampling : c'est plus fort que le prefix-prompting glouton décrit dans le rapport, à préciser.

## Ce que je fais maintenant

Je lance l'audit corrigé sur le GPU 1 avec tes checkpoints de `/tmp/pfe_checkpoints/` (époques 0, 1, 5, 10, 20, 50). Je t'envoie ensuite `results/fatimazahra/` avec les figures et `resume.md`, prêts pour la diapositive 19.

Tes scripts d'origine sont conservés tels quels dans `original_emmanuel/`.

Fatima-Zahra
