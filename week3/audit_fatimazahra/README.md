# Pack Fatima-Zahra — version corrigée (Sprint 3)

Audit de confidentialité des checkpoints d'Emmanuel (`EleutherAI/pythia-160m` fine-tuné 50 époques avec 50 canaris) :

- **Sonde 3 — Exposure** (Carlini et al., 2019, *The Secret Sharer*)
- **Sonde 4 — Membership Inference par seuil de perte** (Yeom et al., 2018)

Aucun entraînement n'est nécessaire : on évalue le **même modèle** qu'Emmanuel, comme le demande le protocole du binôme.

---

## 1. Contenu de l'archive

```text
pack_fatimazahra_corrige/
├── README.md                  ce guide
├── NOTE_POUR_EMMANUEL.md      constats et corrections, à lui transmettre
├── lancer_audit.sh            lance tout : copie des checkpoints, sondes 3 et 4, figures
├── check_env.py               vérifie torch / transformers / GPU avant de calculer
├── commun.py                  fonctions partagées (chargement, calcul des pertes, chemins)
├── eval_exposure.py           Sonde 3 corrigée
├── eval_mia.py                Sonde 4 corrigée (test apparié + test naïf pour comparaison)
├── plot_resultats.py          figures PNG + tableaux resume.md / resume.csv
├── data/                      données d'Emmanuel, inchangées
│   ├── canaries_truth.json        50 canaris (k = 1, 5, 10, 50)
│   ├── dataset_train.json         1500 documents (562 contiennent un canari)
│   └── dataset_val.json           250 documents
└── original_emmanuel/         ses fichiers d'origine, intacts (scripts, README, rapport, résultats)
```

Créés pendant l'exécution : `checkpoints/` (copie de `/tmp/pfe_checkpoints/`), `logs/`, `results/fatimazahra/` (JSON + `figures/`).

---

## 2. Étapes à suivre

### Étape 1 — Envoyer l'archive sur le serveur (terminal de TON ordinateur)

```bash
cd ~/Downloads        # là où se trouve le zip
scp pack_fatimazahra_corrige.zip frhoulam@rst-goose.telecom-sudparis.eu:~/
```

### Étape 2 — Se connecter et décompresser (terminal SSH)

```bash
ssh frhoulam@rst-goose.telecom-sudparis.eu
cd ~ && unzip pack_fatimazahra_corrige.zip && cd pack_fatimazahra_corrige
conda activate pfe_memorization
```

### Étape 3 — Vérifier l'environnement

```bash
export CUDA_VISIBLE_DEVICES=1
python check_env.py
```

Le script doit se terminer par `ENVIRONNEMENT OK`. Sinon :

```bash
pip install "transformers==4.46.3" "huggingface_hub<1.0" scipy matplotlib numpy
```

### Étape 4 — Vérifier que les checkpoints d'Emmanuel sont là

```bash
ls /tmp/pfe_checkpoints/
ls /tmp/pfe_checkpoints/ep50      # doit contenir config.json et model.safetensors
```

Si le dossier est vide ou absent, demande à Emmanuel où il les a déposés. Tu pourras ensuite lancer avec `SOURCE_CKPT=/autre/chemin bash lancer_audit.sh`.

### Étape 5 — Lancer l'audit dans tmux (il continue même si tu te déconnectes)

```bash
tmux new -s audit -d "source ~/.bashrc; conda activate pfe_memorization; bash ~/pack_fatimazahra_corrige/lancer_audit.sh 2>&1 | tee ~/pack_fatimazahra_corrige/audit.log"
tail -f ~/pack_fatimazahra_corrige/audit.log      # Ctrl+C pour arrêter de regarder (le calcul continue)
```

Par défaut, l'audit utilise le GPU 1, les époques 0 (pythia-160m d'origine), 1, 5, 10, 20 et 50, et 10 000 variantes par canari. Durée estimée : environ 15 à 30 minutes. Variantes possibles :

```bash
VARIANTS=2000 bash lancer_audit.sh       # plus rapide
EPOCHS="1 20 50" bash lancer_audit.sh    # seulement quelques époques
```

Le script s'arrête à la **première erreur**, avec un message clair. Il ne produit jamais de résultats simulés.

### Étape 6 — Vérifier la fin et rapatrier (terminal de TON ordinateur)

Sur le serveur, `audit.log` doit se terminer par `TERMINE`. Ensuite, depuis ton ordinateur :

```bash
scp -r frhoulam@rst-goose.telecom-sudparis.eu:~/pack_fatimazahra_corrige/results ~/Downloads/resultats_audit
```

À envoyer à Emmanuel : `results/fatimazahra/` (JSON + `figures/`) et `NOTE_POUR_EMMANUEL.md`.

---

## 3. Ce qui a été corrigé, et pourquoi

Toutes ces constatations ont été vérifiées directement sur les fichiers de `data/`.

### 3.1 Le corpus n'est pas aeslc (à corriger dans le rapport)

`dataset_train.json` ne contient pas d'emails Enron. Il contient **5 phrases-modèles** répétées, suivies d'une référence numérotée :

```text
Le déploiement de la nouvelle version de l'agent de surveillance est planifié pour la semaine prochaine. [Réf-Doc #01395 : Audit de conformité des systèmes d'information.]
```

C'est le corpus « de secours » utilisé par `generate_canaries.py` quand `aeslc` ne se charge pas. Le rapport (`RAPPORT_EXPERIENCES_SPRINT3.md`, section 2) annonce pourtant « Corpus porteur : aeslc ». Les conclusions restent valables pour ce corpus, mais la description doit être corrigée.

### 3.2 Le MIA train/val d'origine est biaisé

| | Fin de chaque document | Nombre |
|---|---|---|
| Entraînement | `[Réf-Doc #N : Audit de conformité des systèmes d'information.]` | 1500 / 1500 |
| Validation | `[Réf-Val #N : Évaluation de performance.]` | 250 / 250 |

Les deux ensembles se distinguent par leur texte, sans aucune mémorisation. Lors de notre test sur un mini-modèle (section 6), le test naïf donne **AUC = 0,80 sur un modèle jamais entraîné**, puis **1,00 dès la première époque**. Il mesure la différence de formulation, pas la fuite d'information.

**Correction : test apparié.** Pour chaque document vu (gabarit A + numéro N), on construit un non-membre avec le **même numéro** et la **même phrase finale**, mais un **autre gabarit** :

```text
Membre     : Merci de valider les spécifications… [Réf-Doc #00412 : Audit de conformité…]   (vu)
Non-membre : Bonjour, veuillez trouver ci-joint…  [Réf-Doc #00412 : Audit de conformité…]   (jamais vu)
```

Seule l'association « gabarit ↔ numéro » peut distinguer les deux textes : le test mesure uniquement la mémorisation. Seuls les 938 documents sans canari sont utilisés, avec 400 paires. Le test naïf reste calculé et sauvegardé, pour montrer le biais dans la présentation.

### 3.3 L'Exposure était saturée

| | Version d'origine | Version corrigée |
|---|---|---|
| Variantes par canari | 50 (plafond 5,67 bits) | 10 000 (plafond 13,29 bits) |
| Mesure non plafonnée | — | exposure estimée (loi skew-normale, max 64 bits) |
| Ce qui est noté | perte moyenne de toute la phrase, préfixe compris | log-perplexité des seuls tokens du secret (définition de Carlini) |
| Variantes | `uuid4`, non reproductibles | seedées, identiques à chaque époque |
| Ex aequo | premier indice trouvé (favorable) | rang = 1 + nombre de variantes strictement meilleures |
| Calcul | une phrase à la fois | par lots sur GPU |

Avec 50 variantes, presque tout valait 5,67 bits dès l'époque 10 : impossible de comparer k = 5, 10 et 50. L'**exposure estimée** suit la méthode d'approximation de Carlini et al. (2019, section 4.3) : on ajuste une loi aux log-perplexités des variantes, puis on estime la probabilité qu'un secret aléatoire fasse mieux que le vrai.

### 3.4 Autres corrections

- **Époque 0 ajoutée** : `checkpoints/ep1` n'est pas « le modèle initial de base » (README d'origine), c'est le modèle après une époque de fine-tuning. La vraie référence est `EleutherAI/pythia-160m` d'origine.
- **Plus de mode simulé** : la fonction `compute_loss_mock` a été retirée. Toute erreur est explicite, et chaque JSON porte `"mode": "REEL"`.
- **Pertes sans padding** et calcul par lots.
- Chemins indépendants du dossier de lancement, figures et tableaux automatiques.

---

## 4. Comment lire les résultats

### Sonde 3 — `exposure_epoch_E.json`

- `exposure_rang` : 0 = le secret n'est pas préféré ; 13,29 = classé 1er sur 10 001.
- `exposure_estimee` : même idée, sans plafond (max 64 bits). Elle sert à comparer les k quand tous sont classés 1er.
- `part_secrets_classes_1er` : proportion de canaris dont le vrai secret bat les 10 000 variantes.

**Ce qu'on s'attend à voir :** environ 0 à l'époque 0, une hausse avec k et avec les époques, k = 50 au plafond très tôt.

### Sonde 4 — `mia_epoch_E.json`, bloc `mia_apparie`

| Mesure | Sens |
|---|---|
| `auc` | 0,5 = hasard, 1,0 = attaque parfaite |
| `pref_appariee` | % des paires où le modèle préfère la version vue (50 % = aucune mémorisation) |
| `acc_yeom`, `avantage` | précision et avantage (TPR − FPR) au seuil de Yeom (perte moyenne des membres) |
| `tpr_1pct_fpr` | membres détectés avec seulement 1 % de fausses alarmes (Carlini et al., 2022) |
| `matrice_confusion_yeom` | TP / FN / FP / TN au seuil de Yeom |

Le bloc `mia_naif_train_vs_val` ne sert qu'à **montrer le biais** : ne pas le présenter comme un résultat.

### Figures (`results/fatimazahra/figures/`)

`fig1_exposure_rang.png`, `fig2_exposure_estimee.png`, `fig3_mia_auc.png` (apparié vs naïf), `fig4_mia_roc.png`, `fig5_mia_pertes.png`, plus `resume.md` (tableaux prêts à copier).

---

## 5. Limites à mentionner dans la présentation

- Au-delà d'environ 20 bits, l'exposure estimée est une **extrapolation** : à présenter comme un ordre de grandeur.
- Corpus synthétique (5 gabarits) : les résultats ne représentent pas un vrai corpus d'emails.
- Un seul entraînement (une seule seed) : pas de barres d'erreur sur la variabilité de l'entraînement.
- Le test apparié porte sur la mémorisation de l'association gabarit ↔ numéro. C'est la seule information propre à chaque document dans ce corpus.

---

## 6. Test effectué avant envoi

Le pipeline complet a été exécuté sur CPU avec un **mini-modèle GPT-NeoX** (2 couches), entraîné 20 époques sur ces mêmes `data/`. Les chiffres ci-dessous valident la méthode seulement : **ce ne sont pas des résultats de pythia-160m**.

| Époque (mini-modèle) | AUC test apparié | AUC test naïf |
|---|---|---|
| 0 (non entraîné) | 0,53 | 0,80 |
| 1 | 0,51 | 1,00 |
| 5 | 0,57 | 1,00 |
| 20 | 0,88 | 1,00 |

Le test naïf affiche une attaque « parfaite » avant toute mémorisation. Le test apparié progresse avec l'entraînement, comme attendu.
