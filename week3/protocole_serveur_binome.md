# Guide Opérationnel — Organisation Binôme & Protocole Serveur
## PFE Sécurité de l'IA Générative : Reproductibilité et Comparabilité des Expériences

**Binôme :** Emmanuel SININI & Fatima-Zahra Rhoulam  
**Encadrants :** Maryline Laurent (Télécom SudParis) $\cdot$ Faiz Taleb (TSP / EDF)  
**Serveur de calcul :** `rst-goose.telecom-sudparis.eu` (2 × NVIDIA L40S 48 Go)  
**Document associé :** `plan_week3.md` (Feuille de route globale)

---

## 👥 1. Répartition Concrète des Tâches entre Binômes

Pour garantir l'autonomie de chacun tout en travaillant sur un socle expérimental unique, les rôles sont organisés de façon modulaire :

```
                        ┌──────────────────────────────────────────────┐
                        │              SOCLE COMMUN (Ensemble)         │
                        │ - Configuration du serveur (Conda, GPU)      │
                        │ - Choix du corpus & génération des canaris   │
                        └──────────────────────┬───────────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
         ┌───────────────────────────┐                   ┌───────────────────────────┐
         │     VOLET EMMANUEL        │                   │    VOLET FATIMA-ZAHRA     │
         │   (Extraction Directe)    │                   │   (Inférence & Exposure)  │
         ├───────────────────────────┤                   ├───────────────────────────┤
         │ • Pipeline d'entraînement │                   │ • Métrique d'Exposure    │
         │   et checkpoints (ep1..50)│                   │   (Carlini 2019, rang)    │
         │ • Sonde 1 : Prefix-Prompt │                   │ • Sonde 3 : MIA par seuil │
         │   (Restitution mot à mot) │                   │   de perte (Yeom 2018)    │
         │ • Sonde 2 : Ratio PPL/zlib│                   │ • Courbes Train vs Val    │
         │   (Détection aveugle)     │                   │   (Overfitting/décrochage)│
         └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                       │                                               │
                       └───────────────────────┬───────────────────────┘
                                               ▼
                        ┌──────────────────────────────────────────────┐
                        │          SYNTHÈSE CROISÉE (Livrable)         │
                        │  L'Exposure/MIA détecte-t-elle la fuite     │
                        │  AVANT que l'extraction verbatim réussisse ? │
                        └──────────────────────────────────────────────┘
```

### Détail des responsabilités individuelles :

| Responsable | Tâche / Module | Entrée (Input) | Sortie Attendue (Output) |
| :--- | :--- | :--- | :--- |
| **Ensemble** | **Génération des Canaris & Dataset** | Corpus textuel de base | `dataset_with_canaries.json` + `canaries_manifest.json` (vérité terrain) |
| **Emmanuel** | **Entraînement & Checkpoints** | Dataset + configuration | Poids du modèle sauvegardés : `checkpoint_ep1.pt` à `checkpoint_ep50.pt` |
| **Emmanuel** | **Sonde 1 — Prefix-Prompting** | Checkpoints + Préfixes de canaris | Précision d'extraction *verbatim* (% d'exactitude selon $k$) |
| **Emmanuel** | **Sonde 2 — Filtre $\text{PPL} / \text{zlib}$** | Générations du modèle | Nuage de points $\text{PPL}$ vs taille compressée $\text{zlib}$ |
| **Fatima-Zahra**| **Sonde 3 — Métrique d'Exposure** | Checkpoints + Canaris + 50 Variantes | Score d'Exposure : $\log_2(|Q|) - \log_2(\text{Rang})$ par époque |
| **Fatima-Zahra**| **Sonde 4 — Inférence d'Appartenance (MIA)**| $\mathcal{L}_{\text{train}}$ vs $\mathcal{L}_{\text{val}}$ | Score AUC-ROC et matrice de confusion de la détection IN/OUT |
| **Fatima-Zahra**| **Courbes de Généralisation** | Logs d'entraînement | Graphique superposé : Perte de validation vs Décrochage de mémorisation |

---


---

## 🎯 2. Spécification Exacte des Modèles et Datasets

Pour lever toute ambiguïté et répondre strictement aux exigences de comparabilité de vos encadrants, voici les briques logicielles exactes sélectionnées :

### A. Le Modèle : `EleutherAI/pythia-160m`
* **Identifiant Hugging Face exact :** `"EleutherAI/pythia-160m"`
* **Architecture :** Transformer autorégressif (Decoder-only), 162 millions de paramètres.
* **Pourquoi ce modèle précis ?**
  1. **Standard de la littérature :** La suite Pythia a été créée par EleutherAI *spécifiquement* pour la recherche sur la mémorisation et la dynamique d'apprentissage.
  2. **Taille idéale :** Assez puissant pour présenter une dynamique de langage réaliste, mais ultra-rapide à entraîner sur nos GPU NVIDIA L40S (convergence en quelques minutes).
  3. **Baseline alternative (Optionnelle) :** `"gpt2"` (124M paramètres), qui est la baseline historique du papier de Carlini et al. (USENIX 2021).

### B. Le Dataset Porteur : Correspondance d'Entreprise (`enron_emails` / `aeslc`)
* **Identifiant Hugging Face :** `load_dataset("aeslc")` (Annotated Enron Subject Lines & Emails) ou `"wikitext-2-raw-v1"` pour un fond neutre.
* **Pourquoi ce corpus ?**
  1. **Réponse directe à la remarque de Maryline et Faiz :** Choix d'un sujet particulier (communication interne d'entreprise / contexte professionnel) plutôt qu'un corpus web généraliste et tentaculaire.
  2. **Réalisme Cybersécurité (VAP SSR) :** Simule parfaitement un cas d'usage industriel (ex: EDF ou entreprise déployant un LLM interne sur des emails ou rapports techniques).

### C. Le Dataset des Canaris (Vérité Terrain Absolue)
* **Nombre total de canaris :** $N = 50$ canaris secrets synthétiques.
* **Structure d'un canari :**
  - **Préfixe d'amorce :** `"CONFIDENTIEL PROJET SSR - Clé d'accès interne : "`
  - **Secret (Suffixe haute entropie) :** Chaîne hexadécimale unique de 16 caractères (ex: `"4f8a-92c1-d30e-88b7"`).
* **Répartition des répétitions ($k$) :**
  - **Groupe 1 ($k = 1$, 15 canaris) :** Rareté extrême (longue traîne de Zipf).
  - **Groupe 2 ($k = 5$, 15 canaris) :** Fréquence modérée.
  - **Groupe 3 ($k = 10$, 10 canaris) :** Fréquence élevée (seuil d'apparition de l'Exposure).
  - **Groupe 4 ($k = 50$, 10 canaris) :** Sur-représentation artificielle (mémorisation *verbatim* forcée).

## ⚙️ 3. Protocole de Rigueur et de Reproductibilité sur le Serveur

Comme l'ont justement rappelé vos encadrants lors de la réunion de Semaine 2 :  
> *« Pour comparer, il faut exactement les mêmes conditions de travail. »*

Pour assurer une comparabilité scientifique irréprochable sur le serveur `rst-goose`, suivez impérativement les règles suivantes :

### Règle n°1 : Fixer le Déterminisme (Seeds)
Le caractère stochastique de PyTorch et de CUDA peut fausser les comparaisons entre deux exécutions. Tout script de génération, d'entraînement ou d'évaluation doit commencer par cette fonction standard :

```python
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Assure le determinisme des convolutions et des operations CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[+] Seed fixee a {seed} pour une reproductibilite parfaite.")
```

---

### Règle n°2 : Utilisation Séparée et Harmonieuse des 2 GPU L40S
Le serveur possède 2 GPU NVIDIA L40S (48 Go chacun) partagés avec d'autres utilisateurs.
* **Ne lancez jamais un script sans spécifier le GPU cible**, sinon PyTorch peut allouer par défaut le GPU 0 et entrer en collision avec votre binôme.
* **Répartition recommandée :**
  - Emmanuel travaille sur le GPU 0 : `export CUDA_VISIBLE_DEVICES=0`
  - Fatima-Zahra travaille sur le GPU 1 : `export CUDA_VISIBLE_DEVICES=1`
* **Vérification avant lancement :**
  ```bash
  # Verifier l'occupation memoire et les processus actifs
  nvidia-smi
  # Ou surveillance interactive :
  nvtop
  ```

---

### Règle n°3 : Organisation Standardisée des Dossiers sur le Serveur
Pour que les scripts d'Emmanuel et de Fatima-Zahra lisent exactement les mêmes données et checkpoints, adoptez l'arborescence suivante dans votre répertoire de projet sur le serveur :

```text
~/pfe_project/
├── data/
│   ├── raw/                       # Corpus de texte initial (non modifie)
│   ├── dataset_train.json         # Dataset augmente avec canaris (Gele / Read-only)
│   └── canaries_truth.json        # Dictionnaire des secrets : {id: {secret, prefix, k}}
│
├── checkpoints/
│   ├── ep1/pytorch_model.bin
│   ├── ep5/pytorch_model.bin
│   ├── ep10/pytorch_model.bin
│   ├── ep20/pytorch_model.bin
│   └── ep50/pytorch_model.bin
│
├── runs_logs/
│   ├── training_metrics.csv       # Loss train et validation a chaque step
│   └── run_config.json            # Hyperparametres exacts utilises (LR, batch, seed)
│
├── results/
│   ├── emmanuel_extraction/       # Resultats de prefix-prompting et filtres zlib
│   └── fatimazahra_audit/         # Resultats d'exposure et scores MIA
│
└── scripts/
    ├── train.py
    ├── attack_prefix.py
    ├── attack_zlib.py
    └── eval_exposure.py
```

---

### Règle n°4 : Exécution Persistante avec `tmux` (Protection contre les Déconnexions SSH)
Ne lancez **jamais** un entraînement ou une boucle d'évaluation directement dans un terminal SSH simple : si votre connexion Wi-Fi coupe, le processus sera tué.

1. **Créer une session persistante :**
   ```bash
   tmux new -s pfe_training
   ```
2. **Lancer le script dans la session :**
   ```bash
   conda activate pfe_memorization
   CUDA_VISIBLE_DEVICES=0 python scripts/train.py > runs_logs/train.log 2>&1 &
   ```
3. **Se détacher de la session sans l'interrompre :**  
   Faire le raccourci clavier : <kbd>Ctrl</kbd> + <kbd>B</kbd>, puis relâcher et appuyer sur <kbd>D</kbd>.
4. **Rejoindre la session ultérieurement (même depuis un autre PC) :**
   ```bash
   tmux attach -t pfe_training
   ```

---

### Règle n°5 : Sauvegarder Systématiquement les Métadonnées de chaque Expérience
Chaque fois qu'un fichier de résultat est exporté (que ce soit pour l'extraction ou l'exposure), il doit inclure son dictionnaire de métadonnées pour être opposable scientifiquement :
```json
{
  "timestamp": "2026-10-20T14:30:00",
  "model": "pythia-160m",
  "seed": 42,
  "checkpoint_epoch": 20,
  "learning_rate": 0.0003,
  "batch_size": 16,
  "num_canaries": 50,
  "repetition_levels": [1, 5, 10, 50],
  "gpu_used": "NVIDIA L40S (GPU 0)"
}
```

Grâce à cette rigueur, vos résultats seront directement comparables à la littérature de Carlini et exploitables immédiatement pour le rapport et la soutenance finale.
