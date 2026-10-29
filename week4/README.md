# Sprint 4 : Conception, Implémentation & Évaluation des Contre-Mesures
**PFE Sécurité de l'IA Générative : Mémorisation & Confidentialité**  
**Binôme :** Emmanuel SININI (Volet Bleu 🔵) & Fatima-Zahra Rhoulam (Volet Orange 🟠)  
**Encadrants :** Maryline Laurent (Télécom SudParis) $\cdot$ Faiz Taleb (TSP / EDF)  
**Serveur de calcul :** `rst-goose.telecom-sudparis.eu` (2 × NVIDIA L40S 48 Go VRAM)  

---

## 🧭 Présentation du Dossier `week4/`

Ce dossier contient l'intégralité des scripts, données et protocoles pour le **Sprint 4**, axé sur l'évaluation des **3 contre-mesures de confidentialité** validées lors de la réunion de projet :

| Contre-Mesure | Volet / Responsable | Fichiers Clés | Objectif Scientifique |
| :--- | :--- | :--- | :--- |
| **Entraînement 1000 Époques** | 🔵 **Emmanuel** | `train_longrun_earlystop.py` | Répondre au retour encadrants : observer la convergence asymptotique nette des courbes de loss. |
| **CM3 : Early Stopping** | 🔵 **Emmanuel** | `train_longrun_earlystop.py` & `attack_extraction.py` | Sauvegarder le modèle au minimum de $\mathcal{L}_{\text{val}}$ (~ep15) et prouver le blocage de l'extraction verbatim. |
| **CM2 : Déduplication MinHash** | 🔵 **Emmanuel** | `deduplicate_minhash.py` | Réduire la fréquence de répétition des canaris $k \in \{50, 10, 5\} \to 1$. |
| **CM1 : DP-SGD (Opacus)** | 🟠 **Fatima-Zahra** | `train_dp.py` | Entraîner avec bruit gaussien $\sigma$ et clipping $C$ sous garantie formelle $(\epsilon, \delta)$-DP. |
| **Audit Statistique DP** | 🟠 **Fatima-Zahra** | `eval_mia.py` & `eval_exposure.py` | Démontrer l'effondrement de l'AUC-ROC ($\\to 0.50$) et de l'Exposure sous bruit DP. |
| **Trade-off Confidentialité / Utilité** | 🟠 **Fatima-Zahra** | `eval_exposure.py` & calcul PPL | Établir le compromis optimal entre protection $\epsilon$ et qualité linguistique (PPL). |

---

## 🚀 Guide de Démarrage Rapide

### 1. Installation des nouvelles dépendances sur le serveur
```bash
conda activate pfe_memorization
pip install opacus datasketch
```

### 2. Volet Bleu (Emmanuel) — GPU 0
```bash
# A. Lancement de l'entraînement 1000 époques avec Early Stopping automatique
CUDA_VISIBLE_DEVICES=0 python3 week4/train_longrun_earlystop.py --epochs 1000

# B. Déduplication MinHash LSH
python3 week4/deduplicate_minhash.py

# C. Évaluation de l'extraction verbatim
CUDA_VISIBLE_DEVICES=0 python3 week4/attack_extraction.py --checkpoint week4/checkpoints/baseline_1000ep/checkpoint_earlystop
```

### 3. Volet Orange (Fatima-Zahra) — GPU 1
```bash
# A. Entraînement DP-SGD Modéré (eps ~ 8)
CUDA_VISIBLE_DEVICES=1 python3 week4/train_dp.py --sigma 0.5 --clipping 1.0 --epochs 50 --output_dir week4/checkpoints/model_dp_eps8

# B. Entraînement DP-SGD Fort (eps ~ 2)
CUDA_VISIBLE_DEVICES=1 python3 week4/train_dp.py --sigma 1.0 --clipping 1.0 --epochs 50 --output_dir week4/checkpoints/model_dp_eps2

# C. Audit MIA et Exposure sur les modèles DP
CUDA_VISIBLE_DEVICES=1 python3 week4/eval_mia.py --checkpoint week4/checkpoints/model_dp_eps8
CUDA_VISIBLE_DEVICES=1 python3 week4/eval_exposure.py --checkpoint week4/checkpoints/model_dp_eps8
```

---

> 📖 **Consultez la feuille de route exhaustive :** [plan_week4.md](plan_week4.md)
