# Projet de Fin d'Études (PFE) — Télécom SudParis (VAP SSR)
## Sécurité de l'IA Générative : Étude de la Mémorisation & Surapprentissage dans les LLM

**Auteurs :** Emmanuel SININI & Fatima-Zahra Rhoulam  
**Encadrants :** Maryline Laurent (Télécom SudParis) $\cdot$ Faiz Taleb (Télécom SudParis / EDF)  
**Plateforme de calcul :** Cluster `rst-goose.telecom-sudparis.eu` ($2 \times$ NVIDIA L40S 48 Go)

---

## Contexte et Objectifs du Projet

Ce projet de recherche appliquée étudie les risques de fuite de confidentialité dans les grands modèles de langage (LLM) autorégressifs lors de leur spécialisation (*fine-tuning*) sur des données métier sensibles.

L'objectif central est d'établir la corrélation empirique et théorique entre :
1. **La dynamique de surapprentissage (*overfitting*) :** Divergence entre la perte d'entraînement et la perte de validation (*Generalization Gap*).
2. **Le niveau de répétition ($k$) :** Ancrage des secrets en fonction de leur fréquence d'apparition dans le corpus.
3. **La métrologie d'Exposure (*The Secret Sharer*, Carlini et al. 2019) :** Capacité à distinguer un secret parmi des leurres aléatoires.
4. **L'extraction offensive en boîte noire :** Capacité d'un attaquant externe à forcer la restitution *verbatim* par complétion textuelle (*Prefix-Prompting*).
5. **L'inférence d'appartenance (*Membership Inference Attack - MIA*) :** Séparation statistique entre données vues et non vues.

---

## Architecture du Dépôt

```text
├── README.md                           # Présentation générale du projet et guide d'accueil
├── .gitignore                          # Filtrage strict (exclusion des checkpoints lourds)
│
├── week3/                              # Sprint 3 : Mémorisation empirique & Overfitting
│   ├── data/                           # Données d'expérimentation scellées
│   │   ├── canaries_truth.json         # 50 canaris secrets contrôlés (k = 1, 5, 10, 50)
│   │   ├── dataset_train.json          # Corpus porteur métier (1500 documents Enron/AESLC)
│   │   └── dataset_val.json            # Corpus de validation (250 documents hors-canaris)
│   │
│   ├── figures/                        # 4 Figures scientifiques haute résolution (300 DPI)
│   │   ├── figure1_training_overfitting.png
│   │   ├── figure2_exposure_carlini.png
│   │   ├── figure3_extraction_rates.png
│   │   └── figure4_pfe_synthesis_memorization.png
│   │
│   ├── presentation/                   # Présentation Beamer officielle Télécom SudParis
│   │   ├── main.tex                    # Source LaTeX (24 diapositives, charte TSP 16:9)
│   │   └── main.pdf                    # Support de présentation compilé
│   │
│   ├── train.py                        # Script d'entraînement GPU (Pythia-160M, AdamW)
│   ├── attack_extraction.py            # Sonde 1 : Attaque d'extraction par préfixe (Emmanuel)
│   ├── eval_exposure.py                # Sonde 3 : Métrique d'Exposure Carlini (Fatima-Zahra)
│   ├── eval_mia.py                     # Sonde 4 : Inférence d'appartenance MIA & AUC-ROC
│   ├── plot_results.py                 # Générateur des graphiques de publication
│   │
│   ├── RAPPORT_EXPERIENCES_SPRINT3.md  # Rapport scientifique complet et analyse des résultats
│   └── README_BINOME_FATIMA_ZAHRA.md   # Guide pas-à-pas dédié aux évaluations de Fatima-Zahra
│
├── server/                             # Documentation du serveur de calcul TSP (rst-goose)
│   └── srv.md
└── template/                           # Modèles et charte graphique Télécom SudParis
```

---

##  Prise en Main Rapide

### 1. Environnement Python (Conda)
```bash
conda create -n pfe_memorization python=3.10 -y
conda activate pfe_memorization
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.42,<4.47" datasets matplotlib numpy
```

### 2. Évaluation des Sondes de Mémorisation
```bash
cd week3

# Évaluation de l'Exposure (The Secret Sharer)
python eval_exposure.py --model_path checkpoints/ep20 --epoch 20

# Évaluation de l'Inférence d'Appartenance (MIA)
python eval_mia.py --model_path checkpoints/ep20 --epoch 20

# Attaque d'extraction par préfixe (Boîte noire)
python attack_extraction.py --model_path checkpoints/ep50 --epoch 50
```

> **Note sur les Checkpoints :** Conformément aux bonnes pratiques de Git, les poids des modèles (`.safetensors`, plusieurs Go) ne sont pas versionnés sur ce dépôt public. Ils sont stockés sur le serveur de calcul partagé `rst-goose` ou disponibles sur demande.

---



