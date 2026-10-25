# Guide d'Exécution — Volet Fatima-Zahra (Sprint 3)
**PFE : Mémorisation & Inférence d'Appartenance dans les Modèles Génératifs**  
**Binôme :** Emmanuel SININI & Fatima-Zahra Rhoulam  
**Encadrants :** Maryline Laurent (TSP) & Faiz Taleb (TSP / EDF)  
**Serveur de calcul :** `rst-goose.telecom-sudparis.eu` (NVIDIA L40S)

---

## 🎯 1. Rôle et Objectif de votre Volet

Dans ce Sprint 3, **vous n'avez aucun entraînement à réaliser**.  
Emmanuel a déjà réalisé le fine-tuning complet de `EleutherAI/pythia-160m` sur 50 époques avec les 50 canaris et a sauvegardé les checkpoints pivots.

> ⚠️ **Principe scientifique fondamental :**  
> Conformément aux exigences de Maryline Laurent et Faiz Taleb (*« Pour comparer, il faut exactement les mêmes conditions »*), nous devons tester **le même modèle unique** (mêmes paramètres neuronaux $\theta$). Ré-entraîner un modèle produirait une divergence stochastique qui rendrait la corrélation croisée impossible.

Votre mission consiste à réaliser **l'audit statistique et l'évaluation de confidentialité** sur le **GPU 1** :
1. **Sonde 3 — Métrique d'Exposure (*The Secret Sharer*, Carlini et al. 2019) :**  
   Mesurer la singularité de chaque canari face à 50 leurres aléatoires sur les époques 1, 10, 20 et 50.
2. **Sonde 4 — Inférence d'Appartenance (*MIA*, Yeom et al. 2018) :**  
   Mesurer la capacité d'un auditeur à distinguer les données vues à l'entraînement (*IN*) des données hors-corpus (*OUT*) selon le seuil de perte, calculer le score AUC-ROC et la matrice de confusion.

---

## 📁 2. Contenu de cette Archive

```text
pack_fatimazahra/
├── README_BINOME_FATIMA_ZAHRA.md   # Ce guide pas-à-pas
├── eval_exposure.py                # Sonde 3 : Calcul de l'Exposure en bits (Carlini)
├── eval_mia.py                     # Sonde 4 : Attaque MIA, AUC-ROC et matrice de confusion
├── data/
│   ├── canaries_truth.json         # Vérité terrain des 50 canaris (k = 1, 5, 10, 50)
│   ├── dataset_train.json          # 1500 documents d'entraînement (IN-sample)
│   └── dataset_val.json            # 250 documents de validation (OUT-sample)
├── resultats_extraction.txt        # Résultats de référence obtenus par Emmanuel (Sonde 1)
├── resultats_exposure.txt          # Résultats de référence obtenus sur l'Exposure
└── RAPPORT_EXPERIENCES_SPRINT3.md  # Rapport complet du Sprint 3 pour votre information
```

---

## 💾 3. Récupération des Checkpoints sur le Serveur

Comme les répertoires personnels `$HOME` sont strictement isolés sur `rst-goose`, Emmanuel a déposé les checkpoints dans le dossier d'échange partagé du serveur :
📍 **Dossier d'échange :** `/tmp/pfe_checkpoints/`

Pour les importer directement dans votre session (sans jamais avoir besoin d'accéder au `$HOME` d'Emmanuel), lancez simplement :

```bash
# Dans votre dossier de projet week3 sur rst-goose :
mkdir -p checkpoints
cp -r /tmp/pfe_checkpoints/* ./checkpoints/
```

Vous disposerez alors localement des 5 checkpoints :
* `checkpoints/ep1` (modèle initial de base)
* `checkpoints/ep5`
* `checkpoints/ep10`
* `checkpoints/ep20` (point d'inflexion critique d'overfitting)
* `checkpoints/ep50` (surapprentissage maximal)

---

## 🚀 4. Protocole d'Exécution Pas-à-Pas sur le GPU 1

### Étape 1 : Connexion et session persistante
Connectez-vous à `rst-goose` et lancez une session persistante `tmux` :
```bash
ssh <votre_login>@rst-goose.telecom-sudparis.eu
tmux new -s audit_fatimazahra
conda activate pfe_memorization
```

### Étape 2 : Assignation stricte du GPU 1
Pour travailler sur le second GPU sans interférer avec le GPU 0 :
```bash
export CUDA_VISIBLE_DEVICES=1
```

---

### Étape 3 : Exécution de la Sonde 3 (Exposure de Carlini)
Lancez l'évaluation de l'Exposure sur les époques clés :

```bash
# Époque 1 (Modèle initial)
python eval_exposure.py --model_path checkpoints/ep1 --epoch 1

# Époque 10 (Zone de consolidation)
python eval_exposure.py --model_path checkpoints/ep10 --epoch 10

# Époque 20 (Décrochage d'overfitting)
python eval_exposure.py --model_path checkpoints/ep20 --epoch 20

# Époque 50 (Surapprentissage maximal)
python eval_exposure.py --model_path checkpoints/ep50 --epoch 50
```

*Les fichiers JSON seront automatiquement générés dans `results/fatimazahra/exposure_epoch_*.json`.*

---

### Étape 4 : Exécution de la Sonde 4 (Inférence d'Appartenance - MIA)
Lancez l'attaque d'inférence d'appartenance pour mesurer l'AUC-ROC et la matrice de confusion :

```bash
# Évaluation sur le modèle généraliste (Époque 1)
python eval_mia.py --model_path checkpoints/ep1 --epoch 1

# Évaluation au point de bascule (Époque 20)
python eval_mia.py --model_path checkpoints/ep20 --epoch 20

# Évaluation sous surapprentissage massif (Époque 50)
python eval_mia.py --model_path checkpoints/ep50 --epoch 50
```

*Les rapports d'inférence seront enregistrés dans `results/fatimazahra/mia_epoch_*.json`.*

---

## 📊 5. Ce qu'il faut transmettre à Emmanuel pour la Présentation

Dès la fin de vos runs, il vous suffit de transmettre à Emmanuel :
1. Le dossier `results/fatimazahra/` contenant les fichiers `exposure_epoch_*.json` et `mia_epoch_*.json`.
2. Si vous avez tracé des graphiques complémentaires (distributions individuelles ou courbes ROC).

Emmanuel a déjà préparé la **Diapositive 19** de la présentation Beamer officielle Télécom SudParis (`week3/presentation/main.pdf`) pour y insérer directement vos graphiques et vos métriques MIA !

---
*Bon travail pour ces évaluations !*
