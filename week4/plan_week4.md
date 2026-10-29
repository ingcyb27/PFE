# Feuille de Route — Sprint 4 : Conception, Implémentation & Évaluation des Contre-Mesures de Confidentialité
## Sécurité de l'IA Générative : Neutralisation de la Mémorisation & Préservation de l'Utilité

**Étudiants :** Emmanuel SININI (Volet Bleu 🔵) & Fatima-Zahra Rhoulam (Volet Orange 🟠)  
**Encadrants :** Maryline Laurent (Télécom SudParis) $\cdot$ Faiz Taleb (TSP / EDF)  
**Environnement de calcul :** Serveur TSP `rst-goose` (2 × NVIDIA L40S 48 Go VRAM, CUDA 13.0)  
**Période :** Semaine 4  

---

## 🎯 1. Synthèse de la Réunion Sprint 3 & Orientation Stratégique du Sprint 4

La présentation du Sprint 3 (`week3/presentation/main.tex`) a validé empiriquement le lien direct entre surapprentissage et mémorisation. Nos encadrants ont validé nos perspectives et ont formulé une consigne méthodologique majeure :

> 📢 **Retour clé des Encadrants :**  
> *« 50 époques n'est pas suffisant pour observer la pleine dynamique d'un modèle de langage. Poussez l'entraînement jusqu'à **1000 époques** : en laissant le modèle tourner longuement, les courbes de perte (Loss) atteindront leur régime asymptotique et seront incontestables pour démontrer la divergence absolue entre la perte d'entraînement et la perte de validation. »*

### Les 3 Contre-Mesures Validées pour le Sprint 4 :
1. <span style="color:#FD7E14; font-weight:bold;">Contre-Mesure 1 (DP-SGD / Opacus) :</span> Entraînement différentiellement privé par descente de gradient stochastique bruitée (clipping $C$, bruit gaussien $\sigma$) et mesure du trade-off formel confidentialité ($\epsilon, \delta$) vs utilité (Perplexité).
2. <span style="color:#007BFF; font-weight:bold;">Contre-Mesure 2 (Déduplication MinHash) :</span> Déduplication algorithmique en amont pour éliminer la multiplicité des secrets dans le corpus et forcer $k \to 1$.
3. <span style="color:#007BFF; font-weight:bold;">Contre-Mesure 3 (Early Stopping) :</span> Arrêt automatique de l'apprentissage au point d'inflexion optimal (vers l'époque 15-20) avant que le modèle ne commence à surapprendre et mémoriser.

---

## ⚖️ 2. Principe Fondamental : Indépendance Totale des Deux Membres du Binôme

Pour maximiser l'efficacité et éliminer tout temps d'attente (aucun blocage où l'un attendrait les checkpoints ou les sorties de l'autre), le travail est scindé en **deux piliers totalement autonomes et parallélisables** :

```
                                  [DATASET INITIAL SPRINT 3]
                            (dataset_train.json & canaries_truth.json)
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
     ┌─────────────────────────────┐       ┌─────────────────────────────┐
     │    🔵 VOLET BLEU (Emmanuel)  │       │ 🟠 VOLET ORANGE (Fatima-Z) │
     │  Défenses Données & Régul.  │       │    Défense Mathématique DP  │
     │     (GPU 0 sur rst-goose)   │       │    (GPU 1 sur rst-goose)    │
     ├─────────────────────────────┤       ├─────────────────────────────┤
     │ 1. Entraînement 1000 époques│       │ 1. Intégration Opacus       │
     │    (Baseline asymptotique)  │       │    (DP-SGD : clipping & σ)  │
     │ 2. Contre-Mesure 3 (CM3) :  │       │ 2. Calcul budget (ε, δ)     │
     │    Early Stopping (ep ~15)  │       │    via RDP Accountant       │
     │ 3. Contre-Mesure 2 (CM2) :  │       │ 3. Audit Sécurité :         │
     │    Déduplication MinHash    │       │    MIA (AUC-ROC) & Exposure │
     │ 4. Évaluation Extraction    │       │ 4. Trade-off Confidentialité│
     │    Verbatim (Sonde 1)       │       │    vs Utilité (Perplexité)  │
     └──────────────┬──────────────┘       └──────────────┬──────────────┘
                    │                                     │
                    │   TRAVAIL 100% INDÉPENDANT EN SEMAINE│
                    │   (Aucune dépendance croisée)       │
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       ▼
                     [CONVERGENCE & LIVRABLE DU SPRINT 4]
                 Tableau Comparatif des 3 Contre-Mesures
            (Extraction Verbatim VS MIA VS Exposure VS Perplexité)
```

### Pourquoi l'indépendance est-elle garantie à 100 % ?
* **Fatima-Zahra (Orange)** part directement du jeu de données déjà existant `week4/data/dataset_train.json`. Elle n'a besoin d'aucun nouveau fichier d'Emmanuel : elle installe `opacus`, lance son propre script d'entraînement DP-SGD sur le GPU 1, génère ses checkpoints privés, et applique directement ses métriques d'Exposure et de MIA (`eval_exposure.py`, `eval_mia.py`). Elle peut comparer immédiatement ses résultats DP aux résultats non-privés du Sprint 3 scellés dans `week3/resultats_exposure.txt` !
* **Emmanuel (Bleu)** travaille sur l'entraînement standard sur le GPU 0 : il pousse la baseline à 1000 époques avec un pas de logging régulier, extrait le point d'Early Stopping (CM3), puis développe l'algorithme MinHash pour créer `dataset_deduplicated.json` (CM2) et ré-entraîne dessus. Il évalue l'impact par extraction verbatim (Sonde 1).
* **Le point de contact :** Une simple mise en commun des chiffres finaux le dernier jour pour fusionner les graphiques et le tableau récapitulatif de la présentation.

---

## 🔵 3. Volet Bleu — Emmanuel SININI
### *Défenses Déterministes : Pipeline Données (MinHash) & Régularisation (1000 Époques & Early Stopping)*

Ce volet s'inscrit dans la continuité directe du rôle d'Emmanuel : gestion de la boucle d'entraînement PyTorch (`train.py`), manipulation du corpus et des canaris ($k \in \{1, 5, 10, 50\}$), et attaque par extraction textuelle directe (Sonde 1 : Prefix-Prompting).

### <span style="color:#007BFF">Tâche B1 : Entraînement Baseline Poussé à 1000 Époques & Courbes Asymptotiques</span>
* **Objectif :** Répondre à la demande explicite des encadrants en confirmant que l'overfitting devient absolu et que les courbes de loss atteignent une convergence asymptotique nette.
* **Méthodologie :**
  * Adapter `train.py` en `train_longrun_earlystop.py` pour exécuter 1000 époques sur `dataset_train.json`.
  * Sauvegarder des logs fins : enregistrement des pertes $\mathcal{L}_{\text{train}}$ et $\mathcal{L}_{\text{val}}$ toutes les 5 époques dans `runs_logs/loss_history_1000ep.json`.
  * Sauvegarde sélective des checkpoints clés : `ep15` (minimum de validation), `ep50`, `ep100`, `ep250`, `ep500`, `ep1000`.
* **Visualisation attendue :** Figure 1bis montrant la divergence spectaculaire : $\mathcal{L}_{\text{train}} \to 0$ ($< 10^{-3}$) tandis que $\mathcal{L}_{\text{val}}$ remonte et plafonne (overfitting complet).

### <span style="color:#007BFF">Tâche B2 : Contre-Mesure 3 (CM3) — Validation de l'Early Stopping</span>
* **Hypothèse :** Arrêter l'entraînement avant la rupture de généralisation empêche l'effondrement des logits sur les canaris rares et élimine l'extraction verbatim.
* **Méthodologie :**
  * Identifier précisément l'époque optimale $E_{\text{opt}}$ où $\mathcal{L}_{\text{val}}$ atteint son minimum absolu (historiquement vers l'époque 15).
  * Sélectionner le checkpoint `checkpoints/checkpoint_earlystop_ep15`.
  * Exécuter l'attaque par préfixe (`attack_extraction.py`) sur le checkpoint Early Stopping VS le checkpoint extrême `ep1000`.
  * Quantifier la protection : taux d'extraction verbatim attendu à **0.0%** à $E_{\text{opt}}$, contre une extraction maximale à l'époque 1000.

### <span style="color:#007BFF">Tâche B3 : Contre-Mesure 2 (CM2) — Déduplication MinHash LSH ($k \to 1$)</span>
* **Hypothèse :** Le Théorème de Feldman et nos résultats du Sprint 3 ont prouvé que les canaris $k=1$ (rareté maximale) ont un taux d'extraction verbatim de 0.0%. Si nous dédupliquons le corpus avant l'entraînement, les secrets répétés ($k=50, 10, 5$) redeviennent des singletons ($k=1$), neutralisant l'extraction même en surapprentissage.
* **Méthodologie :**
  * Développer le script `deduplicate_minhash.py` basé sur le hachage sensible à la localité (MinHash LSH via `datasketch` ou implémentation 3-shingles).
  * Détecter les quasi-doublons de phrases sensibles (similarité de Jaccard $J(A, B) \ge 0.75$).
  * Générer le jeu de données assaini `data/dataset_deduplicated.json` où chaque canari n'apparaît plus qu'une seule fois ($k=1$).
  * Entraîner un modèle dédié sur ce dataset dédupliqué (`checkpoints/model_dedup_1000ep`).
  * Exécuter l'attaque d'extraction : vérifier si l'extraction verbatim des ex-canaris $k=50$ s'effondre à 0%.

---

## 🟠 4. Volet Orange — Fatima-Zahra Rhoulam
### *Défense Mathématique : Confidentialité Différentielle (DP-SGD / Opacus) & Trade-off Confidentialité vs Utilité*

Ce volet s'inscrit dans la continuité directe du rôle de Fatima-Zahra : modélisation statistique, audit de fuite d'information (Métrique d'Exposure de Carlini - Sonde 3), analyse de perte et Inférence d'Appartenance (MIA - Sonde 4, distributions, AUC-ROC).

### <span style="color:#FD7E14">Tâche O1 : Intégration d'Opacus & Entraînement DP-SGD</span>
* **Objectif :** Implémenter la protection mathématique formelle la plus robuste de la littérature en injectant du bruit gaussien calibré dans les gradients lors du fine-tuning.
* **Méthodologie :**
  * Créer le script d'entraînement dédié `train_dp.py` avec la bibliothèque `opacus`.
  * Utiliser `PrivacyEngine` pour encapsuler l'optimiseur et le dataloader.
  * Adapter les couches incompatibles de `EleutherAI/pythia-160m` via `ModuleValidator.fix(model)`.
  * Étalonner les hyperparamètres clés du DP-SGD :
    * Norme maximale de gradient (Clipping) : $C \in \{0.5, 1.0, 2.0\}$.
    * Multiplicateur de bruit gaussien : $\sigma \in \{0.5, 1.0\}$.
    * Cible de probabilité d'échec : $\delta = 10^{-5}$ ($< 1 / N$).
  * Suivre l'accumulation du budget de confidentialité $\epsilon$ à chaque époque via le RDP accountant (Rényi Differential Privacy).
  * Entraîner deux régimes de confidentialité :
    1. Régime **DP Modérée** : $\epsilon \approx 8$ ($\sigma \approx 0.5$).
    2. Régime **DP Forte** : $\epsilon \approx 2$ ($\sigma \approx 1.0$).
  * Sauvegarder les checkpoints : `checkpoints/model_dp_eps8` et `checkpoints/model_dp_eps2`.

### <span style="color:#FD7E14">Tâche O2 : Audit de Confidentialité du Modèle DP (MIA & Exposure)</span>
* **Hypothèse :** La garantie $(\epsilon, \delta)$-DP borne théoriquement la divergence entre les distributions de sortie d'un modèle entraîné avec ou sans une donnée particulière. En pratique, cela doit rendre l'attaque MIA totalement inefficace (AUC-ROC $\to 0.50$, soit le hasard pur).
* **Méthodologie :**
  * **Sonde 4 (Audit MIA) :** Exécuter `eval_mia.py` sur les modèles protégés par DP-SGD.
    * Tracer la courbe ROC et calculer l'AUC-ROC.
    * Vérifier la baisse de l'AUC-ROC (qui culminait à $> 0.85$ en Sprint 3) vers $0.50 - 0.55$.
  * **Sonde 3 (Audit Exposure) :** Exécuter `eval_exposure.py` sur les modèles DP.
    * Calculer l'Exposure de Carlini sur les 50 canaris face à leurs 50 leurres aléatoires.
    * Démontrer que le bruit de gradient disperse le rang du vrai secret et réduit l'Exposure moyenne de $5.67$ bits vers des valeurs proches de 0 bit.

### <span style="color:#FD7E14">Tâche O3 : Mesure du Trade-off Confidentialité vs Utilité (*Privacy-Utility Trade-off*)</span>
* **Problématique :** Protéger les données au détriment de l'utilité rend le modèle inutilisable. Quelle perte de qualité linguistique subit-on pour chaque gain en confidentialité ?
* **Méthodologie :**
  * Évaluer la Perplexité de validation ($\text{PPL}_{\text{val}}$) sur le corpus de test `dataset_val.json` pour chaque modèle :
    * Modèle Non-Privé (Baseline).
    * Modèle DP Modérée ($\epsilon \approx 8$).
    * Modèle DP Forte ($\epsilon \approx 2$).
  * Analyser qualitativement la cohérence de génération de texte (effet du bruit de gradient sur la syntaxe et la grammaire).
  * Construire le graphique du **Front de Pareto : Confidentialité ($\epsilon$) vs Utilité ($\text{PPL}$)**.

---

## 📊 5. Matrice Récapitulative des Tâches & Indépendance Opérationnelle

| Volet & Responsable | Contre-Mesure / Tâche | Données Utilisées | Script Développé / Exécuté | Métrique Clé Mesurée | Dépendance ? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| <span style="color:#007BFF; font-weight:bold;">🔵 Emmanuel</span> | **Baseline 1000 époques** | `dataset_train.json` | `train_longrun_earlystop.py` | Courbes $\mathcal{L}_{\text{train}}$ vs $\mathcal{L}_{\text{val}}$ | **Aucune** (Autonome) |
| <span style="color:#007BFF; font-weight:bold;">🔵 Emmanuel</span> | **CM3 : Early Stopping** | Checkpoint $E_{\text{opt}}$ (~ep15) | `attack_extraction.py` | Taux d'extraction verbatim | **Aucune** (Autonome) |
| <span style="color:#007BFF; font-weight:bold;">🔵 Emmanuel</span> | **CM2 : Déduplication MinHash** | `dataset_train.json` | `deduplicate_minhash.py` + ré-entraînement | Taux d'extraction sur $k \to 1$ | **Aucune** (Autonome) |
| <span style="color:#FD7E14; font-weight:bold;">🟠 Fatima-Zahra</span> | **CM1 : DP-SGD (Opacus)** | `dataset_train.json` | `train_dp.py` | Budget formel $(\epsilon, \delta)$ | **Aucune** (Autonome) |
| <span style="color:#FD7E14; font-weight:bold;">🟠 Fatima-Zahra</span> | **Audit Confidentialité DP** | Checkpoints DP ($\epsilon=8, 2$) | `eval_mia.py` & `eval_exposure.py` | AUC-ROC ($\to 0.5$) & Exposure | **Aucune** (Autonome) |
| <span style="color:#FD7E14; font-weight:bold;">🟠 Fatima-Zahra</span> | **Trade-off Sécurité/Utilité** | `dataset_val.json` | Calcul Perplexité PPL | PPL vs $\epsilon$ | **Aucune** (Autonome) |

---

## 📅 6. Découpage Temporel — Jalons de la Semaine 4

```
Jalons      🔵 Volet Bleu (Emmanuel)                            🟠 Volet Orange (Fatima-Zahra)
────────────────────────────────────────────────────────────────────────────────────────────────
Jour 1      Configuration script 1000 époques + Early Stop      Installation Opacus + Validation Module
Jour 2      Lancement entraînement 1000 ep (GPU 0)              Implémentation train_dp.py (GPU 1)
Jour 3      Implémentation MinHash LSH (deduplicate.py)         Entraînement DP-SGD (ε=8 et ε=2)
Jour 4      Entraînement modèle dédupliqué + Tests Extraction   Audits MIA (AUC-ROC), Exposure & PPL
Jour 5      ────────────────────────────────────────────────────────────────────────────────────
                        SYNTHÈSE CROISÉE & CONCEPTION DES SLIDES SPRINT 4
                        Fusion du Tableau Comparatif des 3 Contre-Mesures
```

### Jalon J1 : Préparation de l'Environnement de Défense
* 🔵 **Emmanuel :**
  * Créer le dossier d'entraînement long `week4/runs_logs/`.
  * Rédiger `week4/train_longrun_earlystop.py` avec arrêt précoce configurable et logging JSON fin.
* 🟠 **Fatima-Zahra :**
  * Installer `opacus` dans l'environnement conda `pfe_memorization` sur `rst-goose` (`pip install opacus`).
  * Valider la compatibilité de `EleutherAI/pythia-160m` avec `ModuleValidator.validate(model)`.

### Jalon J2 : Entraînements Parallèles sur les Deux GPU
* 🔵 **Emmanuel (GPU 0) :**
  * Lancer l'entraînement long sur 1000 époques.
  * Observer la trajectoire de perte et sceller le checkpoint au minimum de $\mathcal{L}_{\text{val}}$ (Early Stopping).
* 🟠 **Fatima-Zahra (GPU 1) :**
  * Lancer le premier entraînement DP-SGD ($\sigma = 0.5, C = 1.0$) pour atteindre $\epsilon \approx 8$.
  * Enregistrer l'historique du budget $(\epsilon, \delta)$ par époque.

### Jalon J3 : Déduplication & Régimes DP Poussés
* 🔵 **Emmanuel :**
  * Coder et exécuter `deduplicate_minhash.py` sur `dataset_train.json`.
  * Vérifier que tous les canaris $k=50, 10, 5$ sont réduits à une occurrence unique ($k=1$).
  * Lancer l'entraînement sur ce dataset assaini.
* 🟠 **Fatima-Zahra :**
  * Lancer le second entraînement DP-SGD à forte confidentialité ($\sigma = 1.0, C = 1.0$) pour atteindre $\epsilon \approx 2$.
  * Enregistrer les métriques d'apprentissage sous fort bruit.

### Jalon J4 : Évaluations Croisées et Mesures d'Impact
* 🔵 **Emmanuel :**
  * Lancer `attack_extraction.py` sur les modèles :
    1. Baseline 1000 époques (overfitting extrême).
    2. Modèle Early Stopping (époque ~15).
    3. Modèle entraîné sur données dédupliquées MinHash.
  * Comparer les taux d'extraction verbatim.
* 🟠 **Fatima-Zahra :**
  * Exécuter `eval_mia.py` sur la baseline non-privée VS les modèles DP : tracer la chute de l'AUC-ROC vers 0.50.
  * Exécuter `eval_exposure.py` : quantifier la réduction en bits de l'Exposure de Carlini.
  * Calculer la perplexité de test pour quantifier le coût en utilité de la DP.

### Jalon J5 : Consolidation & Préparation des Livrables
* Rédaction du rapport d'expérimentation du Sprint 4 (`week4/RAPPORT_DEFENSES_SPRINT4.md`).
* Création des figures comparatives pour la présentation :
  * Figure 1 : Courbe de loss sur 1000 époques avec mise en évidence du point d'Early Stopping.
  * Figure 2 : Efficacité de la Déduplication MinHash sur l'extraction verbatim selon $k$.
  * Figure 3 : Protection différentielle : Évolution de l'AUC-ROC et de l'Exposure en fonction de $\epsilon$.
  * Figure 4 : Courbe de Pareto : Confidentialité ($\epsilon$) vs Utilité ($\text{PPL}_{\text{val}}$).
* Préparation du jeu de diapositives Beamer (`week4/presentation/main.tex`).

---

## 🏆 7. Livrable Final du Sprint 4 : Le Tableau Comparatif Global

À l'issue de la semaine, la présentation aux encadrants articulera l'ensemble des résultats autour d'un grand tableau de synthèse comparant objectivement les trois mécanismes de défense :

| Modèle & Défense | Risque Extraction Verbatim (Sonde 1 - Bleu) | Fuite d'Exposure Carlini (Sonde 3 - Orange) | Vulnérabilité MIA / AUC-ROC (Sonde 4 - Orange) | Dégradation Utilité / Perplexité | Complexité & Faisabilité Industrielle |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline Overfittée (1000 ep)** | 🔴 **Très élevé** (~30-50%) | 🔴 **Saturée** (5.67 bits) | 🔴 **Vulnérable** ($\text{AUC} \ge 0.85$) | ✅ Référence optimale | Aucune défense appliquée |
| **CM3 : Early Stopping (ep ~15)** | 🟢 **Nul** (0.0% extraction) | 🟡 **Moyenne** (~3.5 bits) | 🟡 **Modérée** ($\text{AUC} \approx 0.65$) | ✅ Aucune perte d'utilité | ✅ Simple (zéro surcoût) |
| **CM2 : Déduplication MinHash** | 🟢 **Nul** (0.0% extraction) | 🟡 **Partielle** ($k=1$ résiduel) | 🔴 **Vulnérable** (sur les non-doublons) | ✅ Préservée à 100% | ✅ Pré-traitement data rapide |
| **CM1 : DP-SGD ($\epsilon \approx 8$)** | 🟢 **Nul** (0.0% extraction) | 🟢 **Faible** ($< 1.5$ bit) | 🟢 **Forte protection** ($\text{AUC} \approx 0.58$) | 🟡 Perte légère de PPL | ⚠️ Surcoût de calcul ($\times 2$) |
| **CM1 : DP-SGD ($\epsilon \approx 2$)** | 🟢 **Nul** (0.0% extraction) | 🟢 **Nulle** ($\approx 0.0$ bit) | 🟢 **Indiscernable** ($\text{AUC} \approx 0.51$) | 🔴 Perte notable de syntaxe | ⚠️ Surcoût + calibration fine |

---

## 💡 8. Recommandations Pratiques & Commandes d'Exécution

### Installation des Dépendances sur `rst-goose`
```bash
# Dans l'environnement pfe_memorization :
conda activate pfe_memorization
pip install opacus datasketch
```

### Exécution du Volet Bleu (Emmanuel) sur GPU 0
```bash
# 1. Lancement de l'entraînement 1000 époques avec Early Stopping
CUDA_VISIBLE_DEVICES=0 python3 week4/train_longrun_earlystop.py \
    --train_path week4/data/dataset_train.json \
    --val_path week4/data/dataset_val.json \
    --epochs 1000 \
    --lr 3e-4 \
    --batch_size 16 \
    --output_dir week4/checkpoints/baseline_1000ep

# 2. Déduplication MinHash LSH
python3 week4/deduplicate_minhash.py \
    --input_path week4/data/dataset_train.json \
    --output_path week4/data/dataset_deduplicated.json \
    --threshold 0.75

# 3. Évaluation de l'extraction verbatim
CUDA_VISIBLE_DEVICES=0 python3 week4/attack_extraction.py \
    --checkpoint week4/checkpoints/baseline_1000ep/checkpoint_earlystop \
    --canaries_path week4/data/canaries_truth.json
```

### Exécution du Volet Orange (Fatima-Zahra) sur GPU 1
```bash
# 1. Entraînement DP-SGD avec Opacus
CUDA_VISIBLE_DEVICES=1 python3 week4/train_dp.py \
    --train_path week4/data/dataset_train.json \
    --val_path week4/data/dataset_val.json \
    --sigma 0.5 \
    --clipping 1.0 \
    --target_delta 1e-5 \
    --epochs 50 \
    --output_dir week4/checkpoints/model_dp_eps8

# 2. Audit MIA sur le modèle DP
CUDA_VISIBLE_DEVICES=1 python3 week4/eval_mia.py \
    --checkpoint week4/checkpoints/model_dp_eps8 \
    --train_path week4/data/dataset_train.json \
    --val_path week4/data/dataset_val.json

# 3. Audit Exposure sur le modèle DP
CUDA_VISIBLE_DEVICES=1 python3 week4/eval_exposure.py \
    --checkpoint week4/checkpoints/model_dp_eps8 \
    --canaries_path week4/data/canaries_truth.json
```
