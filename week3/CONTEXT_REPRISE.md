# Contexte et Journal de Bord — Reprise du Projet (Sprint 3)
*Fichier créé le 18 octobre 2026 pour assurer une continuité parfaite de la session de travail.*

---

## 📌 1. Vue d'Ensemble du Projet
* **Cadre :** Projet de Fin d'Études (PFE) — Mastère / Cycle Ingénieur Télécom SudParis (Spécialité VAP SSR - Cybersécurité).
* **Sujet :** Sécurité de l'IA Générative : Étude de la mémorisation dans les modèles génératifs (Liens avec le surapprentissage et implications pour la confidentialité des données).
* **Encadrants :** Maryline Laurent (Télécom SudParis) $\cdot$ Faiz Taleb (TSP / EDF).
* **Binôme :** Emmanuel SININI & Fatima-Zahra Rhoulam.
* **Environnement de calcul fourni :** Serveur partagé `rst-goose.telecom-sudparis.eu` (2 × NVIDIA L40S 48 Go VRAM chacune, 100 cœurs CPU, 15 To disque).

---

## 🎯 2. Décisions Validées (Consignes de la Réunion Week 2)
1. **Modèle exact :** `"EleutherAI/pythia-160m"` (162M paramètres, standard mondial de la recherche sur la mémorisation et le scaling).
2. **Corpus porteur :** Correspondance d'entreprise (`aeslc` - Enron emails) simulant un assistant LLM interne déployé en entreprise / EDF.
3. **Vérité Terrain (Canaris) :** 50 secrets synthétiques à haute entropie insérés avec un préfixe fixe (`"CONFIDENTIEL PROJET SSR - Clé d'accès interne : "`) répartis en 4 groupes de fréquence :
   - $k = 1$ (15 canaris) : Rareté extrême / longue traîne.
   - $k = 5$ (15 canaris) : Fréquence modérée.
   - $k = 10$ (10 canaris) : Zone critique d'apparition de l'Exposure.
   - $k = 50$ (10 canaris) : Mémorisation *verbatim* attendue à 100 %.
4. **Répartition des rôles :**
   - **Fatima-Zahra (GPU 1) :** Inférence d'Appartenance (MIA) + Métrique d'Exposure (*The Secret Sharer*, Carlini 2019).
   - **Emmanuel (GPU 0) :** Entraînement/Checkpoints + Extraction par préfixe (*Prefix-Prompting*) + Filtre Perplexité/zlib.

---

## 🛠️ 3. État d'Avancement Technique (Ce qui est fait aujourd'hui)
Tous les fichiers de la Semaine 3 ont été structurés, programmés et validés localement :

1. [week3/plan_week3.md](week3/plan_week3.md) : Feuille de route globale (philosophie de validation sans surcharge technique).
2. [week3/protocole_serveur_binome.md](week3/protocole_serveur_binome.md) : Répartition détaillée, règles de déterminisme (`set_seed`), séparation des GPU L40S, utilisation de `tmux`.
3. [week3/setup_server.sh](week3/setup_server.sh) : Script Bash automatisant la création de l'environnement Conda `pfe_memorization` et l'installation de PyTorch CUDA sur le serveur.
4. [week3/test_gpu.py](week3/test_gpu.py) : Script de test d'allocation mémoire et de détection des 2 GPU L40S.
5. [week3/generate_canaries.py](week3/generate_canaries.py) : **Exécuté avec succès !** A produit :
   - `week3/data/canaries_truth.json` (les 50 canaris et leurs métadonnées).
   - `week3/data/dataset_train.json` (1500 documents contenant 690 occurrences de canaris).
   - `week3/data/dataset_val.json` (250 documents de validation sans canaris).
6. [week3/eval_exposure.py](week3/eval_exposure.py) : Sonde pour Fatima-Zahra, testée et fonctionnelle (calcule l'Exposure en bits sur 50 variantes et sauvegarde dans `week3/results/fatimazahra/`).
7. [week3/train.py](week3/train.py) : Script d'entraînement d'Emmanuel pour fine-tuner Pythia-160M et sauvegarder les checkpoints aux époques 1, 5, 10, 20 et 50.
8. [week3/attack_extraction.py](week3/attack_extraction.py) : Sonde d'extraction d'Emmanuel, testée et fonctionnelle (mesure le taux d'Exact Match par $k$ et le ratio PPL/zlib, sauvegarde dans `week3/results/emmanuel/`).
9. **Transmission binôme :** Une archive ZIP du dossier `week3` a été envoyée à Fatima-Zahra pour qu'elle puisse installer son environnement et lancer ses évaluations de son côté.

---

## 🚀 4. Plan de Reprise pour Demain (Focus Partie Emmanuel)

Dès la reprise, voici l'enchaînement exact à dérouler :

### Étape 1 : Connexion au serveur de calcul
Se connecter en SSH depuis le terminal :
```bash
ssh <LoginTSP>@rst-goose.telecom-sudparis.eu
```

### Étape 2 : Déploiement de l'environnement sur le serveur
* Uploader le dossier `week3` sur le serveur :
  ```bash
  # Depuis votre machine locale :
  scp -r week3/ <LoginTSP>@rst-goose.telecom-sudparis.eu:~/
  ```
* Lancer la configuration de l'environnement :
  ```bash
  cd ~/week3
  bash setup_server.sh
  python test_gpu.py
  ```

### Étape 3 : Lancement de l'entraînement sous `tmux`
* Ouvrir une session persistante :
  ```bash
  tmux new -s pfe_training
  conda activate pfe_memorization
  ```
* Lancer le fine-tuning sur le **GPU 0** :
  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python week3/train.py --epochs 50 --lr 3e-4 --batch_size 16
  ```
  *(Laisser tourner : sur la L40S, les 50 époques prendront environ 5 à 10 minutes).*

### Étape 4 : Exécution des sondes offensives
Dès que les checkpoints (`ep1`, `ep5`, `ep10`, `ep20`, `ep50`) sont générés dans `week3/checkpoints/` :
* Lancer l'extraction verbatim et zlib :
  ```bash
  python week3/attack_extraction.py --epoch 20
  python week3/attack_extraction.py --epoch 50
  ```
* Prévenir Fatima-Zahra pour qu'elle lance son `eval_exposure.py` sur les mêmes checkpoints sur le GPU 1.

### Étape 5 : Visualisation des résultats
Tracer les courbes finales d'extraction vs surapprentissage pour la réunion de projet.
