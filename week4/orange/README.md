# Volet Orange — Sprint 4 : DP-SGD, audits et utilité

Contre-mesure 1 : entraîner pythia-160m avec la **confidentialité différentielle** (DP-SGD, bibliothèque Opacus), puis mesurer ce qu'elle change sur la fuite (Exposure, MIA apparié) et sur la qualité du modèle (perplexité).

Dossier à placer dans le dépôt : `~/PFE/week4/orange/` (les scripts lisent automatiquement `week4/data/`).

## Contenu

| Fichier | Rôle |
|---|---|
| `lancer_orange.sh` | Lance tout : 3 entraînements, audits de 4 à 5 modèles, figures |
| `train_dp.py` | Entraînement DP-SGD (ou référence sans DP avec `--no_dp`) |
| `eval_exposure.py` | Sonde 3 : Exposure à 10 000 leurres (même méthode qu'au Sprint 3) |
| `eval_mia.py` | Sonde 4 : MIA apparié (+ test naïf pour comparaison) |
| `eval_utilite.py` | Perplexité de validation, perplexité sur texte général, exemples de génération |
| `plot_orange.py` | 4 figures + `figures/resume.md` (tableau récapitulatif) |
| `check_env.py`, `commun.py` | Vérification de l'environnement, fonctions partagées |

## Les modèles comparés

| Nom (`runs/<nom>/`) | Ce que c'est | Pourquoi |
|---|---|---|
| `base` | pythia-160m d'origine, jamais fine-tuné | Niveau du hasard |
| `sprint3_ep50` | Modèle du Sprint 3 (époque 50), si `/tmp/pfe_checkpoints/ep50` existe | Lien avec le Sprint 3 |
| `nodp` | Même entraînement que les modèles DP, **sans DP** | Référence équitable : mêmes données, même lr, mêmes lots |
| `dp_eps8` | DP-SGD, ε ≈ 8 (protection modérée) | |
| `dp_eps2` | DP-SGD, ε ≈ 2 (protection forte) | |

## Lancer (sur rst-goose)

```bash
cd ~/PFE/week4/orange
conda activate pfe_memorization
pip install opacus
export CUDA_VISIBLE_DEVICES=1
python check_env.py                      # doit finir par ENVIRONNEMENT OK

# essai rapide d'abord (environ 30 à 45 min) :
EPOCHS=2 VARIANTS=1000 bash lancer_orange.sh
# si tout passe : effacer l'essai puis lancer le vrai run dans tmux
rm -rf runs figures logs
tmux new -s orange -d "source ~/.bashrc; conda activate pfe_memorization; bash ~/PFE/week4/orange/lancer_orange.sh 2>&1 | tee ~/PFE/week4/orange/orange.log"
tail -f ~/PFE/week4/orange/orange.log
```

Durée estimée du vrai run : 4 à 6 h (environ 30 min pour `nodp`, 1 à 2 h par modèle DP, environ 10 min d'audit par modèle). Un modèle déjà entraîné n'est pas réentraîné : après une coupure, relancer la même commande.

## Ce que les scripts corrigent par rapport à `week4/`

- **ε réellement visé** : `make_private_with_epsilon` calcule le bruit σ qui donne ε = 8 ou 2. Avec les réglages du README d'origine, σ = 0,5 donnait ε ≈ 33 et σ = 1,0 donnait ε ≈ 4,8.
- **Pas de mode simulé** : sans Opacus ou sans GPU, le script s'arrête avec une erreur.
- **Padding exclu** et perte calculée **par document** : chaque document pèse pareil dans le clipping.
- **Mêmes audits qu'au Sprint 3** (10 000 leurres, MIA apparié) : les anciens `week4/eval_*.py` utilisaient 50 leurres et le test MIA biaisé, et leur option `--checkpoint` n'existe pas.
- **Référence sans DP** entraînée avec les mêmes réglages, pour isoler l'effet de la DP.
- **Perplexité** mesurée de la même façon pour tous les modèles.

## Lire les résultats

`figures/resume.md` contient le tableau final. Pour chaque modèle :

- **ε** : budget de confidentialité. Plus il est petit, plus la garantie est forte.
- **PPL val** : perplexité sur les documents de validation. Plus elle est basse, meilleur est le modèle.
- **PPL générale** : perplexité sur du texte encyclopédique (wikitext-2). Elle montre si le modèle a gardé ses connaissances générales.
- **AUC MIA apparié** : 0,5 = aucune fuite détectable sur les documents.
- **Exposure par k** : environ 1,4 bit = hasard ; 13,3 bits = secret classé 1er parmi 10 001.

## Point clé à vérifier dans les résultats : la confidentialité de groupe

La garantie (ε, δ) protège **un document**. Un secret k = 50 apparaît dans 50 documents : sa protection formelle tombe à environ 50 × ε (aucune garantie utile). Seuls les secrets k = 1 sont couverts. Il est donc attendu que l'Exposure baisse nettement pour k = 1 et k = 5, mais reste élevée pour k = 50. C'est un résultat à présenter, qui justifie de combiner la DP avec la **déduplication** (volet d'Emmanuel).

## Améliorations possibles (si le temps le permet)

1. **Plus de points sur la courbe** : `EPSILONS="1 2 4 8 16" bash lancer_orange.sh` donne un vrai front de Pareto.
2. **Utilité à niveau égal** : la DP apprend plus lentement. Comparer aussi des modèles DP entraînés plus longtemps (`EPOCHS=100`), à ε fixé.
3. **Plusieurs graines** : relancer avec `--seed 43` et `--seed 44` pour obtenir des barres d'erreur.
4. **DP + déduplication** : réentraîner en DP sur `week4/data/dataset_deduplicated.json` d'Emmanuel. C'est la combinaison qui devrait protéger aussi les secrets répétés.
