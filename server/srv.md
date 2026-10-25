# Serveur rst-goose.telecom-sudparis.eu

Serveur partagé de calcul pour le ML. Configuration minimale obligatoire : ne pas installer de logiciels système non autorisés ; travailler dans `$HOME`.

## Accès

- SSH : `ssh <LoginTSP>@rst-goose.telecom-sudparis.eu`
- Mot de passe : mot de passe TSP.
- Accès externe : passer par un jump SSH via `ssh1` ou `ssh2`.

## Matériel

| Ressource | Valeur |
|---|---|
| OS | Ubuntu 24.04 LTS sous Proxmox |
| CPU | 100 cœurs |
| Disque | 15 To |
| GPU | 2 × NVIDIA LS40, partagés |

## Machine learning

- Environnements Conda dans `$HOME` :
  ```bash
  conda env create environment_name python=3.9
  conda activate environment_name
  ```
- Installation de paquets : `pip install <paquet>` ou `conda install <paquet>`.
- PyTorch / TensorFlow peuvent être installés dans l’environnement utilisateur.

## GPU

- Surveillance :
  ```bash
  nvtop
  nvidia-smi
  ```
- GPU strictement partagés : vérifier la charge avant usage, éviter la monopolisation.

## Logiciels

- Pas de droits d’administration système.
- Installation utilisateur possible via Homebrew on Linux :
  ```bash
  brew install <paquet>
  ```
- Préinstallés : `vim`, `nano`, `git`, `htop`, `btop`, `tmux`, `screen`, `tree`, `curl`, `wget`, `zip`, `unzip`, `file`, `build-essential`, `procps`.

## Stockage et sauvegarde

- RAID présent, mais **aucune sauvegarde** des répertoires personnels.
- Aucun quota actuellement : ne pas saturer les 15 To.
- Sauvegarder code, modèles et résultats importants hors du serveur.

## Bonnes pratiques

- Utiliser `tmux` ou `screen` pour les tâches longues.
- Stocker environnements, datasets et checkpoints dans `$HOME`.
- Éviter les duplications de gros fichiers.
- Vérifier l’état GPU avant et pendant les entraînements.

## Administration / TODO

- Décision possible sur l’ajout de quotas.
- Wiki, HOWTO et documentation à mettre en place.
- Monitoring du serveur à déployer.
