# Plan d'Action - Sprint 2 : Étude Architecturale et Déduction des Attaques

## 🎯 Objectif du Sprint
L'objectif de cette semaine est de faire un vrai travail de rétro-ingénierie intellectuelle. Avant de lancer des attaques à l'aveugle, nous allons **étudier en détail l'architecture et la logique interne de plusieurs petits modèles**. 
C'est seulement à l'issue de cette étude que nous déduirons nous-mêmes quelles attaques (Extraction, MIA, etc.) sont théoriquement possibles. Le livrable sera une présentation exposant nos déductions.

---

> **Répartition du travail (Binômes) :** 
> 🔵 **Périmètre Bleu :** Analyse des modèles Génératifs (Texte & Image par Diffusion).
> 🟠 **Périmètre Orange :** Analyse des modèles Discriminatifs / Séquentiels (Classification Image & Séries Temporelles).

---

## 🔬 1. Méthodologie de Recherche et Bases de Code

Pour chaque architecture, l'objectif est d'explorer le code source d'un petit modèle "jouet", d'en comprendre la mécanique, et d'en déduire les vecteurs d'attaque potentiels pour l'implémentation en Semaine 3.

### <span style="color:#007BFF">🔵 L'Équipe Bleue : Focus sur le Génératif</span>
**1. Texte (Transformer basique type GPT)**
*   **🔗 Modèle d'étude (Code) :** [NanoGPT (Andrej Karpathy)](https://github.com/karpathy/nanoGPT) - Implémentation minimaliste d'un Transformer.
*   **À étudier :** Comment les probabilités (logits) sont calculées token par token lors de la génération.
*   **Objectif :** Déduire comment formuler un préfixe (Prompt) pour forcer le modèle à régurgiter une donnée mémorisée.

**2. Image (Modèle de Diffusion)**
*   **🔗 Modèle d'étude (Code) :** [Min-Diffusion (lucidrains)](https://github.com/lucidrains/denoising-diffusion-pytorch) - Modèle de diffusion en quelques centaines de lignes.
*   **À étudier :** La logique de débruitage (denoising) pas à pas.
*   **Objectif :** Déduire comment le modèle gère et restitue les images atypiques (outliers) vues à l'entraînement.

---

### <span style="color:#FD7E14">🟠 L'Équipe Orange : Focus sur la Prédiction</span>
**3. Image (Classification classique type CNN)**
*   **🔗 Modèle d'étude (Code) :** [PyTorch Minimal CNN (CIFAR-10)](https://github.com/pytorch/tutorials/blob/main/beginner_source/blitz/cifar10_tutorial.py) - Un classifieur d'images basique.
*   **À étudier :** Le calcul de la *Loss* et de l'indice de confiance lors de l'inférence.
*   **Objectif :** Déduire comment la différence de confiance entre une image du *train set* et du *test set* permet une Inférence d'Appartenance (MIA).

**4. Séries Temporelles (Séquentiel type LSTM)**
*   **🔗 Modèle d'étude (Code) :** [TSGM (Time Series Generative Models)](https://github.com/AlexanderVNikitin/tsgm) - Contient des modèles de séries temporelles et des métriques de vulnérabilité.
*   **À étudier :** La prédiction sur des séquences numériques continues.
*   **Objectif :** Déduire mathématiquement comment utiliser l'erreur de prédiction pour affirmer qu'une série spécifique a été mémorisée.

---

## 📊 2. Structure de la Présentation (Livrable de Fin de Sprint)

1. **Introduction :** Pourquoi l'architecture dicte la surface d'attaque.
2. **<span style="color:#007BFF">Restitution Bleue :</span>** Explication de la mécanique interne du Génératif (avec appui sur les codes NanoGPT/Diffusion) et présentation des vecteurs d'attaque déduits.
3. **<span style="color:#FD7E14">Restitution Orange :</span>** Explication de la mécanique Discriminative (CNN/TSGM) et démonstration logique des attaques MIA déduites.
4. **Conclusion de l'étude :** Notre **Matrice Modèles ↔ Attaques** finale, et sélection des modèles GitHub exacts avec lesquels nous allons coder les expérimentations lors du Sprint 3.
