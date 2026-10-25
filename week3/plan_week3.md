# Feuille de Route — Sprint 3 : Le Passage de la Théorie à la Pratique
## Sécurité de l'IA Générative : Validation Empirique des Mécanismes de Mémorisation

**Étudiants :** Emmanuel SININI & Fatima-Zahra Rhoulam  
**Encadrants :** Maryline Laurent (Télécom SudParis) $\cdot$ Faiz Taleb (TSP / EDF)  
**Environnement de calcul :** Serveur TSP `rst-goose` (2 × NVIDIA L40S 48 Go)  
**Période :** Semaine 3  

---

## 🎯 1. Philosophie du Sprint : Valider sans se noyer dans la technique

En Semaines 1 et 2, nous avons posé les fondations théoriques :
- La factorisation autorégressive causale et le calcul des logits.
- Le Théorème de Feldman (nécessité de la mémorisation sur la longue traîne).
- Les mécanismes d'extraction (Prefix-Prompting, ratios de perplexité relative / zlib, métrique d'exposure).
- Les remarques de la réunion Week 2 : besoin d'une **vérité terrain indiscutable**, d'un **sujet ciblé**, et d'un **protocole rigoureusement comparable à la littérature**.

> **Principe directeur de la Semaine 3 :**  
> *Ne pas chercher une complexité technique prématurée, mais construire une chaîne expérimentale minimale, maîtrisée de bout en bout, qui transforme nos équations théoriques en signaux observables et quantifiables.*

---

## 🔬 2. Matrice de Correspondance : De l'Équation au Signal Pratique

Chaque expérience de cette semaine est directement reliée à un concept présenté aux encadrants :

| Concept Théorique (Week 2) | Référence Littérature | Traduction Pratique (Week 3) | Ce que l'on observe / mesure |
| :--- | :--- | :--- | :--- |
| **Effondrement des Logits** ($z_{\text{cible}} \gg z_w$) | Karpathy (NanoGPT) | **Attaque par Préfixe** (*Prefix-Prompting*) | Le modèle complète-t-il mot à mot le secret dès qu'on lui donne son amorce ? |
| **Rareté & Longue Traîne** (Loi de Zipf) | Feldman (STOC 2020) | **Fréquence d'injection variable** ($k \in \{1, 5, 10, 50\}$) | À partir de combien d'occurrences ($k$) le secret bascule-t-il de la généralisation vers la mémorisation *verbatim* ? |
| **Biais de la Perplexité Brute** | Carlini et al. (USENIX 2021) | **Calcul du Ratio $\text{PPL} / \text{zlib}$** | Les vrais secrets mémorisés se détachent-ils des répétitions triviales (faible PPL, forte entropie zlib) ? |
| **Point de bascule Overfitting** | Yeom (2018) / Carlini (2019) | **Sauvegarde par Checkpoints** (Époques 1, 5, 10, 20, 50) | À quelle époque exacte apparaît la mémorisation par rapport au décrochage de la perte de validation ? |

---

## 🛠️ 3. Architecture Expérimentale : Le Protocole Canari Contrôlé

Pour répondre à l'exigence de comparabilité soulevée par Faiz Taleb et Maryline Laurent (*« Pour comparer, il faut les mêmes conditions »*), nous adoptons le protocole standard du **Secret Sharer** (Carlini et al., 2019) :

```
[Corpus Textuel Spécialisé] + [Générateur de Canaris (k = 1, 5, 10, 50)]
                    │
                    ▼
          [Entraînement Contrôlé] ────> [Checkpoints par Époque (1, 5, 10, 20, 50)]
                    │
                    ▼
          [Sondes Offensives]
          ├─ 1. Prefix-Prompting (Restitution verbatim)
          └─ 2. Ratio PPL / zlib (Détection aveugle)
                    │
                    ▼
      [Courbes & Résultats pour la Réunion Week 3]
```

### Pourquoi ce choix est le plus pertinent ?
1. **Vérité Terrain Absolue :** Nous savons exactement ce qui a été injecté, où, et combien de fois. Aucune ambiguïté sur ce qui constitue une « fuite ».
2. **Léger et reproductible :** Pas besoin de crawler le web pendant des jours. Un jeu de données ciblé (quelques mégaoctets) suffit à observer le phénomène.
3. **Exploitation raisonnée du serveur `rst-goose` :** Les GPU L40S permettent d'exécuter l'entraînement et l'évaluation en quelques minutes, sans contrainte de mémoire.

---

## 📅 4. Découpage Opérationnel de la Semaine 3

### Jalon 1 : Socle Technique & Connexion Serveur (Jour 1)
- [ ] Tester la connexion SSH à `rst-goose.telecom-sudparis.eu`.
- [ ] Créer l'environnement Conda de travail dans `$HOME` :
  - Python 3.10
  - PyTorch (avec support CUDA pour L40S)
  - `transformers`, `datasets`, `accelerate`
  - Utilitaires de compression (`zlib`)
- [ ] Script de test rapide : vérifier la détection GPU (`torch.cuda.is_available()`) et l'allocation mémoire.

### Jalon 2 : Conception du Banc de Données et des Canaris (Jour 2 - 3)
- [ ] **Sélection du corpus textuel :** Choisir un corpus spécialisé (ex: documentation technique, rapports scientifiques, ou corpus semi-structuré).
- [ ] **Conception des Canaris synthétiques :**
  - Format réaliste : fausses clés d'API, faux identifiants, phrases sensibles d'entreprise.
  - Exemple : `« La clé d'accès interne du serveur confidentiel est : 9f8a-44bc-21d0 »`.
- [ ] **Génération du dataset augmenté :**
  - Canaris injectés 1 seule fois (rareté maximale).
  - Canaris injectés 5 fois.
  - Canaris injectés 10 fois.
  - Canaris injectés 50 fois (sur-représentation artificielle).

### Jalon 3 : Entraînement et Sauvegarde des Checkpoints (Jour 3 - 4)
- [ ] **Modèle sélectionné :** `EleutherAI/pythia-160m` (162M paramètres, standard de recherche en mémorisation) ou `gpt2` (124M paramètres, baseline de Carlini 2021).
- [ ] Sauvegarder les poids à intervalles réguliers (checkpoints aux époques 1, 5, 10, 20, 50).
- [ ] Tracer les deux courbes fondamentales :
  - $\mathcal{L}_{\text{train}}$ (Loss d'entraînement)
  - $\mathcal{L}_{\text{val}}$ (Loss de validation) pour visualiser le moment exact où le surapprentissage débute.

### Jalon 4 : Implémentation des Deux Sondes Offensives (Jour 4 - 5)
- [ ] **Sonde 1 — Extraction par Préfixe (*Prefix-Prompting*) :**
  - Fournir le préfixe du canari (ex: `« La clé d'accès interne du serveur confidentiel est : »`).
  - Mesurer le taux de restitution exacte (Accuracy / Levenshtein) selon $k$ et l'époque.
- [ ] **Sonde 2 — Extraction par Perplexité Relative ($\text{PPL} / \text{zlib}$) :**
  - Calculer la perplexité du texte généré via le modèle.
  - Calculer la taille compressée via `zlib.compress(texte)`.
  - Vérifier si les canaris se séparent nettement du texte normal sur un nuage de points.

---

## 📊 5. Ce que l'on présentera à la prochaine réunion (Livrables Sprint 3)

À la fin de la semaine, nous aurons des résultats concrets et visuels à montrer à Maryline Laurent et Faiz Taleb :
1. **Le graphique clé :** Évolution du taux de mémorisation verbatim en fonction de la fréquence $k$ ($1, 5, 10, 50$).
2. **Le lien avec le surapprentissage :** Le canari est-il mémorisé avant ou après le décrochage de la courbe de validation ? (Lien direct avec le Théorème de Feldman).
3. **Le nuage de points PPL vs zlib :** Visualisation empirique prouvant la pertinence du filtre de Carlini.
