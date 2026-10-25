# Idées et Réflexions pour la Semaine 2 (Sprint 2)

*Notes retravaillées à l'issue de la présentation de cadrage.*

## 1. Adaptation des méthodes d'attaque
- **Concept brut :** "Prendre les attaques et faire des choix différents en fonction de la méthode d'attaque"
- **Raffinement :** La présentation a mis en évidence différentes attaques (Inférence d'appartenance / MIA, Extraction) et un modèle de menace explicite (boîte noire, blanche, grise). L'idée est de ne pas appliquer une attaque à l'aveugle, mais de choisir l'attaque la plus pertinente selon le contexte (ex: accès aux probabilités/logits vs accès au texte généré) et la modalité.

## 2. Maîtrise de l'environnement : Entraînement sur petits corpus
- **Concept brut :** "Model a transformer avec entraînement sur un petit corpus", "Modèle petit (texte, image, séries temporelles)", "Voir sur github des modèles d'entraînement"
- **Raffinement :** Plutôt que de subir l'opacité de modèles pré-entraînés massifs, la stratégie s'oriente vers des environnements de test "jouets" où l'on contrôle 100% du pipeline d'entraînement. 
- **Ouverture multi-modale :** Étendre l'étude (Texte, Image, Séries Temporelles) permet d'observer si les principes de mémorisation sont universels ou propres à une structure de données.
- **Piste technique :** Sourcer des petits projets GitHub "from scratch" simples à manipuler.

## 3. Dynamique temporelle de la mémorisation
- **Concept brut :** "un pool d'attaques variées à 20 epoch, 100, 200"
- **Raffinement :** La présentation citait le "nombre d'époques" et "l'étape d'entraînement" comme facteurs déterminants. En analysant le modèle à t=20, t=100 et t=200 époques, on pourra visualiser *quand* la mémorisation se déclenche par rapport à l'apprentissage des règles générales (et potentiellement la distinguer du surapprentissage classique sur les courbes).

## 4. Expérimentation ciblée : Injection et Hyperparamètres
- **Concept brut :** "Injecter des problèmes", "augmenter/réduire"
- **Raffinement :** 
    - *L'injection (Canaris) :* En lien direct avec l'argument de la "longue traîne" (Feldman) abordé en présentation. Injecter des "problèmes" revient à insérer volontairement des données aberrantes, rares ou des secrets dans le corpus pour vérifier si le modèle est forcé de les mémoriser.
    - *L'échelle :* "Augmenter/réduire" implique d'étudier l'impact des facteurs listés dans les slides : duplication de la donnée injectée (effet non linéaire), taille du modèle (sur-paramétrisation), ou budget de requêtes de l'attaque.

## 5. L'axe défensif (Blue Teaming)
- **Concept brut :** "des modèles de mitigation ......"
- **Raffinement :** Réponse à la question ouverte de la présentation ("Une partie défense est-elle attendue ?"). Oui, l'idée est de tester des contre-mesures (ex: Differential Privacy, déduplication, limitation des époques) sur nos petits modèles fraîchement attaqués pour évaluer la baisse du risque de fuite.

## 6. Objectif central du Sprint 2
- **Concept brut :** "Faire le sprint 2 : lier les attaques en fonction des modèles"
- **Raffinement :** L'aboutissement des réflexions actuelles est de créer une cartographie précise. Il s'agit de structurer l'approche expérimentale en associant le bon type d'attaque au bon type de modèle et de modalité, tout en définissant l'environnement de test associé.
