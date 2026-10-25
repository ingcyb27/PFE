# 🎯 Guide de Soutenance & Maîtrise Technique Approfondie — Emmanuel
### *PFE Cybersécurité & IA : Étude de la Mémorisation dans les Modèles Génératifs*
**Auteur :** Emmanuel SININI  
**Binôme :** Fatima-Zahra Rhoulam  
**Encadrants :** Maryline Laurent (TSP) $\cdot$ Faiz Taleb (TSP / EDF)  
**Support de référence :** `week2/main_week2.pdf` (28 slides)

---

## 🧭 1. Vue Stratégique & Fil Conducteur de ta Partie

### Ton Rôle dans la Présentation
Fatima-Zahra a présenté le **Périmètre Prédictif / Séquentiel (Orange)** : les CNN (classification d'images CIFAR-10) et les LSTM (séries temporelles TSGM). Elle a démontré comment le surapprentissage crée un **signal résiduel d'erreur ou de confiance** exploité par des attaques d'**inférence** (MIA, calcul de rang d'exposure), où le modèle ne restitue jamais la donnée elle-même.

**Ta mission (Périmètre Bleu) :** Faire monter la soutenance d'un cran en basculant dans le **monde génératif**. Tu traites les deux grandes architectures reines de l'état de l'art :
1. **NanoGPT (Texte) :** Architecture autorégressive discrète par Transformer causal.
2. **DDPM (Image) :** Modèle de diffusion probabiliste continu par Denoising Score Matching.

### Le Message Maître que tu portes
> *« Dans les modèles génératifs, la mémorisation n'est pas seulement un problème de confiance ou d'inférence d'appartenance : c'est un effondrement stochastique où la sortie du réseau devient la donnée d'entraînement elle-même, restituée bit par bit (texte) ou pixel par pixel (image). En comprenant la factorisation causale et la géométrie des champs de score, nous démontrons mathématiquement d'où naît la vulnérabilité et comment déduire des attaques d'extraction à l'aveugle sans aucune connaissance préalable. »*

---

## 🎤 2. Détail Slide par Slide : Pitch, Théorie Profonde & FAQ Encadrants

---

### 📍 SLIDE 13 : Anatomie de NanoGPT : La Factorisation Autorégressive

#### 🎯 Objectif Stratégique du Slide
Établir que la vulnérabilité d'un modèle de langage découle directement de sa formulation probabiliste fondamentale : la règle de décomposition causale de gauche à droite.

#### 🗣️ Script Oral (Ce que tu dis)
> *"Merci Fatima-Zahra. Nous venons de voir comment les modèles prédictifs trahissent la présence d'une donnée à travers leur score de confiance. Je vous propose maintenant d'entrer dans le cœur des modèles génératifs, en commençant par le texte avec NanoGPT.*
>
> *Le fondement d'un Transformer causal comme GPT repose sur la factorisation autorégressive : la probabilité conjointe d'un texte $P(X)$ est décomposée en un produit de probabilités conditionnelles de chaque token sachant tous les tokens passés. Dans le code officiel de Karpathy (`model.py`), cette dynamique se traduit par une boucle récursive : à chaque pas, le modèle prédit les logits du token suivant, les divise par la température $T$, applique un Softmax pour obtenir une distribution, échantillonne un token, et le concatène au contexte pour le tour suivant.*
>
> *Regardons maintenant comment cette boucle de rétroaction crée la brèche de mémorisation."*

#### 🧠 Compréhension Profonde (Ce que tu dois maîtriser)
* **Factorisation causale :** $P(X) = \prod_{t=1}^N P(x_t \mid x_{<t})$. C'est la règle de Bayes appliquée à une séquence. Le terme « causal » signifie que le token à l'instant $t$ ne dépend que du passé $x_1, \dots, x_{t-1}$.
* **Le masque causal dans `model.py` :** Dans le bloc d'attention (`CausalSelfAttention`), Karpathy définit un tenseur triangulaire inférieur `bias = torch.tril(torch.ones(T, T))`. Les valeurs au-dessus de la diagonale sont mises à $-\infty$. Quand on passe au Softmax, $\exp(-\infty) = 0$. Cela interdit physiquement au modèle de "voir" les tokens futurs pendant l'entraînement.
* **Le rôle de la Température $T$ :**
  $$P(x_{t+1} = w) = \frac{\exp(z_w / T)}{\sum_j \exp(z_j / T)}$$
  Si $T \to 0$, le Softmax devient un `argmax` pur (déterministe). Si $T \to \infty$, la distribution devient uniforme. Mais attention : si un logit $z_{\text{secret}}$ est massivement sur-ajusté (ex: $z_{\text{secret}} = 40$ alors que les autres sont à $2$), même à température $T = 1.0$, $\exp(40) \gg \sum \exp(z_j)$, rendant l'échantillonnage quasi-déterministe.

#### ❓ Anticipation des Questions d'Encadrants
* **Question piège de Maryline Laurent :** *« Pourquoi cette factorisation causale rend-elle le modèle plus vulnérable à l'extraction qu'une architecture bidirectionnelle comme BERT ? »*
  * **Ta réponse :** *« Parce que BERT résout une tâche de Masked Language Modeling (remplissage de trous au milieu d'un contexte figé) sans boucle de génération autonome. À l'inverse, dans NanoGPT, chaque token généré devient immédiatement l'entrée du calcul suivant (`torch.cat([idx, idx_next])`). Si le modèle génère le premier mot d'un secret mémorisé, ce mot conditionne le second, qui conditionne le troisième : c'est un mécanisme d'auto-amorçage séquentiel irréversible. »*

---

### 📍 SLIDE 14 : Pipeline de Décision Autorégressif (NanoGPT)

#### 🎯 Objectif Stratégique du Slide
Visualiser schématiquement la boucle fermée et faire comprendre visuellement le phénomène « boule de neige » (*snowball effect*) qui verrouille la génération sur un texte mémorisé.

#### 🗣️ Script Oral (Ce que tu dis)
> *"Voici le graphe d'exécution de la décision au sein de NanoGPT. Remarquez la flèche rouge en pointillés à gauche : c'est la boucle de rétroaction autorégressive.*
>
> *Le pipeline fonctionne en 6 étapes : on injecte un contexte initial $x_{1:t}$, qui traverse l'attention causale et les blocs MLP pour produire un vecteur de logits $z$. Ce vecteur passe par le Softmax pondéré par la température, puis un échantillonnage multinomial produit le token suivant $x_{t+1}$.*
>
> *Ce nouveau token est immédiatement concaténé au prompt initial pour réattaquer le réseau. C'est ici que réside la faille structurelle majeure : dès lors qu'un token issu d'une donnée d'entraînement possède un logit qui écrase la concurrence ($z_w \gg z_j$), il est sélectionné, réinjecté en entrée, et verrouille la prédiction suivante sur la suite exacte du texte mémorisé. La boucle s'auto-renforce de manière totalement déterministe."*

#### 🧠 Compréhension Profonde (Ce que tu dois maîtriser)
* **Effet d'avalanche (Snowball Effect) :** En génération de texte, l'entropie conditionnelle $H(x_{t+1} \mid x_{1:t})$ mesure l'incertitude du modèle. Pour un texte ordinaire, $H > 2$ nats. Mais dès qu'un préfixe rentre dans une zone mémorisée par cœur (ex: une clé d'API ou une adresse privée), l'entropie s'effondre à $H \approx 0$. Le modèle n'a plus aucun degré de liberté stochastique : il récite.
* **Top-k et Top-p (Nucleus) :** Même si l'attaquant active un filtrage Top-k (ex: $k=50$), si le vrai token a une probabilité de $0{,}999$, il fait partie du Top-1 et sera tiré dans $99{,}9\%$ des cas. Le filtrage probabiliste standard n'immunise pas contre l'extraction.

#### ❓ Anticipation des Questions d'Encadrants
* **Question de Faiz Taleb :** *« Est-ce que le sampling multinomial n'introduit pas assez d'aléa pour masquer cette mémorisation ? »*
  * **Ta réponse :** *« Non, parce que le multinomial tire selon les probabilités normalisées par le Softmax. Si la différence de logits $\Delta z = z_{\text{mémorisé}} - z_{\text{autre}} > 10$, le ratio de probabilité est $\exp(10) \approx 22\,000$. Le tirage aléatoire sélectionne le token mémorisé 21 999 fois sur 22 000. L'aléa n'est qu'apparent, la distribution est dégénérée. »*

---

### 📍 SLIDE 15 : Théorie de la Mémorisation : Perte & Longue Traîne (Feldman 2020)

#### 🎯 Objectif Stratégique du Slide
Démontrer que la mémorisation n'est pas un bogue ou une maladresse de sur-entraînement, mais une **nécessité mathématique inhérente à l'optimisation** sur des données naturelles à distribution en loi de puissance.

#### 🗣️ Script Oral (Ce que tu dis)
> *"Une question fondamentale se pose : pourquoi le modèle mémorise-t-il ? Est-ce simplement un défaut d'apprentissage ?*
>
> *La réponse est non, et elle est mathématiquement démontrée par Vitaly Feldman dans son article fondateur à STOC 2020. Lors de l'entraînement dans `train.py`, nous minimisons la perte d'entropie croisée $\mathcal{L}_{CE}$. Pour annuler cette perte sur un échantillon, le modèle doit pousser la probabilité du token cible vers 1, ce qui grave la séquence mot pour mot dans les poids.*
>
> *Mais surtout, les données de langage suivent une distribution de Zipf, illustrée sur l'histogramme de droite. Vous avez une tête avec des motifs très fréquents, où le modèle généralise en apprenant des règles de grammaire. Et vous avez une immense longue traîne, composée d'exemples rares ou uniques. Le théorème de Feldman prouve que sur de telles distributions, pour atteindre l'erreur de généralisation minimale, le réseau est mathématiquement contraint de mémoriser individuellement les exemples atypiques de la longue traîne. La mémorisation est le prix à payer pour l'optimalité prédictive."*

#### 🧠 Compréhension Profonde (Ce que tu dois maîtriser)
* **Loi de Zipf :** La fréquence d'un mot ou d'un n-gramme est inversement proportionnelle à son rang : $f(r) \propto 1/r^\alpha$ (avec $\alpha \approx 1$). Cela implique qu'environ $50\%$ de la masse totale de la distribution est concentrée dans des événements observés très peu de fois (les « singletons » et « doubletons »).
* **Le Théorème de Feldman (STOC 2020) :** Si la distribution des données contient une longue traîne de sous-populations rares (chacune ayant une probabilité $< \epsilon$), aucun algorithme d'apprentissage ne peut atteindre l'erreur Bayesienne minimale sans mémoriser les labels de ces sous-populations rares. Si le modèle essayait de moyenner ou de généraliser sur ces cas rares, son erreur globale augmenterait.
* **Conséquence sécurité :** Les secrets (numéros de sécurité sociale, dossiers médicaux, clés privées, citations uniques) se situent précisément dans cette longue traîne. Ils sont donc les premières victimes de la mémorisation requise par l'optimisation.

#### ❓ Anticipation des Questions d'Encadrants
* **Question de Maryline Laurent :** *« Est-ce qu'on ne pourrait pas simplement régulariser avec du weight decay (L2) ou du dropout pour empêcher cette mémorisation ? »*
  * **Ta réponse :** *« Une régularisation excessive (fort dropout ou fort weight decay) dégrade immédiatement la performance du modèle sur les tâches complexes (benchmark de perplexité). C'est le dilemme central de la vie privée en IA : la confidentialité parfaite nuit à la précision sur les cas rares. C'est pour cela que la mémorisation persiste dans les modèles commerciaux comme GPT-4. »*

---

### 📍 SLIDE 16 : Attaques Déduites : Prefix-Prompting vs Extraction Aveugle

#### 🎯 Objectif Stratégique du Slide
Classifier les vecteurs d'attaque sur LLM selon les capacités de l'attaquant et introduire le défi colossal de l'extraction à l'aveugle.

#### 🗣️ Script Oral (Ce que tu dis)
> *"À partir de cette compréhension architecturale, nous déduisons deux familles d'attaques offensives.*
>
> *La première, c'est l'extraction par préfixe (*Prefix-Prompting*). Si l'attaquant connaît déjà le début d'un secret, par exemple « Le numéro fiscal du client X est », il fournit ce préfixe au modèle. Grâce à l'attention causale, le modèle complète déterministement le suffixe exact mémorisé, même avec une température normale.*
>
> *Mais en cybersécurité, le scénario le plus redoutable et le plus réaliste est le second : l'extraction à l'aveugle, formalisée par Nicholas Carlini en 2021. Ici, l'attaquant n'a aucun préfixe. Il est en boîte noire complète. Il envoie un prompt vide ou aléatoire et génère des centaines de milliers de textes. Le verrou scientifique majeur devient alors : parmi 200 000 textes générés, comment isoler automatiquement les données privées mémorisées d'une hallucination banale mais bien formulée ? Voyons la réponse de Carlini au slide suivant."*

#### 🧠 Compréhension Profonde (Ce que tu dois maîtriser)
* **Définition formelle de l'extraction par préfixe :** L'attaquant dispose d'un couple $(p, s)$ dans les données d'entraînement. En fournissant le préfixe $p$, le modèle génère $\hat{s} = \arg\max P(s \mid p)$. Si $\hat{s} == s$, l'extraction est réussie.
* **Le défi de l'extraction à l'aveugle (Blind Extraction) :** Si on génère 200 000 phrases, la plupart seront des phrases grammaticalement parfaites générées par extrapolation (ex: « The weather today is sunny »). Comment une machine peut-elle savoir si « John Doe lives at 12 Elm Street » est une vraie personne tirée du dataset d'entraînement ou une simple fiction inventée par le modèle ? Ce problème exige une métrique discriminante.

---

### 📍 SLIDE 17 : Workflow Officiel d'Extraction Aveugle (Carlini et al., USENIX 2021)

#### 🎯 Objectif Stratégique du Slide
Présenter la démarche méthodologique officielle du papier d'extraction de Carlini et montrer que nous maîtrisons la chaîne expérimentale complète.

#### 🗣️ Script Oral (Ce que tu dis)
> *"Voici le workflow officiel publié par Carlini et son équipe à USENIX Security 2021, sur le modèle GPT-2 XL de 1,5 milliard de paramètres. La démarche est scindée en deux phases distinctes.*
>
> *La Phase 1 est l'Attaque en boîte noire : l'attaquant génère 200 000 séquences en conditionnant le modèle sur des préfixes vides ou aléatoires. Il supprime les doublons triviaux, puis ordonne l'ensemble des textes selon 6 métriques de score concurrentes pour sélectionner les 100 meilleurs candidats de chaque métrique.*
>
> *La Phase 2 est l'Évaluation : ces 100 candidats sont inspectés. L'équipe a d'abord effectué des recherches manuelles et automatisées sur Internet pour vérifier la présence du texte, puis OpenAI a vérifié directement dans le jeu de données d'entraînement WebText privé. Résultat : plus de 600 secrets authentiques ont été extraits mot pour mot (noms, adresses, codes source, données confidentielles).*
>
> *Mais quelle est la métrique qui a permis ce tri ? C'est l'objet du slide suivant."*

#### 🧠 Compréhension Profonde (Ce que tu dois maîtriser)
* **GPT-2 XL :** 1,5 milliard de paramètres, entraîné sur WebText (40 Go de texte issu de liens Reddit ayant au moins 3 upvotes).
* **Les 6 métriques testées par Carlini :**
  1. Perplexité brute du modèle cible $\text{PPL}(x)$.
  2. Ratio de perplexité avec un modèle plus petit (GPT-2 Small).
  3. Ratio de perplexité avec un modèle moyen (GPT-2 Medium).
  4. Ratio de compression zlib : $\text{PPL}(x) / \text{zlib}(x)$.
  5. Perplexité conditionnée sur texte minuscule.
  6. Perplexité glissante (sliding window).
* **Pourquoi la déduplication est essentielle :** Le modèle a tendance à générer plusieurs fois les mêmes fragments communs (ex: des licences open-source MIT ou Apache). Si on ne déduplique pas, le Top 100 sera saturé par 100 copies de la licence MIT, masquant les secrets uniques.

---

### 📍 SLIDE 18 : Pourquoi la Perplexité Seule Échoue : Le Filtrage zlib

#### 🎯 Objectif Stratégique du Slide
Expliquer l'intuition géniale de Carlini : pourquoi la perplexité brute échoue et comment l'entropie de compression zlib agit comme un calibrateur sans apprentissage.

#### 🗣️ Script Oral (Ce que tu dis)
> *"Cette figure est centrale dans notre étude : elle montre pourquoi la perplexité brute ne suffit pas et comment le ratio zlib résout le problème.*
>
> *En abscisse, vous avez la perplexité calculée par GPT-2 (plus elle est basse vers la gauche, plus le modèle est confiant). En ordonnée, vous avez l'entropie de compression zlib (plus elle est haute vers le haut, plus le texte est complexe et incompressible).*
>
> *Le nuage de points gris représente les 200 000 textes générés. Si nous utilisions uniquement la perplexité brute en sélectionnant les points les plus à gauche, nous n'extrairions que des séquences triviales : des répétitions infinies comme « 1 1 1 1 » ou « the the the ». Pourquoi ? Parce que le modèle ne commet aucune erreur sur une répétition, sa perte est minimale ! Mais zlib compresse ces répétitions en 2 octets, leur entropie zlib est nulle.*
>
> *À l'inverse, regardez les points bleus : ce sont les 59 secrets réels confirmés extraits de WebText. Ils ont une caractéristique unique : une perplexité GPT-2 anormalement basse, mais une complexité zlib très élevée ! En divisant la perplexité par la taille compressée zlib, on isole exactement cette zone en haut à gauche (les points rouges du Top 100). zlib sert de modèle de référence universel sans entraînement."*

#### 🧠 Compréhension Profonde (Ce que tu dois maîtriser)
* **Formule mathématique de la Perplexité :**
  $$\text{PPL}(X) = \exp\left( - \frac{1}{N} \sum_{i=1}^N \log P(x_i \mid x_{<i}) \right) = \exp(\mathcal{L}_{CE})$$
  C'est l'exponentielle de l'entropie croisée moyenne par token. Elle représente intuitivement le nombre de choix équiprobables entre lesquels le modèle hésite à chaque mot. Si $\text{PPL} = 1$, il est sûr à $100\%$.
* **Pourquoi zlib ?** L'algorithme DEFLATE de zlib combine la compression LZ77 (détection de motifs répétés) et le codage de Huffman (affectation de bits selon la fréquence). Une séquence triviale comme `a a a a a` a une taille compressée de quelques octets. Une phrase naturelle complexe contenant des noms propres ou des chiffres (`John Doe, SSN 123-45-6789`) est quasiment incompressible par zlib.
* **Le Ratio :** $\text{Ratio} = \frac{\text{PPL}_{\text{Target}}(x)}{\text{Compression}_{\text{zlib}}(x)}$. Ce ratio pénalise lourdement les textes compressibles et favorise les textes complexes que seul le modèle trouve faciles !

#### ❓ Anticipation des Questions d'Encadrants
* **Question de Faiz Taleb :** *« Pourquoi utiliser zlib plutôt qu'un autre modèle de langage plus petit comme référence ? »*
  * **Ta réponse :** *« Carlini a testé les deux ! Utiliser un modèle plus petit (ex: GPT-2 Small) fonctionne très bien, mais cela demande d'entraîner et d'héberger un deuxième LLM en mémoire, avec un temps de calcul important. L'avantage magistral de zlib, c'est qu'il est instantané, sans apprentissage, standardisé sur n'importe quelle machine et totalement indépendant de tout biais d'entraînement. »*

---

### 📍 SLIDE 19 : Inférence d'Appartenance Textuelle (MIA)

#### 🎯 Objectif Stratégique du Slide
Faire le pont entre l'extraction complète (verbatim) et l'inférence d'appartenance (MIA) sur le texte, en montrant la version textuelle du travail de Fatima-Zahra.

#### 🗣️ Script Oral (Ce que tu dis)
> *"Pour clore le volet texte, regardons l'Inférence d'Appartenance (MIA). Contrairement à l'extraction aveugle où l'attaquant cherche à régénérer le secret, la MIA cherche à répondre à une question binaire : est-ce que ce texte $x$ précis faisait partie du jeu d'entraînement ?*
>
> *Le test classique repose sur le seuil de perte : si l'entropie croisée $\mathcal{L}_{CE}(x)$ est inférieure à un seuil $\tau$, on prédit que $x$ est membre. Cependant, sur le texte, ce seuil naïf souffre d'un biais majeur : une phrase comme « La terre tourne autour du soleil » aura une perte faible sur n'importe quel modèle, qu'elle soit dans le train set ou non, simplement parce qu'elle est universellement probable.*
>
> *Pour corriger cela, Carlini et al. utilisent le ratio de référence : on compare la perte du modèle cible à la perte d'un modèle de référence neutre entraîné sur des données disjointes. Si le modèle cible a une perte drastiquement plus faible que le modèle de référence sur $x$, c'est la preuve indiscutable d'une mémorisation spécifique et non d'une facilité intrinsèque du texte."*

#### 🧠 Compréhension Profonde (Ce que tu dois maîtriser)
* **Décision MIA par ratio :** $\text{Ratio}_{\text{MIA}}(x) = \frac{\mathcal{L}(x; \theta_{\text{cible}})}{\mathcal{L}(x; \theta_{\text{ref}})}$. Si ce ratio est $\ll 1$, le texte est classé Membre (IN).
* **Biais de difficulté intrinsèque :** Certains textes ont une haute entropie naturelle (ex: jargon médical, code source obscur). Sans modèle de référence, ces textes auraient une perte élevée et ne seraient jamais détectés comme membres, même s'ils ont été mémorisés ! Le modèle de référence normalise par la complexité intrinsèque de la phrase.

---

### 📍 SLIDE 21 : Le Paradigme de Diffusion : Bruitage & Débruitage

#### 🎯 Objectif Stratégique du Slide
Introduire le deuxième grand pilier génératif : les modèles de diffusion d'images (DDPM), en posant rigoureusement les deux processus stochastiques jumeaux (Forward et Reverse).

#### 🗣️ Script Oral (Ce que tu dis)
> *"Nous quittons maintenant le domaine discret du texte pour aborder les données continues avec les modèles de diffusion probabiliste (DDPM), popularisés par Ho et al. en 2020.*
>
> *Le paradigme repose sur deux processus stochastiques jumeaux, illustrés sur le schéma officiel :*
> * *À gauche, le Processus Direct (Forward $q$) : on part d'une image réelle $x_0$, et on lui ajoute progressivement un bruit gaussien sur $T$ pas de temps (souvent $T = 1000$). La propriété remarquable est que cette dégradation est analytique : grâce à la variable $\bar{\alpha}_t$, on peut calculer directement l'état bruité $x_t$ à n'importe quel instant $t$ sans calculer les étapes intermédiaires.*
> * *À droite, le Processus Inverse (Reverse $p_\theta$) : c'est le processus génératif. On part d'un pur bruit gaussien $x_T$, et on effectue 1000 étapes de débruitage itératif. À chaque pas, un réseau de neurones U-Net $\epsilon_\theta$ prédit le bruit exact à soustraire pour remonter vers l'image nette.*
>
> *Voyons maintenant les mathématiques exactes qui régissent ce débruitage."*

#### 🧠 Compréhension Profonde (Ce que tu dois maîtriser)
* **Processus Direct (Forward) :**
  $$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$$
  En posant $\alpha_t = 1 - \beta_t$ et $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$, la récurrence se résout sous forme fermée :
  $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
  C'est fondamental : à $t=0$, $\bar{\alpha}_0 \approx 1 \implies x_0$. À $t=T$, $\bar{\alpha}_T \approx 0 \implies x_T \approx \epsilon$ (bruit blanc pur).
* **Processus Inverse (Reverse) :**
  $$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$
  La moyenne $\mu_\theta$ s'exprime directement en fonction du bruit prédit par le réseau U-Net :
  $$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

---

### 📍 SLIDE 22 : Fondement Mathématique : Denoising Score Matching

#### 🎯 Objectif Stratégique du Slide
Élever le niveau mathématique de la présentation en faisant le lien formel entre la perte d'entraînement de Ho et la théorie des équations différentielles stochastiques (SDE) de Yang Song (ICLR 2021).

#### 🗣️ Script Oral (Ce que tu dis)
> *"Quelle est la fonction d'objectif minimisée par ce U-Net ? Dans le code de référence (`p_losses`), Ho et al. utilisent une fonction de perte simplifiée $\mathcal{L}_{\text{simple}}$ : l'erreur quadratique moyenne entre le bruit gaussien injecté $\epsilon$ et le bruit prédit par le U-Net $\epsilon_\theta(x_t, t)$.*
>
> *Mais quel est le sens géométrique profond de cette équation ? C'est la contribution majeure de Song et al. à ICLR 2021 : le débruitage est en réalité une estimation du champ de score !*
>
> *Comme le montre l'équation en bas à gauche, prédire le bruit $\epsilon_\theta$ équivaut exactement à calculer le gradient spatial de la log-densité de probabilité des données : $\nabla_{x_t} \log p_t(x_t)$. Physiquement, cela signifie que le U-Net n'est pas un classifieur : c'est une boussole vectorielle. À chaque pas temporel et en tout point de l'espace, il pousse l'image bruitée le long du champ de gradient vers les régions de plus haute probabilité des données réelles.*
>
> *Et c'est précisément ce champ de force qui engendre la mémorisation."*

#### 🧠 Compréhension Profonde (Ce que tu dois maîtriser)
* **Score-Based Modeling :** Le « score » d'une distribution de probabilité continue $p(x)$ est défini comme le gradient spatial de son logarithme : $S(x) = \nabla_x \log p(x)$. Il pointe toujours vers les maxima locaux de probabilité (les modes).
* **Denoising Score Matching (Vincent 2011, Song 2021) :** Pour une perturbation gaussienne $q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$, le gradient théorique exact est :
  $$\nabla_{x_t} \log q(x_t \mid x_0) = - \frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{1 - \bar{\alpha}_t} = - \frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$$
  En apprenant $\epsilon_\theta(x_t, t) \approx \epsilon$, le réseau apprend donc directement :
  $$\epsilon_\theta(x_t, t) = - \sqrt{1 - \bar{\alpha}_t} \nabla_{x_t} \log p_t(x_t)$$
* **Lien avec la physique :** L'intégration de $x_T$ vers $x_0$ est analogue au mouvement sur-amorti d'une particule dans un paysage de potentiel gravitationnel ou énergétique.

#### ❓ Anticipation des Questions d'Encadrants
* **Question pointue de Maryline Laurent :** *« Pourquoi parle-t-on de Score Matching plutôt que d'apprentissage par Maximum de Vraisemblance classique ? »*
  * **Ta réponse :** *« Parce que calculer la vraie vraisemblance $p(x) = \frac{\tilde{p}(x)}{Z}$ sur des images de haute dimension est intraitable à cause de la constante de normalisation $Z = \int \tilde{p}(x) dx$. En prenant le gradient du logarithme $\nabla_x \log p(x) = \nabla_x \log \tilde{p}(x) - \nabla_x \log Z$, la constante $Z$ disparaît car elle ne dépend pas de $x$ ! Le Score Matching contourne l'impossibilité de calculer la constante de normalisation. »*

---

### 📍 SLIDE 23 : La Mécanique de Mémorisation : Les Bassins d'Attraction

#### 🎯 Objectif Stratégique du Slide
Révéler la mécanique physique de la mémorisation visuelle : comment le surapprentissage déforme le champ de gradient en creusant des puits de potentiel gravitationnels appelés **bassins d'attraction**.

#### 🗣️ Script Oral (Ce que tu dis)
> *"Voici la clé de voûte de la mémorisation dans les modèles de diffusion, formalisée par Carlini et Somepalli en 2023.*
>
> *Dans un modèle sain qui généralise, deux bruits gaussiens initiaux distincts suivent des lignes de champ différentes et convergent vers deux images différentes : le modèle crée de la diversité.*
>
> *Mais que se passe-t-il si une image d'entraînement $x_0^*$ est répétée plusieurs fois dans la base, ou si le modèle est surentraîné ? Le gradient $\nabla \log p(x)$ devient colossal autour de cette image. Le réseau creuse ce que la physique appelle un **bassin d'attraction**, représenté par l'ellipse bordeaux sur mon schéma.*
>
> *Ce bassin agit comme un véritable trou noir : regardez les trajectoires en pointillés bleus. Des bruits initiaux $x_T^{(1)}, x_T^{(2)}, x_T^{(3)}$ totalement indépendants et tirés au hasard sont aspirés vers le même minimum local et s'effondrent sur l'image réelle $x_0^*$. Plusieurs graines différentes génèrent la même image d'entraînement au pixel près. C'est l'effondrement des trajectoires."*

#### 🧠 Compréhension Profonde (Ce que tu dois maîtriser)
* **Définition mathématique du bassin d'attraction :** Le bassin d'attraction $\mathcal{B}(x_0^*)$ d'un point $x_0^*$ est l'ensemble de tous les bruits $x_T \in \mathbb{R}^D$ tels que la trajectoire inverse discrétisée $p_\theta$ partant de $x_T$ converge dans un voisinage $\epsilon$ de $x_0^*$ :
  $$\mathcal{B}(x_0^*) = \{ x_T \sim \mathcal{N}(0, I) \mid \| \text{Reverse}(x_T; \theta) - x_0^* \| < \epsilon \}$$
* **Volume stochastique :** Dans un espace à $D = 12\,288$ dimensions (images $64 \times 64 \times 3$), le volume total de la sphère gaussienne est immense. Pour une image non mémorisée, son bassin d'attraction a un volume quasi-nul. Pour une image dupliquée ou sur-apprise, le volume de $\mathcal{B}(x_0^*)$ devient non négligeable (ex: $0{,}1\%$ ou $1\%$ de l'espace total).
* **Conséquence :** Si le volume est de $1\%$, en tirant seulement 100 bruits au hasard, vous avez une probabilité élevée de tomber dans le bassin et de régénérer l'image privée !

---

### 📍 SLIDE 24 : Déduction des Attaques : Détection de Cliques Multi-Graines

#### 🎯 Objectif Stratégique du Slide
Déduire l'attaque d'extraction aveugle sur modèle de diffusion (Carlini USENIX 2023) à partir de la propriété géométrique des bassins d'attraction, et expliquer le rôle amplificateur du Classifier-Free Guidance (CFG).

#### 🗣️ Script Oral (Ce que tu dis)
> *"Comment transformer cette propriété physique en une attaque cyber concrète sans aucune connaissance préalable ? C'est l'attaque par détection de cliques multi-graines de Carlini (USENIX 2023).*
>
> *L'intuition probabiliste est imparable : en dimension 12 288 (une simple image 64x64), la probabilité que deux bruits gaussiens tirés aléatoirement génèrent spontanément deux images identiques par pur hasard est inférieure à $10^{-100}$. C'est rigoureusement impossible dans l'histoire de l'univers.*
>
> *Par conséquent, le protocole d'attaque est simple :*
> 1. *L'attaquant tire $M$ bruits gaussiens $x_T^{(i)}$ et génère $M$ images.*
> 2. *Il calcule les distances paires à paires en MSE ou SSIM.*
> 3. *Si deux images ont une distance inférieure à un seuil $\delta$, elles forment une **clique** : elles proviennent obligatoirement du même bassin d'attraction, ce qui prouve mathématiquement qu'il s'agit d'une image d'entraînement mémorisée !*
>
> *Enfin, regardez le bloc de droite : le Classifier-Free Guidance (CFG). En augmentant le scale de guidage $s > 1$, on amplifie artificiellement la force du gradient vers les modes conditionnels. Cela élargit la taille des bassins d'attraction et multiplie massivement le taux de succès de l'extraction."*

#### 🧠 Compréhension Profonde (Ce que tu dois maîtriser)
* **Formule du Classifier-Free Guidance (CFG) (Ho & Salimans 2021) :**
  $$\hat{\epsilon} = \epsilon_\theta(x_t, \emptyset) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$
  Où $\emptyset$ est le conditionnement nul (inconditionnel) et $c$ est le prompt textuel ou la classe. Le terme $s$ est l'échelle de guidage (*guidance scale*).
  * Si $s = 1$, guidage standard.
  * Si $s > 1$ (couramment $s = 7{,}5$ sur Stable Diffusion), on extrapole dans la direction du gradient conditionnel en s'éloignant du mode inconditionnel.
  * **Effet sur la mémorisation :** Le CFG sur-accentue les détails caractéristiques de la classe $c$. Si une image de la classe $c$ a été vue plusieurs fois pendant le training, le CFG transforme le bassin d'attraction en gouffre géant : l'extraction devient triviale.
* **Métrique SSIM (Structural Similarity Index) :** Combine la luminance, le contraste et la structure locale. Deux images peuvent différer de quelques pixels de bruit tout en ayant une SSIM de $0{,}95$, confirmant qu'il s'agit de la même scène mémorisée.

#### ❓ Anticipation des Questions d'Encadrants
* **Question de Faiz Taleb :** *« Quel est le coût de calcul pour l'attaquant ? Calculer les distances entre $M$ images demande $O(M^2)$ comparaisons, est-ce praticable ? »*
  * **Ta réponse :** *« C'est une excellente question. Si $M = 50\,000$, $M^2 \approx 2{,}5 \times 10^9$ comparaisons. Mais pour éviter de comparer les images pixel par pixel, Carlini utilise une étape de pré-filtrage : on calcule l'embedding de chaque image avec un réseau pré-entraîné (comme CLIP) ou un hachage perceptuel, et on fait une recherche de plus proches voisins par arbre KD ou FAISS en $O(M \log M)$. Le calcul complet ne prend que quelques minutes sur un GPU standard. »*

---

### 📍 SLIDE 27 : Matrice Croisée : Modèles $\leftrightarrow$ Logique $\leftrightarrow$ Attaques (Emmanuel)

#### 🎯 Objectif Stratégique du Slide
Fournir ta synthèse individuelle du Périmètre Bleu en confrontant NanoGPT et DDPM, et formuler le constat transversal majeur de sécurité sur l'effondrement de distribution.

#### 🗣️ Script Oral (Ce que tu dis)
> *"Pour conclure mon périmètre, voici la matrice croisée qui synthétise nos découvertes sur les modèles génératifs.*
>
> *À première vue, NanoGPT et DDPM n'ont rien en commun : le premier opère sur des tokens discrets par attention causale et cross-entropy, tandis que le second opère sur un espace continu de pixels par intégration de champ de score et diffusion.*
>
> *Pourtant, notre analyse de sécurité démontre qu'ils souffrent de la même pathologie sous-jacente :*
> * *Côté NanoGPT, la mémorisation naît de l'écrasement des logits sur la longue traîne de Feldman, et s'exploite par le filtrage zlib.*
> * *Côté DDPM, elle naît du surapprentissage du champ de score créant des bassins d'attraction géants, et s'exploite par la détection de collisions de bruits (cliques).*
>
> *Le constat transversal est éclatant : dans les deux cas, le surapprentissage produit **l'effondrement de la distribution vers des modes mémorisés déterministes**. La génération cesse d'être une création stochastique pour devenir une restitution de données privées."*

#### 🧠 Compréhension Profonde (Ce que tu dois maîtriser)
* Maîtrise parfaite des 4 colonnes du tableau : Architecture, Dynamique Générative, Cause de Mémorisation, Attaque Déduite.
* Être capable d'expliquer le parallèle conceptuel :
  * Logit écrasant ($z_w \gg z_j$) $\longleftrightarrow$ Puits de potentiel gravitationnel ($\nabla \log p \to \infty$).
  * Ratio PPL/zlib $\longleftrightarrow$ Seuil de distance SSIM/MSE entre graines.

---

### 📍 SLIDE 28 : L'Argument Transversal Majeur (Co-présenté avec Fatima-Zahra)

#### 🎯 Objectif Stratégique du Slide
Porter l'argument suprême de toute la réunion de Sprint 2 : la distinction conceptuelle et opérationnelle entre les modèles prédictifs (inférence) et les modèles génératifs (extraction verbatim).

#### 🗣️ Script Oral (Ce que tu dis pour ta partie)
> *(Fatima-Zahra prend d'abord la parole pour résumer le côté prédictif, puis tu enchaînes :)*
>
> *"Et comme l'a résumé Fatima-Zahra, dans les modèles prédictifs et séquentiels, le réseau ne produit qu'une étiquette ou une valeur scalaire : l'attaque se limite donc à de l'inférence indirecte d'appartenance.*
>
> *Mais dès que l'on bascule sur notre périmètre génératif — texte ou image —, la menace change radicalement de nature : la sortie du réseau **peut être l'échantillon d'entraînement lui-même**, restitué à l'identique. Nous passons d'un risque d'inférence statistique à une violation directe et massive de confidentialité et de propriété intellectuelle.*
>
> *C'est sur ce constat que se conclut notre Sprint 2. Nous vous remercions pour votre attention et sommes prêts pour vos questions."*

---

## 🛡️ 3. Guide de Répartie : Les Questions Redoutables des Encadrants

| Encadrant | Thématique | Question Potentielle | Ta Réponse Idéale |
|---|---|---|---|
| **Maryline Laurent** | Définitions & Vie Privée | *« Pourquoi l'extraction aveugle de Carlini viole-t-elle le RGPD si l'attaquant ne ciblait personne en particulier ? »* | *« Parce que le RGPD protège les données à caractère personnel (PII), quelle que soit l'intention de l'attaquant. Si un modèle mis à disposition du public est capable de recracher spontanément des adresses e-mails ou des dossiers médicaux réels lors de simples requêtes aléatoires, le fournisseur du modèle est légalement coupable de fuite de données non maîtrisée. »* |
| **Faiz Taleb** | Rôle industriel EDF | *« Dans un contexte industriel comme EDF, quel type de données nos modèles risquent-ils d'extraire ? »* | *« Sur des modèles de séries temporelles ou des LLM internes, ce sont des relevés de consommation confidentiels de sites sensibles, des plans d'architecture réseau ou des fragments de code source d'infrastructure critique injectés par inadvertance dans les corpus d'entraînement sans anonymisation préalable. »* |
| **Maryline Laurent** | Défenses & Mitigation | *« La Differential Privacy (DP-SGD) ne règle-t-elle pas définitivement ce problème ? »* | *« En théorie mathématique, oui : DP-SGD garantit une borne $\epsilon$ stricte sur la fuite d'un exemple individuel. Mais en pratique industrielle sur les LLM et les modèles de diffusion, DP-SGD détruit la qualité de génération (baisse drastique du score FID sur les images et hausse de la perplexité) et multiplie le temps d'entraînement par 2 ou 3. Les modèles actuels sont donc presque tous entraînés sans DP. »* |

---

## 📋 4. Checklist des Références Clés à Citer Naturellement
Quand tu parles, glisse naturellement ces 4 noms fondateurs pour asseoir ton autorité scientifique :
1. **Vitaly Feldman (STOC 2020) :** *« Does Learning Require Memorization? »* $\to$ Pour la longue traîne et la distribution de Zipf.
2. **Nicholas Carlini et al. (USENIX Security 2021) :** *« Extracting Training Data from Large Language Models »* $\to$ Pour le ratio perplexité / compression zlib sur GPT-2 XL.
3. **Jonathan Ho et al. (NeurIPS 2020) :** *« Denoising Diffusion Probabilistic Models »* $\to$ Pour l'équation simplifiée du débruitage $\mathcal{L}_{\text{simple}}$.
4. **Yang Song et al. (ICLR 2021) :** *« Score-Based Generative Modeling through Stochastic Differential Equations »* $\to$ Pour la preuve que le U-Net estime le gradient spatial de la log-densité.
