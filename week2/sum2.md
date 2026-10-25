# Fiche de préparation — Génération textuelle (NanoGPT) et modèles de diffusion (DDPM)

**Source :** présentation *Sécurité de l’IA générative — Étude de la mémorisation*, sprint 2.  
**Périmètre :** uniquement les deux parties « Génération textuelle (NanoGPT) » et « Modèles de diffusion (DDPM) » — diapositives 13 à 19 et 21 à 24. Les diapositives de plan ne sont pas détaillées.

## Fil conducteur à retenir

Ces deux familles de modèles génèrent des données, mais pas de la même manière :

- **NanoGPT (texte)** prédit le prochain token à partir des tokens précédents, puis recommence. C’est une génération **autorégressive**.
- **DDPM (images)** part d’un bruit aléatoire et le débruite progressivement pour produire une image.

Dans les deux cas, un modèle peut parfois trop bien apprendre certains exemples d’entraînement. Cela peut permettre de retrouver des éléments mémorisés. Pour le texte, on recherche des séquences reproduites mot pour mot ; pour la diffusion, on cherche notamment des images identiques ou presque identiques produites à partir de bruits initiaux différents.

> **Vocabulaire :** un *token* est une unité de texte (mot, morceau de mot ou signe) ; un *logit* est un score brut avant softmax ; une *perte* mesure l’erreur d’un modèle ; un *surapprentissage* apparaît lorsque le modèle s’adapte excessivement aux exemples d’entraînement et généralise moins bien.

---

# Partie A — Génération textuelle : NanoGPT

## Diapositive 13 — Anatomie de NanoGPT : factorisation autorégressive

### 1. La formule centrale

\[
P(X)=\prod_{t=1}^{N}P(x_t\mid x_{<t})
\]

- \(X=(x_1,\ldots,x_N)\) est une séquence de tokens.
- \(x_{<t}\) désigne tous les tokens qui précèdent le token \(x_t\).
- \(P(x_t\mid x_{<t})\) est la probabilité attribuée au token courant sachant le contexte précédent.
- Le produit signifie que la probabilité de la séquence complète est obtenue en multipliant les probabilités conditionnelles successives.

**Exemple intuitif :** pour générer « Le chat dort », le modèle estime d’abord la suite possible après « Le », puis après « Le chat », puis après « Le chat dort ». Il ne choisit pas toute la phrase d’un seul coup.

### 2. Pourquoi « causal » ?

Le masque d’attention causal empêche la position \(t\) de regarder les tokens futurs \(x_{t+1},x_{t+2},\ldots\). Pendant l’apprentissage comme pendant la génération, la prédiction doit dépendre du passé, pas d’informations futures indisponibles en situation réelle.

### 3. Des logits aux probabilités

Pour chaque token candidat \(w\), le réseau produit un logit \(z_w\). La softmax avec température \(T\) donne :

\[
P(w)=\frac{\exp(z_w/T)}{\sum_j \exp(z_j/T)}
\]

- Le numérateur transforme le score du token candidat en valeur positive.
- Le dénominateur additionne ces valeurs pour que toutes les probabilités totalisent 1.
- \(T\) règle la dispersion de la distribution.

### 4. Que fait le code ?

1. Le modèle calcule les logits du contexte.
2. Il conserve ceux de la dernière position, qui correspondent au prochain token.
3. Il divise les logits par la température.
4. La softmax transforme les scores en probabilités.
5. `torch.multinomial` tire un token selon cette distribution.
6. Le token choisi est ajouté à la séquence, qui sert de nouveau contexte.

**Implication sécurité :** si un contexte conduit le modèle à donner une très forte probabilité à la suite d’un texte sensible mémorisé, le modèle peut la reproduire. La génération récursive peut ensuite continuer cette séquence.

**Nuance importante :** une forte probabilité ne rend pas automatiquement la sortie déterministe. Avec un échantillonnage aléatoire, une température finie et une distribution non dégénérée, plusieurs tokens restent possibles. Une température plus basse rend les choix plus concentrés ; elle ne prouve pas à elle seule une mémorisation.

---

## Diapositive 14 — Pipeline de décision autorégressif

Cette diapositive reprend le mécanisme précédent sous forme de chaîne de traitement :

1. **Prompt / contexte :** texte fourni au modèle.
2. **Attention causale + MLP :** le Transformer combine le contexte disponible pour construire une représentation utile.
3. **Tête linéaire :** produit un logit pour chaque token du vocabulaire.
4. **Softmax et température :** convertissent les logits en distribution de probabilité.
5. **Échantillonnage :** sélectionne le prochain token.
6. **Boucle :** ajoute ce token au contexte et recommence.

### Pourquoi une séquence mémorisée peut-elle se poursuivre ?

Si le contexte correspond au début d’un passage appris par cœur, le modèle peut attribuer une probabilité élevée au token suivant du passage. Ce token devient alors une partie du contexte, ce qui peut rendre le token suivant du même passage très probable, et ainsi de suite.

C’est un **effet de continuation** : une sortie mémorisée peut s’auto-entretenir au fil des étapes.

**À ne pas dire trop fortement :** « le token mémorisé domine, donc la boucle est forcément déterministe ». La formulation correcte est : *si les logits favorisent fortement la continuation mémorisée, le modèle a davantage de chances de la reproduire*. Le caractère exact de la génération dépend aussi de la température, de l’échantillonnage et du contexte.

---

## Diapositive 15 — Perte, mémorisation et longue traîne

### 1. La cross-entropy

\[
L_{\mathrm{CE}}=-\frac{1}{T}\sum_{t=1}^{T}\log P_\theta(x_t\mid x_{<t})
\]

Ici, \(T\) désigne la longueur de la séquence dans cette formule (et non la température de génération utilisée à la diapositive précédente).

Pour chaque position, le modèle est pénalisé lorsque la probabilité du vrai token est faible :
- vraie suite très probable : \(-\log P\) proche de 0 ;
- vraie suite peu probable : perte plus élevée.

La moyenne des pertes sur les tokens mesure donc à quel point le modèle prédit bien la séquence.

### 2. Que signifie \(L_{\mathrm{CE}}\to 0\) ?

Pour les tokens évalués, une perte proche de zéro signifie que le modèle attribue une probabilité proche de 1 aux tokens attendus. Leurs logits sont alors très élevés **relativement** à ceux des autres candidats.

Cela peut être un signe de mémorisation si cette perte très basse concerne un exemple d’entraînement précis et ne se retrouve pas sur des exemples nouveaux comparables. Ce n’est toutefois pas une preuve universelle de mémorisation : une séquence courante ou très prévisible peut aussi avoir une faible perte.

### 3. La longue traîne et la loi de Zipf

Dans une distribution de type Zipf, les éléments fréquents sont nombreux à être rencontrés, tandis qu’une longue traîne contient des éléments très rares.

- **Exemples fréquents :** le modèle peut apprendre des régularités générales qui se réutilisent.
- **Exemples rares :** il dispose de peu d’occasions d’apprendre leur structure. Pour réduire l’erreur sur ces cas, il peut devoir retenir des détails spécifiques de certains exemples.

Le travail de Vitaly Feldman cité sur la diapositive établit, dans certains cadres théoriques de distributions à longue traîne, que mémoriser des exemples rares peut être nécessaire pour obtenir de bonnes performances moyennes. Cela aide à comprendre le paradoxe : **la mémorisation peut améliorer la généralisation**, mais les détails mémorisés peuvent aussi être sensibles.

**Limite à rappeler à l’oral :** ce résultat théorique ne signifie pas que tout LLM mémorise forcément chaque exemple rare, ni que toute baisse de perte prouve qu’un texte complet est stocké mot pour mot. La mention « 100–200 époques » est un scénario illustratif de la présentation, pas une règle valable pour tous les modèles et jeux de données.

---

## Diapositive 16 — Attaques : extraction par préfixe et extraction aveugle

### Attaque 1 : extraction guidée par préfixe

**Hypothèse :** l’attaquant connaît le début d’un texte, par exemple « L’employé n°441 est… ».

**Mécanisme :**
1. Il soumet le préfixe comme prompt.
2. Le modèle prédit les tokens suivants.
3. Si ce préfixe correspond à un passage mémorisé, le modèle peut poursuivre avec le suffixe d’origine.

**Ce que l’attaque peut révéler :** la suite d’un document, des données personnelles ou un passage confidentiel présent dans les données d’entraînement.

Le préfixe n’est pas une preuve que le modèle connaît le secret : il sert à orienter la génération vers une continuation particulière.

### Attaque 2 : extraction aveugle

**Hypothèse :** l’attaquant ne connaît pas le début du texte à extraire.

**Mécanisme général :**
1. Générer un grand nombre de textes.
2. Les classer selon des signaux susceptibles de révéler des passages mémorisés.
3. Dédupliquer les résultats.
4. Examiner les candidats les plus prometteurs.
5. Vérifier si les textes correspondent réellement à des passages de données d’entraînement accessibles ou connus.

**Difficulté centrale :** un texte cohérent peut être généré sans avoir été mémorisé mot pour mot. Il faut distinguer une reproduction d’un exemple d’entraînement d’une simple phrase plausible.

**Comparaison :** l’extraction par préfixe bénéficie d’un indice sur le passage recherché ; l’extraction aveugle doit explorer beaucoup de sorties sans savoir à l’avance où se trouve le secret.

---

## Diapositive 17 — Workflow d’extraction aveugle (Carlini et al., 2021)

La diapositive décrit une étude sur GPT-2 XL, et non une garantie que tous les modèles se comportent de la même façon.

### Phase 1 — Générer et filtrer
- Produire un grand ensemble d’échantillons.
- Calculer plusieurs métriques de score.
- Trier les échantillons susceptibles d’être mémorisés.
- Supprimer les doublons afin de ne pas examiner plusieurs fois le même passage.

### Phase 2 — Vérifier
- Examiner manuellement les meilleurs candidats.
- Comparer les textes à des sources disponibles.
- Vérifier, lorsque c’est possible, leur présence dans le corpus d’entraînement étudié.

Le volume « 200 000 échantillons » et l’inspection du « Top 100 » sont les paramètres rapportés par la présentation pour ce protocole précis. Ce ne sont pas des seuils universels.

**Pourquoi la vérification est indispensable :** une métrique peut sélectionner un texte très prévisible sans qu’il provienne des données d’entraînement. L’extraction doit être confirmée par des éléments externes, pas seulement par un score.

---

## Diapositive 18 — Pourquoi la perplexité seule ne suffit pas : filtrage zlib

### 1. La perplexité (PPL)

La perplexité est liée à la probabilité que le modèle attribue aux tokens d’un texte. En général, une perplexité basse signifie que le texte est facile à prédire pour le modèle.

Mais un texte facile à prédire n’est pas nécessairement mémorisé. Par exemple, une répétition triviale comme « 1 1 1 1 1 » peut être très prévisible.

### 2. L’idée de la compression zlib

zlib estime la compressibilité d’un texte :
- un texte très répétitif se compresse généralement bien ;
- un texte contenant davantage de structure et de variété est souvent moins compressible.

La diapositive propose le ratio :

\[
R(x)=\frac{\mathrm{PPL}_{\mathrm{cible}}(x)}{\mathrm{Compression}_{\mathrm{zlib}}(x)}
\]

Ce ratio combine deux indices : la facilité de prédiction par le modèle et la compressibilité du texte. L’objectif est de mieux faire ressortir des passages que le modèle prédit particulièrement bien sans que leur score soit expliqué uniquement par une répétition triviale.

**Attention à l’interprétation :** la présentation ne précise pas ici la convention numérique exacte utilisée pour « Compression zlib » (taille compressée, ratio de compression ou autre normalisation). Il ne faut donc pas affirmer une direction universelle du ratio sans consulter la définition de l’expérience originale. Le message à retenir est l’utilisation conjointe de la perplexité et de la compressibilité, suivie d’une validation.

Les « 59 confirmés » sont le résultat rapporté par la diapositive pour l’étude citée, pas un taux de réussite garanti sur d’autres modèles.

---

## Diapositive 19 — Inférence d’appartenance textuelle (MIA)

La *Membership Inference Attack* cherche à déterminer si un texte donné a fait partie des données d’entraînement.

### 1. Test par seuil de perte

\[
M(x)=
\begin{cases}
1 & \text{si }L_{\mathrm{CE}}(x;\theta)\leq \tau\\
0 & \text{sinon}
\end{cases}
\]

- \(x\) : texte cible.
- \(\theta\) : paramètres du modèle.
- \(L_{\mathrm{CE}}\) : perte calculée sur ce texte.
- \(\tau\) : seuil choisi à l’avance.
- \(M(x)=1\) : le test classe le texte comme membre potentiel ; \(M(x)=0\) : comme non-membre.

**Intuition :** si le modèle a surappris ses exemples d’entraînement, il peut leur attribuer une perte plus faible qu’à des textes comparables qu’il n’a jamais vus.

Le seuil doit être calibré sur des données de validation disjointes. En pratique, les textes membres ne sont pas toujours les seuls à obtenir une faible perte : la difficulté intrinsèque, la fréquence et la prévisibilité du texte peuvent aussi influencer le score. Il peut donc y avoir des faux positifs et des faux négatifs.

### 2. Ratio de référence

\[
\mathrm{RatioMIA}(x)=\frac{L(x;\theta_{\mathrm{cible}})}
{L(x;\theta_{\mathrm{ref}})}
\]

On compare la perte du modèle cible à celle d’un modèle de référence. L’idée est de corriger partiellement le fait que certains textes sont naturellement plus faciles à prédire que d’autres.

- Si le modèle cible prédit un texte beaucoup mieux que le modèle de référence, le ratio peut devenir faible.
- Cette différence peut signaler une familiarité particulière du modèle cible avec le texte.

Ce ratio reste un **indice statistique**, pas une preuve certaine d’appartenance. Son interprétation dépend notamment du modèle de référence et de la méthode de calibration.

### Bilan de la partie NanoGPT

La chaîne logique est :

**texte d’entraînement rare ou sensible → apprentissage très poussé → perte faible / continuation très probable → inférence d’appartenance ou extraction du texte.**

- **MIA :** inférer si un texte était dans l’entraînement.
- **Extraction :** tenter de récupérer directement son contenu.
- **Perplexité + zlib :** classer les candidats à vérifier, sans remplacer la vérification.

---

# Partie B — Modèles de diffusion : DDPM

## Diapositive 21 — Bruitage direct et débruitage inverse

Un DDPM apprend à générer des données en construisant deux processus.

### 1. Processus direct : ajouter du bruit

\[
x_t=\sqrt{\bar{\alpha}_t}\,x_0+
\sqrt{1-\bar{\alpha}_t}\,\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I)
\]

- \(x_0\) : image originale.
- \(x_t\) : image après ajout de bruit à l’étape \(t\).
- \(\epsilon\) : bruit gaussien aléatoire.
- \(\bar{\alpha}_t\) : coefficient qui contrôle la quantité de signal original encore présente à l’étape \(t\).

Au début, \(x_t\) ressemble encore à l’image. Plus \(t\) augmente, plus le bruit prend de place. À l’étape finale, l’image est approximativement du bruit gaussien pur.

Ce processus est défini mathématiquement : il ne demande pas au réseau d’apprendre comment ajouter le bruit.

### 2. Processus inverse : retirer le bruit

Le réseau \(\epsilon_\theta(x_t,t)\), généralement basé sur un U-Net, estime le bruit présent dans l’image à l’étape \(t\). Le modèle utilise cette estimation pour calculer une étape de débruitage, répétée jusqu’à obtenir une image.

La formule présentée pour la moyenne de l’étape inverse est :

\[
\mu_\theta(x_t,t)=
\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,
\epsilon_\theta(x_t,t)\right)
\]

- \(\alpha_t\) et \(\beta_t\) sont des coefficients du calendrier de bruit.
- \(\epsilon_\theta\) est le bruit estimé par le réseau.
- \(\mu_\theta\) indique le centre de la distribution de l’étape inverse ; selon la formulation DDPM, une composante aléatoire peut aussi intervenir dans l’échantillonnage.

**À retenir :** le réseau ne dessine pas nécessairement l’image entière en une fois. Il estime le bruit et le processus inverse transforme progressivement le bruit initial en image.

**Lien avec la sécurité :** si l’apprentissage a créé une préférence très forte pour une image d’entraînement, certaines trajectoires de débruitage peuvent aboutir à cette image ou à une version presque identique.

---

## Diapositive 22 — Denoising Score Matching : que minimise le réseau ?

### 1. Fonction objectif

\[
L_{\mathrm{simple}}(\theta)=
\mathbb{E}_{t,x_0,\epsilon}
\left[\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|^2\right]
\]

Cette formule signifie : on choisit une image d’entraînement \(x_0\), un instant \(t\) et un bruit aléatoire \(\epsilon\), puis on compare :
- le **vrai bruit** qui a été ajouté ;
- le **bruit estimé** par le réseau.

La norme au carré mesure l’écart entre ces deux valeurs. L’espérance \(\mathbb{E}\) signifie qu’on moyenne cette erreur sur de nombreux choix d’images, de temps et de bruits.

**Si la perte baisse :** le réseau estime mieux le bruit dans les exemples et conditions utilisés pour l’apprentissage. Cela ne signifie pas automatiquement qu’il a mémorisé une image précise : il peut aussi avoir appris des régularités générales.

### 2. Lien avec le score

La diapositive donne :

\[
\nabla_{x_t}\log p_t(x_t)
\approx -\frac{\epsilon_\theta(x_t,t)}
{\sqrt{1-\bar{\alpha}_t}}
\]

- \(p_t(x_t)\) représente la distribution des images bruitées à l’étape \(t\).
- \(\log p_t(x_t)\) est le logarithme de cette densité.
- Le gradient \(\nabla_{x_t}\log p_t(x_t)\), appelé **score**, indique dans quelle direction la densité augmente le plus localement.
- L’estimation du bruit permet donc d’estimer cette direction de variation de la densité.

L’image intuitive du « champ de gradient » signifie que le processus inverse utilise des directions locales pour transformer progressivement un bruit vers des régions plus compatibles avec les images apprises.

**Nuance :** la formule est une relation théorique sous les hypothèses du processus de bruitage. Elle ne signifie pas que le réseau connaît explicitement la probabilité exacte de chaque image, ni qu’il suit toujours une trajectoire déterministe.

---

## Diapositive 23 — Mémorisation et bassins d’attraction

### 1. Qu’est-ce qu’un bassin d’attraction ?

Imaginez plusieurs points de départ bruités. Le débruitage les transforme progressivement en images. Un **bassin d’attraction** est une région de départs qui conduisent vers une même sortie, ou vers des sorties très proches.

Dans le contexte de la présentation :
1. une image est présente de façon répétée dans les données, ou le modèle est fortement ajusté à cette image ;
2. le modèle peut apprendre une préférence particulièrement forte pour cette image ;
3. des bruits initiaux différents peuvent alors aboutir à la même image mémorisée.

La métaphore du « puits de potentiel » représente cette attraction. Ce n’est pas nécessairement un potentiel physique explicitement calculé par le modèle.

### 2. Pourquoi est-ce un risque de confidentialité ?

Si des bruits indépendants produisent plusieurs fois la même photo privée, cette répétition peut indiquer que le modèle ne fait pas que générer une image plausible : il pourrait reproduire un exemple précis de l’entraînement.

Cependant, une sortie visuellement similaire ne prouve pas à elle seule qu’il s’agit d’une copie exacte. Il faut comparer les sorties entre elles et, si possible, à l’image source.

### 3. Quel rôle jouent les doublons ?

Des doublons dans le jeu d’entraînement peuvent renforcer l’influence d’un exemple : l’optimisation rencontre cette image plusieurs fois. Cela peut favoriser sa reproduction, notamment en présence de surentraînement. Mais le résultat dépend aussi de l’architecture, du jeu de données, de l’entraînement et de la procédure de génération.

**À retenir :** l’hypothèse de l’attaque est que des départs différents convergent anormalement vers une même sortie mémorisée.

---

## Diapositive 24 — Détection de cliques multi-graines et CFG

### 1. Extraction aveugle par plusieurs bruits

Une *graine* (*seed*) permet de fixer l’aléa initial utilisé pour générer une sortie. Le protocole décrit consiste à :

1. tirer plusieurs bruits initiaux indépendants \(x_T^{(i)}\) ;
2. générer une image pour chaque bruit ;
3. comparer les images obtenues deux à deux ;
4. relier les paires suffisamment similaires ;
5. rechercher des groupes fortement connectés, appelés **cliques** dans la présentation.

Une clique est ici un groupe dans lequel chaque paire d’images respecte le critère de similarité choisi. Un groupe de sorties quasi identiques malgré des bruits initiaux distincts est un signal possible de mémorisation.

### 2. Comment mesurer la similarité ?

La diapositive cite deux possibilités :
- **MSE (Mean Squared Error)** : moyenne des différences pixel par pixel au carré. Une MSE basse indique des valeurs de pixels proches, si les images sont alignées et comparables.
- **SSIM** : mesure de similarité structurelle, qui compare notamment la luminance, le contraste et la structure visuelle.

On choisit un seuil \(\delta\). Pour une distance, on peut considérer une paire comme proche si \(d(\hat{x}^{(i)},\hat{x}^{(j)})<\delta\). Le choix du seuil compte : trop permissif, il regroupe des images différentes ; trop strict, il peut manquer des copies légèrement transformées.

**Prudence sur la conclusion :** une clique signale une répétition anormale à investiguer ; elle ne constitue pas à elle seule une preuve que toutes les images sont des extractions privées. Des images communes ou des sorties très similaires peuvent aussi apparaître naturellement.

### 3. CFG : Classifier-Free Guidance

La formule de la diapositive est :

\[
\hat{\epsilon}=
\epsilon_\theta(x_t,\varnothing)+
s\big(\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing)\big)
\]

- \(\epsilon_\theta(x_t,\varnothing)\) : estimation du bruit sans condition.
- \(\epsilon_\theta(x_t,c)\) : estimation du bruit avec la condition \(c\) (par exemple, un texte).
- \(s\) : intensité du guidage.
- La différence entre les deux estimations représente l’effet de la condition ; le coefficient \(s\) amplifie ou réduit cet effet.

Lorsque \(s>1\), le modèle suit généralement plus fortement la condition, ce qui peut rendre les sorties plus conformes au prompt. La présentation avance que cela peut aussi accentuer des attracteurs mémorisés et élargir leur bassin. Il faut le présenter comme un **risque à tester expérimentalement**, pas comme une conséquence garantie pour tout modèle ou toute valeur de \(s\).

### 4. Pourquoi la collision aléatoire est-elle intéressante ?

Dans un espace de très grande dimension, deux bruits indépendants ont peu de chances de produire par hasard exactement la même image. Si plusieurs bruits distincts convergent vers une sortie presque identique, cela motive l’hypothèse d’un attracteur appris.

La probabilité « inférieure à \(10^{-100}\) » indiquée sur la diapositive dépend toutefois d’hypothèses sur l’espace, la notion de distance et la distribution des sorties. Il vaut mieux l’expliquer comme une illustration de la rareté attendue d’une collision aléatoire exacte, et non comme une borne universelle démontrée pour toutes les images générées.

### Bilan de la partie DDPM

La chaîne logique est :

**image sensible répétée ou surapprise → préférence très forte du modèle → convergence de plusieurs bruits vers des sorties similaires → détection possible par comparaison multi-graines → vérification d’une éventuelle reproduction.**

---

# Comparaison finale : NanoGPT vs DDPM

| Point | NanoGPT — texte | DDPM — images |
|---|---|---|
| Données générées | Tokens discrets | Valeurs d’image continues |
| Mécanisme | Prédire le prochain token, puis recommencer | Débruiter progressivement un bruit initial |
| Signal possible de mémorisation | Faible perte sur un texte, continuation très probable | Plusieurs bruits donnant des images presque identiques |
| Attaque principale présentée | Extraction guidée par préfixe ou extraction aveugle | Extraction aveugle par multi-graines et recherche de cliques |
| Mesures citées | Cross-entropy, perplexité, ratio avec zlib, ratio de pertes | MSE, SSIM, distance entre sorties |
| Risque | Restitution verbatim d’un passage sensible | Reproduction d’une image privée ou d’un exemple d’entraînement |
| Limite de l’indice | Une faible perte ne prouve pas une copie mémorisée | Une similarité ne prouve pas à elle seule une extraction privée |

## Trois distinctions à maîtriser à l’oral

1. **Inférence vs extraction :** l’inférence d’appartenance cherche à savoir si un exemple était dans l’entraînement ; l’extraction tente de récupérer le contenu lui-même.
2. **Perte faible vs preuve de mémorisation :** une perte faible est un signal possible, mais la fréquence et la difficulté intrinsèque d’un exemple peuvent aussi l’expliquer.
3. **Sorties similaires vs preuve de fuite :** une convergence répétée est suspecte, mais il faut comparer et valider les sorties avant de conclure à la restitution d’une donnée privée.

## Mini-script de conclusion

« Pour NanoGPT, le modèle prédit les tokens les uns après les autres. Si un passage rare a été fortement mémorisé, un préfixe adapté ou une exploration massive des sorties peut permettre de le retrouver. Pour DDPM, le modèle apprend à retirer le bruit ; une image mémorisée peut agir comme un attracteur vers lequel plusieurs bruits différents convergent. Dans les deux cas, le surapprentissage peut transformer l’apprentissage d’une distribution en reproduction d’exemples précis. Les métriques d’attaque servent à identifier des candidats, mais une fuite doit être validée : un score faible ou une similarité visuelle n’est pas, isolément, une preuve absolue. »
