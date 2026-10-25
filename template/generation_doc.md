# Documentation Technique : Génération de Présentation LaTeX/Beamer
*Projet : PFE Sécurité de l'IA Générative - Télécom SudParis*

Ce document trace l'ensemble des choix techniques, esthétiques et correctifs mis en œuvre pour générer une présentation LaTeX professionnelle, fidèle à une charte graphique imposée, sans débordement ni erreur de compilation. Ce guide servira de référence pour les futures soutenances (mi-parcours, soutenance finale).

---

## 1. Outils et Moteur de Compilation
*   **Moteur recommandé :** `xelatex` (ou `lualatex`).
    *   *Pourquoi ?* Contrairement à `pdflatex`, `xelatex` permet de charger nativement des polices système (TrueType/OpenType) via le package `fontspec`, ce qui est indispensable pour respecter la typographie officielle de l'école (ex: *Liberation Sans Narrow* ou *Arial Narrow*).
*   **Commande de compilation :**
    ```bash
    xelatex -interaction=nonstopmode main.tex
    ```
    *(À exécuter deux ou trois fois pour générer correctement la table des matières et les références).*

---

## 2. Configuration Beamer & Charte Visuelle
Le préambule doit configurer le format et les couleurs exactes.
*   **Format :** `\documentclass[aspectratio=169, 10pt, table]{beamer}` (Écran large 16:9, taille de base 10pt pour la lisibilité).
*   **Palette de couleurs Télécom SudParis :**
    Toutes les teintes parasites (comme le violet générique) ont été bannies au profit d'une déclaration stricte :
    ```latex
    \definecolor{TSPblue}{HTML}{0052A0}   % Titres, en-têtes de tableaux, puces, cadres classiques
    \definecolor{TSPcyan}{HTML}{00A3E0}   % Sous-titres (italique)
    \definecolor{TSPblack}{HTML}{1D1D1B}  % Noir profond pour le texte d'alerte / logo
    \definecolor{TSPdark}{RGB}{128, 0, 32} % Bordeaux (pour les alertblocks, filets et mises en évidence)
    \definecolor{TSPgrey}{HTML}{F0F0F0}   % Gris très clair pour le fond des blocs
    \definecolor{TSPtext}{HTML}{333333}   % Gris très foncé (plus doux que le noir) pour le texte courant
    ```

---

## 3. Packages Indispensables
*   `fontspec` : Gestion des polices.
*   `polyglossia` (avec `\setmainlanguage{french}`) : Gestion de la langue et de la typographie française (plus robuste que `babel` sous `xelatex`).
*   `graphicx` & `adjustbox` : **Attention**, toujours charger `adjustbox` avec l'option `[export]` pour débloquer les paramètres `max width` et `max height` dans `\includegraphics`.
    ```latex
    \usepackage[export]{adjustbox}
    ```
*   `tabularx`, `array`, `booktabs`, `colortbl` : Pour les tableaux avancés.
*   `tikz` (avec les librairies `shapes, arrows.meta, positioning, shadows`) : Idéal pour dessiner des puces personnalisées ou des diagrammes de flux, afin d'éviter les glyphes Unicode non reconnus qui font planter `xelatex`.
*   `hyperref` : Pour les liens. **Astuce anti-bug :** Ajouter impérativement `\urlstyle{same}` après le chargement pour éviter l'erreur `\textfont 0 is undefined` causée par le mode mathématique des URLs dans les notes de bas de page.

---

## 4. Pièges classiques et Solutions (Lessons Learned)

### A. Débordements de texte (Overfull \vbox / \hbox)
*   **Problème :** Le texte sort de la slide ou se superpose au pied de page.
*   **Solution :** Ne **jamais** écraser drastiquement la taille de police (en dessous de 9pt, la slide devient illisible). Si le contenu d'une slide ne tient pas, **la scinder en deux** avec `(1/2)` et `(2/2)` dans le `\frametitle`.

### B. Mots coupés de manière disgracieuse
*   **Problème :** Beamer justifie le texte par défaut, ce qui provoque des césures (coupures de mots) laides ou de grands espaces blancs.
*   **Solution :** Aligner le texte à gauche et interdire la césure avec cette configuration :
    ```latex
    \AtBeginEnvironment{frame}{\raggedright\hyphenpenalty=10000}
    \AtBeginEnvironment{itemize}{\raggedright\hyphenpenalty=10000}
    \AtBeginEnvironment{block}{\raggedright\hyphenpenalty=10000}
    ```

### C. Espacements bizarres dans les tableaux (`tabularx`)
*   **Problème :** Les colonnes de type `X` (qui s'étirent automatiquement) justifient le texte. Si la colonne est étroite, on se retrouve avec d'immenses trous entre les mots.
*   **Solution :** Utiliser systématiquement la définition `>{\raggedright\arraybackslash}X`.
    ```latex
    \begin{tabularx}{\textwidth}{>{\bfseries}l >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X}
    ```

### D. Le texte a l'air "barré" par la couleur de fond
*   **Problème :** Dans les titres de blocs ou les en-têtes de tableau colorés (`\rowcolor`), la couleur de fond est trop fine en hauteur et le texte déborde au-dessus/en-dessous.
*   **Solutions :**
    1.  **Pour les tableaux :** Ajouter de l'épaisseur aux lignes avec `\renewcommand{\arraystretch}{1.4}`. Ne **jamais** placer de commande `\leavevmode` dans une cellule colorée avec `\rowcolor`, cela casse le centrage vertical.
        *Code correct :* `\rowcolor{TSPblue}\color{white}\textbf{\footnotesize Titre}`
    2.  **Pour les Blocs Beamer :** Ajuster manuellement la hauteur (`ht`) et la profondeur (`dp`) de la `beamercolorbox` dans le template :
        ```latex
        \begin{beamercolorbox}[dp=1.5ex,ht=3ex,leftskip=1ex,colsep*=.75ex]{block title}
        ```

### E. Résolution des images
*   **Problème :** L'image s'affiche en tout petit à cause de contraintes trop strictes.
*   **Solution :** Utiliser le paramètre `keepaspectratio` couplé à des limites relatives bien pensées :
    ```latex
    \includegraphics[max width=\textwidth, max height=0.7\textheight, keepaspectratio]{image.png}
    ```
