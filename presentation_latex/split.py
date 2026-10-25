with open("main.tex", "r") as f:
    c = f.read()

c = c.replace(
r"""\begin{frame}{Définitions}
    \subtitleline{Trois notions à ne pas confondre}
    
    \begin{block}{Généralisation}
        Le modèle performe bien sur des données jamais vues pendant l'entraînement.\\
        Mesure : écart entre erreur de train et de test.
    \end{block}
    
    \begin{alertblock}{Surapprentissage}
        Bon sur les données d'entraînement, mauvais sur des données nouvelles.\\
        Symptôme : grand écart train / test. Problème global de performance.
    \end{alertblock}
    
    \begin{exampleblock}{Mémorisation}
        Le modèle reproduit (presque) exactement un exemple d'entraînement précis.\\
        Feldman (2020) : différence de prédiction correcte avec / sans l'exemple dans le train. Phénomène \textbf{local}, pas global.
    \end{exampleblock}
    
    \vspace{0.2cm}
    \textbf{Point clé du sujet : mémorisation $\neq$ surapprentissage}
\end{frame}""",
r"""\begin{frame}{Définitions (1/2)}
    \subtitleline{Trois notions à ne pas confondre}
    
    \begin{block}{Généralisation}
        Le modèle performe bien sur des données jamais vues pendant l'entraînement.\\
        Mesure : écart entre erreur de train et de test.
    \end{block}
    
    \begin{alertblock}{Surapprentissage}
        Bon sur les données d'entraînement, mauvais sur des données nouvelles.\\
        Symptôme : grand écart train / test. Problème global de performance.
    \end{alertblock}
\end{frame}

\begin{frame}{Définitions (2/2)}
    \subtitleline{Trois notions à ne pas confondre}
    
    \begin{exampleblock}{Mémorisation}
        Le modèle reproduit (presque) exactement un exemple d'entraînement précis.\\
        Feldman (2020) : différence de prédiction correcte avec / sans l'exemple dans le train. Phénomène \textbf{local}, pas global.
    \end{exampleblock}
    
    \vspace{0.4cm}
    \textbf{Point clé du sujet : mémorisation $\neq$ surapprentissage}
\end{frame}""")

c = c.replace(
r"""\begin{frame}{Choix à trancher en réunion}
    \subtitleline{Sur quelle étape se concentrer ?}
    
    \begin{description}
        \item[\textbf{Prédiction / génération — modèles pré-entraînés}] \hfill \\
        Cœur du sujet : extraction et MIA en boîte noire/grise sur GPT-2, Pythia, petits modèles de diffusion — pas besoin de ré-entraîner
        \vspace{0.2cm}
        \item[\textbf{Entraînement — via les checkpoints Pythia}] \hfill \\
        Répond à l'objectif 1 (quand la mémorisation apparaît) sans le coût d'un entraînement complet
        \vspace{0.2cm}
        \item[\textbf{Classification — repère théorique seulement}] \hfill \\
        Utile en slide pédagogique (MIA historique), mais hors périmètre expérimental — s'éloigne de l'IA générative
    \end{description}
    \vspace{0.4cm}
    \textbf{Question ouverte : quel budget de calcul est disponible ?}
\end{frame}""",
r"""\begin{frame}{Choix à trancher en réunion (1/2)}
    \subtitleline{Sur quelle étape se concentrer ?}
    
    \begin{description}
        \item[\textbf{Prédiction / génération — modèles pré-entraînés}] \hfill \\
        Cœur du sujet : extraction et MIA en boîte noire/grise sur GPT-2, Pythia, petits modèles de diffusion — pas besoin de ré-entraîner
        \vspace{0.4cm}
        \item[\textbf{Entraînement — via les checkpoints Pythia}] \hfill \\
        Répond à l'objectif 1 (quand la mémorisation apparaît) sans le coût d'un entraînement complet
    \end{description}
\end{frame}

\begin{frame}{Choix à trancher en réunion (2/2)}
    \subtitleline{Sur quelle étape se concentrer ?}
    
    \begin{description}
        \item[\textbf{Classification — repère théorique seulement}] \hfill \\
        Utile en slide pédagogique (MIA historique), mais hors périmètre expérimental — s'éloigne de l'IA générative
    \end{description}
    \vspace{0.4cm}
    \textbf{Question ouverte : quel budget de calcul est disponible ?}
\end{frame}""")

c = c.replace(
r"""\begin{frame}{Mon angle}
    \subtitleline{Un modèle de menace explicite}
    
    \begin{columns}[T]
        \begin{column}{0.48\textwidth}
            \begin{itemize}
                \item \textbf{Qui est l'attaquant ?} \\ Utilisateur curieux, concurrent, insider
                \item \textbf{Quel accès ?} \\ Boîte blanche, grise ou noire
            \end{itemize}
        \end{column}
        \begin{column}{0.48\textwidth}
            \begin{itemize}
                \item \textbf{Que sait-il ?} \\ Distribution, architecture, shadow models, préfixe connu
                \item \textbf{Que veut-il ?} \\ Extraction, MIA, inférence d'attributs, reconstruction
            \end{itemize}
        \end{column}
    \end{columns}
    
    \vspace{0.4cm}
    \begin{alertblock}{}
    Risque = probabilité de succès de l'attaque × sensibilité de la donnée × faisabilité pour l'attaquant. \textbf{Objectif : relier métriques de mémorisation et risque réel de fuite.}
    \end{alertblock}
\end{frame}""",
r"""\begin{frame}{Mon angle (1/2)}
    \subtitleline{Un modèle de menace explicite}
    
    \begin{columns}[T]
        \begin{column}{0.48\textwidth}
            \begin{itemize}
                \item \textbf{Qui est l'attaquant ?} \\ Utilisateur curieux, concurrent, insider
                \item \textbf{Quel accès ?} \\ Boîte blanche, grise ou noire
            \end{itemize}
        \end{column}
        \begin{column}{0.48\textwidth}
            \begin{itemize}
                \item \textbf{Que sait-il ?} \\ Distribution, architecture, shadow models, préfixe connu
                \item \textbf{Que veut-il ?} \\ Extraction, MIA, inférence d'attributs, reconstruction
            \end{itemize}
        \end{column}
    \end{columns}
\end{frame}

\begin{frame}{Mon angle (2/2)}
    \subtitleline{Un modèle de menace explicite}
    
    \vspace{0.4cm}
    \begin{alertblock}{}
    Risque = probabilité de succès de l'attaque × sensibilité de la donnée × faisabilité pour l'attaquant. \textbf{Objectif : relier métriques de mémorisation et risque réel de fuite.}
    \end{alertblock}
\end{frame}""")

c = c.replace(
r"""    \vspace{0.2cm}
    Un modèle peut très bien généraliser et pourtant mémoriser certains exemples : les défenses classiques contre l'overfitting ne suffisent donc pas à protéger la confidentialité.
\end{frame}""",
r"""    \vspace{0.1cm}
    {\footnotesize Un modèle peut très bien généraliser et pourtant mémoriser certains exemples : les défenses classiques contre l'overfitting ne suffisent donc pas à protéger la confidentialité.\par}
\end{frame}""")

with open("main.tex", "w") as f:
    f.write(c)
