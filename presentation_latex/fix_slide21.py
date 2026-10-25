with open("main.tex", "r") as f:
    c = f.read()

old_slide = r"""\begin{frame}{Objectifs du projet}
    \subtitleline{Trois objectifs, un fil conducteur}
    
    \begin{block}{\obj{01} Comprendre la mémorisation}
        Comment et pourquoi les modèles mémorisent. À quelle étape (début / milieu / fin) et quels facteurs favorisent l'émergence.
    \end{block}
    
    \begin{block}{\obj{02} Relier mémorisation et surapprentissage}
        Dans quelles conditions ces deux phénomènes coïncident, ou au contraire se distinguent clairement.
    \end{block}
    
    \begin{block}{\obj{03} Détecter et évaluer le risque}
        Comparer des métriques de détection et les relier au risque réel de divulgation d'informations sensibles — \textbf{mon angle sécurité}.
    \end{block}
\end{frame}"""

new_slide = r"""\begin{frame}{Objectifs du projet (1/2)}
    \subtitleline{Trois objectifs, un fil conducteur}
    
    \begin{block}{\obj{01} Comprendre la mémorisation}
        Comment et pourquoi les modèles mémorisent. À quelle étape (début / milieu / fin) et quels facteurs favorisent l'émergence.
    \end{block}
    
    \begin{block}{\obj{02} Relier mémorisation et surapprentissage}
        Dans quelles conditions ces deux phénomènes coïncident, ou au contraire se distinguent clairement.
    \end{block}
\end{frame}

\begin{frame}{Objectifs du projet (2/2)}
    \subtitleline{Trois objectifs, un fil conducteur}
    
    \begin{block}{\obj{03} Détecter et évaluer le risque}
        Comparer des métriques de détection et les relier au risque réel de divulgation d'informations sensibles — \textbf{mon angle sécurité}.
    \end{block}
\end{frame}"""

c = c.replace(old_slide, new_slide)

with open("main.tex", "w") as f:
    f.write(c)
