with open("main.tex", "r") as f:
    c = f.read()

c = c.replace(r"\begin{beamercolorbox}[dp=1ex,ht=2.5ex,leftskip=1ex,colsep*=.75ex]{block title}", r"\begin{beamercolorbox}[dp=1.5ex,ht=3ex,leftskip=1ex,colsep*=.75ex]{block title}")
c = c.replace(r"\begin{beamercolorbox}[dp=1ex,ht=2.5ex,leftskip=1ex,colsep*=.75ex]{block title alerted}", r"\begin{beamercolorbox}[dp=1.5ex,ht=3ex,leftskip=1ex,colsep*=.75ex]{block title alerted}")
c = c.replace(r"\leavevmode\color{white}", r"\color{white}")

with open("main.tex", "w") as f:
    f.write(c)
