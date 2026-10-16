"""
Typesets the formulas of the PoC slide deck with LaTeX (latex + dvipng, MiKTeX).

Each formula -> <name>.png, transparent background, rendered at 2x so it stays
sharp; the slide shows it at half its pixel size (printed below).

Usage (from this folder):  python build_formulas.py
"""

import os
import subprocess
import tempfile

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DPI = (
    440  # 12pt at 440 dpi = 73 px per em; shown at 1/2 -> ~37 px on the 1920x1080 slide
)

INK = "15202E"  # dark text, light cards
LIGHT = "F6F5F1"  # light text, dark cards

FORMULAS = {
    # slide 7 - formulation
    "f_baseline": (INK, r"\hat y(x) = \arg\max_{c \in \mathcal{C}} \; z_c(x)"),
    "f_ensemble": (
        INK,
        r"\big\{ (\mathbf{p}_q,\, m_q) \big\}_{q=1}^{N}, \qquad N \in \{100,\, 200\}",
    ),
    "f_inference": (
        INK,
        r"S_c(x) = \sum_{q=1}^{N} p_q(c)\; \sigma\!\big(m_q(x)\big), \qquad \hat y(x) = \arg\max_{c} S_c(x)",
    ),
    "f_cibles": (
        INK,
        r"\mathcal{G}(y) = \big\{ (c_j,\, g_j) \big\}_{j=1}^{K}, \qquad g_j(x) = \mathbb{1}\big[\, y(x) = c_j \,\big]",
    ),
    # slide 8 - appariement
    "f_cout": (
        INK,
        r"\mathcal{C}(j,q) = -2\, p_q(c_j) + 5\, \mathrm{BCE}(m_q, g_j) + 5\, \mathrm{Dice}(m_q, g_j)",
    ),
    "f_hongrois": (
        INK,
        r"\hat\sigma = \arg\min_{\sigma} \sum_{j=1}^{K} \mathcal{C}\big(j, \sigma(j)\big)",
    ),
    "f_perte": (
        INK,
        r"\mathcal{L} = \sum_{l} \Big[\, 2\, \mathcal{L}^{(l)}_{\mathrm{cls}} + 5\, \mathcal{L}^{(l)}_{\mathrm{BCE}} + 5\, \mathcal{L}^{(l)}_{\mathrm{Dice}} \Big]",
    ),
    # slide 9 - attention masquee (dark card)
    "f_attention": (
        LIGHT,
        r"X_l = \mathrm{softmax}\!\Big( \mathcal{M}_{l-1} + \frac{Q_l K_l^{\top}}{\sqrt{d}} \Big) V_l + X_{l-1}",
    ),
    "f_masque": (
        LIGHT,
        r"\mathcal{M}_{l-1}(q,x) = \begin{cases} 0 & \text{si } \sigma\big(m_q^{(l-1)}(x)\big) \ge 0{,}5 \\ -\infty & \text{sinon} \end{cases}",
    ),
    # slide 10 - EoMT
    "f_classe": (INK, r"\mathbf{p}_n = \mathrm{softmax}\big( W \mathbf{q}_n \big)"),
    "f_masque_eomt": (
        INK,
        r"m_n = \big\langle \mathrm{MLP}_3(\mathbf{q}_n),\; \mathcal{U}(T) \big\rangle",
    ),
    # slide 11 - mask annealing
    "f_annealing": (
        INK,
        r"P_b(t) = \Big( 1 - \frac{t - t^{s}_b}{t^{e}_b - t^{s}_b} \Big)^{0{,}9}, \quad t^{s}_b \le t < t^{e}_b",
    ),
}

TEMPLATE = r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb,bm,dsfont}
\usepackage{xcolor}
\pagestyle{empty}
\renewcommand{\mathbb}[1]{\mathds{#1}}
\begin{document}
\definecolor{fg}{HTML}{%s}\color{fg}
$\displaystyle %s$
\end{document}
"""


def build(name, color, body):
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "f.tex"), "w", encoding="utf-8") as f:
            f.write(TEMPLATE % (color, body))
        subprocess.run(
            ["latex", "-interaction=nonstopmode", "-halt-on-error", "f.tex"],
            cwd=tmp,
            check=True,
            capture_output=True,
        )
        out = os.path.join(HERE, f"{name}.png")
        subprocess.run(
            [
                "dvipng",
                "-q",
                "-T",
                "tight",
                "-D",
                str(DPI),
                "-bg",
                "Transparent",
                "-o",
                out,
                "f.dvi",
            ],
            cwd=tmp,
            check=True,
            capture_output=True,
        )
    w, h = Image.open(out).size
    return w, h


if __name__ == "__main__":
    for name, (color, body) in FORMULAS.items():
        w, h = build(name, color, body)
        print(f"{name:15s} {w:5d}x{h:<4d} -> shown {w // 2}x{h // 2}")
