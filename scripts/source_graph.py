"""Resolve only active LaTeX inputs and figure dependencies from main.tex."""
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sources():
    out=[]
    def visit(p):
        p=p.resolve()
        assert p.is_relative_to(ROOT.resolve())
        if p in out:return
        out.append(p)
        text=p.read_text(encoding='utf-8')
        for item in re.findall(r'\\(?:input|include|tableinput)\{([^}]+)\}',text):
            q=ROOT/item
            if not q.suffix:q=q.with_suffix('.tex')
            visit(q)
    visit(ROOT/'main.tex');return out
def figure_files():
    return sorted({ROOT/name for p in sources() for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',p.read_text(encoding='utf-8'))})
