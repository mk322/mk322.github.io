"""Export the user's Fig. 2 source, retaining its original artwork and typography.
Only four overview gradient arrows are omitted; the HTML adds them as animated
SVG paths at the original coordinates. The pristine source and helper definitions
live in fig2-animation/. Intermediate PDFs stay in a temporary directory.
"""
from pathlib import Path
import ast
import tempfile
import fitz
ROOT = Path(__file__).resolve().parent
source = ROOT / 'fig2-animation/fig2/fig2-overview-v45-reference.py'
tree = ast.parse(source.read_text())
with tempfile.TemporaryDirectory(prefix='uni-fig2-') as tmp:
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'OUT' for t in node.targets):
            node.value = ast.parse('Path(' + repr(str(Path(tmp)/'fig2.pdf')) + ')', mode='eval').body
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == 'gradient_arrow'
            and ast.literal_eval(node.value.args[0])[0][0] != 128):
            continue
        body.append(node)
    tree.body = body
    scope = {'__file__':str(source),'__name__':'__main__'}
    exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), scope)
    doc = fitz.open(Path(tmp)/'fig2.pdf')
    svg = doc[0].get_svg_image(text_as_path=True)
    (ROOT.parent/'fig2-forward.svg').write_text(svg)
    inner = svg[svg.index('>')+1:svg.rindex('</svg>')]
    include = ROOT.parents[3]/'_includes/blog/uni-ladir/fig2-art.html'
    include.write_text('<g id="uni-fig2-art" transform="scale(4.54545454545)">'+inner+'</g>\n')
    doc[0].get_pixmap(matrix=fitz.Matrix(4,4),alpha=False).save('/tmp/uni-fig2-forward.png')
print('Exported fig2-forward.svg')
