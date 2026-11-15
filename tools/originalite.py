# Comparaison d'originalite (15-11) : part des lignes du moteur identiques a Stockfish.
#   python tools/originalite.py engine/src <dossier Stockfish> [<autre version> ...]
import glob, os, re, sys
def norm_lines(path):
    t = open(path, encoding='utf-8', errors='ignore').read()
    t = re.sub(r'/\*.*?\*/', '', t, flags=re.S)
    out = []
    for l in t.splitlines():
        l = re.sub(r'//.*', '', l)
        l = re.sub(r'\s+', '', l)
        if len(l) >= 25 and not l.startswith('#include'):
            out.append(l)
    return out
root = sys.argv[1]; sf_dirs = sys.argv[2:]
sf = {}
for d in sf_dirs:
    for f in glob.glob(os.path.join(d, 'src', '**', '*.[ch]*'), recursive=True):
        for l in norm_lines(f):
            sf.setdefault(l, os.path.basename(d))
tot = same = 0
rows = []
for f in sorted(glob.glob(os.path.join(root, '*.[ch]*'))):
    ls = norm_lines(f)
    s = sum(1 for l in ls if l in sf)
    tot += len(ls); same += s
    rows.append((s / max(1, len(ls)), os.path.basename(f), s, len(ls)))
for r in sorted(rows, reverse=True):
    print(f"{r[1]:22s} {r[2]:5d}/{r[3]:5d}  {100*r[0]:5.1f} %")
print(f"TOTAL {same}/{tot} = {100*same/tot:.1f} % des lignes (>=25 car.) identiques a une ligne de Stockfish")
