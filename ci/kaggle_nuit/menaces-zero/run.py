# Phase 10, D (16-11) : les menaces etroites sans defenses, entrainees DEPUIS ZERO.
# Les greffes sur le final ne s'en servaient presque pas (-0,5 cp) ; ici deux reseaux partent
# de zero, memes donnees, meme nombre de positions vues, meme pas :
#   temoin         : sans menaces
#   etroit_ennemis : bloc de 256 neurones, sans les defenses (-18 % de vitesse dans le moteur)
# Donnees : TOUT nodes5000pv2_UHO (40 Go, ~10 G positions, ~320 Go sur le disque de 970 Go du TPU) :
# les menaces ont 5 fois plus de poids que la base, elles ont besoin de beaucoup de positions differentes.
# Juge : l'erreur de validation (meme fichier), puis un match au temps entre les deux reseaux.
# Ne pas importer jax ici : le processus reserverait le TPU.
import glob, io, os, re, subprocess, sys, tarfile, time, urllib.request

W = '/kaggle/working'
t0 = time.time()


def lancer(cmd, journal):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sortie = []
    with open(W + '/' + journal, 'a', buffering=1) as j:
        for l in p.stdout:
            print(l, end='', flush=True)
            j.write(l)
            sortie.append(l)
    return p.wait(), ''.join(sortie)


_d = subprocess.run([sys.executable, '-c', 'import jax; print(jax.devices()[0].platform, jax.device_count())'],
                    capture_output=True, text=True).stdout.strip().splitlines()
print('machine :', _d, flush=True)
if not _d or not _d[-1].startswith('tpu'):
    sys.exit('PAS DE TPU : arret')

tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(
    'https://codeload.github.com/ovnice-dev/onechess-tools/tar.gz/main').read())).extractall('/tmp/code')
TOOLS = glob.glob('/tmp/code/*/tools')[0]
for mod, pip in (('optax', ['optax']), ('chess', ['chess']), ('torch', ['torch', '--index-url', 'https://download.pytorch.org/whl/cpu'])):
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pip, check=True)

# Donnees : tout le fichier, du premier au dernier octet
code, _ = lancer([sys.executable, '-u', TOOLS + '/hf_sources.py', '--url',
                  'https://huggingface.co/datasets/official-stockfish/master-binpacks/resolve/main/nodes5000pv2_UHO.binpack',
                  '--prefixe', 'n5', '--dest', '/tmp/donnees', '--octets', '40300000000', '--garde', '1.0', '--fixe', '240',
                  '--graine', '17', '--morceau', '4000000000', '--parts', '4'], 'menaces.log')
if code:
    sys.exit('preparation des donnees en echec')
import numpy as np
v = sorted(glob.glob('/tmp/donnees/*_p00_*.bin'))[0]
os.rename(v, '/tmp/donnees/z_val.bin')
np.fromfile('/tmp/donnees/z_val.bin', dtype='uint8').reshape(-1, 32)[:2_000_000].tofile('/tmp/donnees/z_val.bin')
print(f"donnees pretes : {time.time() - t0:.0f} s", flush=True)

VARIANTES = [('temoin', []),
             ('etroit_ennemis', ['--menaces', '1', '--menaces-largeur', '256', '--menaces-ennemis', '1'])]
COMMUN = [sys.executable, '-u', TOOLS + '/train_jax.py', '--stream', '/tmp/donnees/*.bin', '--hidden', '1024',
          '--buckets', 'mat4', '--out-buckets', '4', '--chunk-files', '10', '--epoch-files', '60',
          '--lr', '1e-3', '--val', '2000000', '--ft-dense', '1']
# 1. Vitesse de chaque variante (300 pas) : meme nombre de positions vues pour les deux,
#    fixe par la plus lente pour tenir dans la session (9 h au plus)
vitesse = {}
for nom, opts in VARIANTES:
    _, out = lancer(COMMUN + opts + ['--max-pas', '300'], 'menaces.log')
    m = re.search(r'= ([\d,]+) positions/s', out)
    vitesse[nom] = int(m.group(1).replace(',', '')) if m else 0
    print(f"VITESSE {nom} : {vitesse[nom]:,} positions/s", flush=True)
plus_lente = min(v for v in vitesse.values() if v > 0)
budget = int(3.4 * 3600)   # par variante
epoques = max(4, min(100, int(budget * plus_lente / 300e6)))
print(f"EPOQUES par variante : {epoques} (variante la plus lente : {plus_lente:,} positions/s)", flush=True)

# 2. Entrainements depuis zero
bilan = {}
for nom, opts in VARIANTES:
    _, out = lancer(COMMUN + opts + ['--epochs', str(epoques), '--limite-s', str(int(budget * 1.15)),
                                     '--out', f'{W}/{nom}.nnue', '--save-pt', f'{W}/{nom}.pt'], f'{nom}.log')
    cp = re.findall(r'erreur\s+([\d.]+) cp', out)
    bilan[nom] = cp[-1] if cp else '?'
    print(f"BILAN {nom} : {bilan[nom]} cp", flush=True)
print('RESUME :', bilan, f'{time.time() - t0:.0f} s', flush=True)
