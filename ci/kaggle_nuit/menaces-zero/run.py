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

# Sonde (18-11) : le disque AFFICHE (~1 To) n'est pas forcement le vrai : sur GPU, l'ecriture
# echouait vers 97 Go malgre 1 To affiche. On reserve vraiment la place, puis on la rend ; si
# elle ne tient pas, on prend moins d'octets du fichier au lieu de tomber en route.
os.makedirs('/tmp/donnees', exist_ok=True)
with open('/proc/meminfo') as f:
    print('MEMOIRE :', ' '.join(f.readline().split()[1:]), flush=True)


def tient(go):
    p = '/tmp/donnees/sonde.bin'
    try:
        fd = os.open(p, os.O_CREAT | os.O_WRONLY)
        try:
            os.posix_fallocate(fd, 0, int(go * 1e9))
            return True
        except OSError as e:
            if e.errno not in (95,):   # 95 : reservation non geree, on ecrit pour de vrai
                return False
            bloc = bytes(1 << 26)
            for _ in range(int(go * 1e9) >> 26):
                os.write(fd, bloc)
            return True
        finally:
            os.close(fd)
    except OSError:
        return False
    finally:
        if os.path.exists(p):
            os.remove(p)


GO_PAR_GO = 7.95   # disque occupe par Go de binpack (garde 1,0)
disque = next((g for g in (340, 250, 180, 120, 90) if tient(g)), 60)
octets = min(40_300_000_000, int(disque * 0.92 / GO_PAR_GO * 1e9))
print(f"DISQUE : {disque} Go reserves avec succes -> {octets / 1e9:.1f} Go du fichier", flush=True)

# Donnees : tout le fichier (ou ce qui tient sur le disque), du premier octet
code, _ = lancer([sys.executable, '-u', TOOLS + '/hf_sources.py', '--url',
                  'https://huggingface.co/datasets/official-stockfish/master-binpacks/resolve/main/nodes5000pv2_UHO.binpack',
                  '--prefixe', 'n5', '--dest', '/tmp/donnees', '--octets', str(octets), '--garde', '1.0', '--fixe', '240',
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
