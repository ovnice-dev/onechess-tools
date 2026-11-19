# Phase 10, empilement, etape 2 (prepare le 19-11) : la reference (gagnant de menaces-zero)
# + 8 experts par materiel, UN reseau neuf depuis zero, memes donnees (meme nombre d'octets du
# fichier, meme graine), meme nombre d'epoques que l'etape 1. Juge : match au temps contre le
# gagnant de l'etape 1. A REMPLIR apres le verdict de menaces-zero (lignes A_REMPLIR), puis
# deplacer ce dossier dans ci/kaggle_nuit/ (le relais le lance).
# Ne pas importer jax ici : le processus reserverait le TPU.
import glob, io, os, re, subprocess, sys, tarfile, time, urllib.request

W = '/kaggle/working'
BASE_OPTS = None      # A_REMPLIR : [] (temoin) ou ['--menaces', '1', '--menaces-largeur', '256', '--menaces-ennemis', '1']
EPOQUES = None        # A_REMPLIR : ligne « EPOQUES par variante » du journal de menaces-zero
OCTETS = None         # A_REMPLIR : octets du fichier lus par menaces-zero (ligne DISQUE)
assert BASE_OPTS is not None and EPOQUES and OCTETS, 'constantes de l etape 1 non remplies'
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
octets = OCTETS
if octets > disque * 0.92 / GO_PAR_GO * 1e9:
    sys.exit(f'DISQUE insuffisant pour les octets de l etape 1 ({disque} Go)')
print(f"DISQUE : {disque} Go reserves avec succes -> {octets / 1e9:.1f} Go du fichier", flush=True)

# Donnees : tout le fichier (ou ce qui tient sur le disque), du premier octet
code, _ = lancer([sys.executable, '-u', TOOLS + '/hf_sources.py', '--url',
                  'https://huggingface.co/datasets/official-stockfish/master-binpacks/resolve/main/nodes5000pv2_UHO.binpack',
                  '--prefixe', 'n5', '--dest', '/tmp/donnees', '--octets', str(octets), '--garde', '1.0', '--fixe', '240',
                  '--graine', '17', '--morceau', '4000000000', '--parts', '4'], 'mat8.log')
if code:
    sys.exit('preparation des donnees en echec')
import numpy as np
v = sorted(glob.glob('/tmp/donnees/*_p00_*.bin'))[0]
os.rename(v, '/tmp/donnees/z_val.bin')
np.fromfile('/tmp/donnees/z_val.bin', dtype='uint8').reshape(-1, 32)[:2_000_000].tofile('/tmp/donnees/z_val.bin')
print(f"donnees pretes : {time.time() - t0:.0f} s", flush=True)

VARIANTES = [('mat8', BASE_OPTS + ['--buckets', 'mat8'])]
COMMUN = [sys.executable, '-u', TOOLS + '/train_jax.py', '--stream', '/tmp/donnees/*.bin', '--hidden', '1024',
          '--out-buckets', '4', '--chunk-files', '10', '--epoch-files', '60',
          '--lr', '1e-3', '--val', '2000000', '--ft-dense', '1']
budget = int(6.6 * 3600)   # une seule variante ; limite x1,15 = 7 h 35, + preparation des donnees < 9 h
epoques = EPOQUES
print(f"EPOQUES : {epoques} (comme l'etape 1)", flush=True)

# 2. Entrainements depuis zero
bilan = {}
for nom, opts in VARIANTES:
    _, out = lancer(COMMUN + opts + ['--epochs', str(epoques), '--limite-s', str(int(budget * 1.15)),
                                     '--out', f'{W}/{nom}.nnue', '--save-pt', f'{W}/{nom}.pt'], f'{nom}.log')
    cp = re.findall(r'erreur\s+([\d.]+) cp', out)
    bilan[nom] = cp[-1] if cp else '?'
    print(f"BILAN {nom} : {bilan[nom]} cp", flush=True)
print('RESUME :', bilan, f'{time.time() - t0:.0f} s', flush=True)
