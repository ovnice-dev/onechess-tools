# Phase 10, experience A (15-11) : les donnees sont-elles le verrou ?
# Deux reseaux identiques (1024, mat4, 4 sorties), entraines DEPUIS ZERO, avec le meme calcul
# (meme nombre de positions vues, meme programme du pas) ; une seule difference :
#   petit : 1/3 des donnees (~0,6 milliard de positions differentes), repassees 3 fois plus
#   grand : toutes les donnees (~1,8 milliard de positions differentes)
# Juge : l'erreur de validation (meme fichier), puis un match au temps entre les deux reseaux.
# Ne pas importer jax ici : le processus reserverait le TPU.
import glob, io, os, subprocess, sys, tarfile, time, urllib.request

W = '/kaggle/working'
t0 = time.time()


def lancer(cmd, journal='donnees.log'):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(W + '/' + journal, 'a', buffering=1) as j:
        for l in p.stdout:
            print(l, end='', flush=True)
            j.write(l)
    return p.wait()


_d = subprocess.run([sys.executable, '-c', 'import jax; print(jax.devices()[0].platform, jax.device_count())'],
                    capture_output=True, text=True).stdout.strip().splitlines()
print('machine :', _d, flush=True)
if not _d or not _d[-1].startswith('tpu'):
    sys.exit('PAS DE TPU : arret')

tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(
    'https://codeload.github.com/ovnice-dev/onechess-tools/tar.gz/main').read())).extractall('/tmp/code')
TOOLS = glob.glob('/tmp/code/*/tools')[0]
for mod, pip in (('optax', ['optax']), ('torch', ['torch', '--index-url', 'https://download.pytorch.org/whl/cpu'])):
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pip, check=True)

# 7 Go de nodes5000pv2_UHO (~1,8 milliard de positions apres filtres)
if lancer([sys.executable, '-u', TOOLS + '/hf_sources.py', '--url',
           'https://huggingface.co/datasets/official-stockfish/master-binpacks/resolve/main/nodes5000pv2_UHO.binpack',
           '--prefixe', 'n5', '--dest', '/tmp/tout', '--octets', '7000000000', '--garde', '1.0', '--fixe', '240',
           '--graine', '21', '--morceau', '4000000000', '--parts', '4']):
    sys.exit('preparation des donnees en echec')
import numpy as np
fichiers = sorted(glob.glob('/tmp/tout/*.bin'))
v = fichiers.pop(0)
np.fromfile(v, dtype='uint8').reshape(-1, 32)[:2_000_000].tofile('/tmp/val.bin')
os.remove(v)
# Le petit jeu : un fichier sur trois (liens, pas de copie)
os.makedirs('/tmp/petit', exist_ok=True)
for i, f in enumerate(fichiers):
    if i % 3 == 0:
        os.symlink(f, '/tmp/petit/' + os.path.basename(f))
for d in ('/tmp/tout', '/tmp/petit'):
    os.symlink('/tmp/val.bin', d + '/z_val.bin')  # validation commune, rangee en dernier
n_tout = sum(os.path.getsize(f) for f in fichiers) // 32
print(f"donnees : {n_tout:,} positions (grand), ~{n_tout // 3:,} (petit) ; {time.time() - t0:.0f} s", flush=True)

# Meme calcul pour les deux : 100 epoques de 60 fichiers (~30 milliards de positions vues), depuis zero
COMMUN = [sys.executable, '-u', TOOLS + '/train_jax.py', '--ft-dense', '1', '--hidden', '1024', '--buckets', 'mat4',
          '--out-buckets', '4', '--chunk-files', '10', '--epoch-files', '60', '--epochs', '100', '--lr', '1e-3',
          '--val', '2000000', '--limite-s', str(int(3.6 * 3600))]
bilan = {}
for nom in ('petit', 'grand'):
    src = '/tmp/petit/*.bin' if nom == 'petit' else '/tmp/tout/*.bin'
    out = W + f'/{nom}.nnue'
    lancer(COMMUN + ['--stream', src, '--out', out, '--save-pt', W + f'/{nom}.pt'], f'{nom}.log')
    j = open(W + f'/{nom}.log').read()
    import re
    cp = re.findall(r'erreur\s+([\d.]+) cp', j)
    bilan[nom] = cp[-1] if cp else '?'
    print(f"BILAN {nom} : {bilan[nom]} cp", flush=True)
print('RESUME :', bilan, f'{time.time() - t0:.0f} s', flush=True)
