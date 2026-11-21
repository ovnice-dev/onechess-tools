# Phase 10, reglage du « mini-LoRA » (21-11) : --rang-expert 32 --ema 0.9999 --perte-exp 2.5 : 8 experts depuis zero sur GPU T4, memes donnees, memes
# epoques, une seule difference : l'ecart de chaque expert au tronc commun est de rang 32.
# Petit calcul (~1,5 G positions vues) : c'est un filtre, pas un verdict ; si le faible rang gagne
# nettement, il devient une variante de l'etape « 8 experts » de l'empilement sur TPU.
import glob, io, os, subprocess, sys, tarfile, time, urllib.request
W = '/kaggle/working'
t0 = time.time()


def lancer(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(W + '/lora.log', 'a', buffering=1) as j:
        for l in p.stdout:
            print(l, end='', flush=True)
            j.write(l)
    return p.wait()


tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(
    'https://codeload.github.com/ovnice-dev/onechess-tools/tar.gz/main').read())).extractall('/tmp/code')
TOOLS = glob.glob('/tmp/code/*/tools')[0]
if lancer([sys.executable, '-u', TOOLS + '/hf_sources.py', '--url',
           'https://huggingface.co/datasets/official-stockfish/master-binpacks/resolve/main/nodes5000pv2_UHO.binpack',
           '--prefixe', 'n5', '--dest', '/tmp/donnees', '--octets', '7000000000', '--garde', '1.0', '--fixe', '240',
           '--graine', '23', '--morceau', '4000000000', '--parts', '4']):
    sys.exit('preparation des donnees en echec')
import numpy as np
v = sorted(glob.glob('/tmp/donnees/*_p00_*.bin'))[0]
os.rename(v, '/tmp/donnees/z_val.bin')
np.fromfile('/tmp/donnees/z_val.bin', dtype='uint8').reshape(-1, 32)[:5_000_000].tofile('/tmp/donnees/z_val.bin')
print(f"donnees pretes : {time.time() - t0:.0f} s", flush=True)
reste = int(4.75 * 3600 - (time.time() - t0))
code = lancer(['timeout', str(reste), sys.executable, '-u', TOOLS + '/train.py', '--stream', '/tmp/donnees/*.bin',
               '--chunk-files', '10', '--epoch-files', '60', '--hidden', '1024', '--buckets', 'mat8',
               '--out-buckets', '4', '--rang-expert', '32', '--ema', '0.9999', '--perte-exp', '2.5', '--epochs', '3', '--lr', '1e-3',
               '--out', W + '/lora_r32p.nnue', '--checkpoint', W + '/lora_r32p.ckpt', '--save-pt', W + '/lora_r32p.pt'])
print('code', code, f'{time.time() - t0:.0f} s', flush=True)
