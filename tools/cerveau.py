#!/usr/bin/env python3
"""Regarder a l'interieur du reseau (20-11-2026) : que font les neurones de la 1re couche ?

Pour chaque neurone de l'accumulateur et chaque expert : s'allume-t-il (activation > 0) ?
est-il sature (>= QA) ? quelle part de la sortie porte-t-il ? Un neurone qui ne s'allume
jamais (ou dont le poids de sortie est nul) est de la capacite perdue.

  python tools/cerveau.py --reseau engine/bin/onechess.nnue --positions _kaggle/sorties/autopsie-2/coups.jsonl
"""
import argparse, json, os, sys
import numpy as np
import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_nnue as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reseau', required=True)
    ap.add_argument('--positions', required=True, help='fichier jsonl avec un champ fen (coups de l\'autopsie)')
    ap.add_argument('--max', type=int, default=20000)
    a = ap.parse_args()
    net = C.Net(a.reseau)
    if net.hidden or net.fullthr:
        sys.exit('reseau a couches cachees ou menaces completes : non gere ici')
    H, QA = net.hl, C.QA
    fens = [json.loads(l)['fen'] for l in open(a.positions) if l.strip()][:a.max]
    nb = net.nb
    allume = np.zeros((nb, H), np.int64)      # nombre de fois > 0
    sature = np.zeros((nb, H), np.int64)      # nombre de fois >= QA
    contrib = np.zeros((nb, H), np.float64)   # somme |activation * poids de sortie| (cote trait)
    vus = np.zeros(nb, np.int64)
    for fen in fens:
        b = chess.Board(fen)
        e = min(max((len(b.piece_map()) - 2) * nb // 32, 0), nb - 1)
        ob = (len(b.piece_map()) - 2) // (32 // net.ob_n)
        for persp in (chess.WHITE, chess.BLACK):
            acc = net.fb.copy()
            for f in net.features(b, persp):
                acc = acc + net.fw[f]
            if net.pairs:
                for f in net.pair_features(b, persp):
                    acc = acc + net.pw[f]
            acc = acc.astype(np.int16).astype(np.int64)
            act = np.clip(acc, 0, QA)
            allume[e] += act > 0
            sature[e] += acc >= QA
            w = net.ow[ob][:H] if persp == b.turn else net.ow[ob][H:]
            contrib[e] += np.abs(act * w)
        vus[e] += 2
    print(f'reseau {a.reseau} : {H} neurones, {nb} experts, {len(fens)} positions')
    print('\n| Expert | Positions | Jamais allumés | Allumés < 1 % | Toujours saturés | Allumés en moyenne | Neurones portant 50 % de la sortie |')
    print('|---|---|---|---|---|---|---|')
    for e in range(nb):
        if not vus[e]:
            continue
        p_on = allume[e] / vus[e]
        tri = np.sort(contrib[e])[::-1]
        n50 = int(np.searchsorted(np.cumsum(tri), tri.sum() * 0.5)) + 1
        print(f'| {e} | {vus[e] // 2} | {int((allume[e] == 0).sum())} | {int((p_on < 0.01).sum())} | '
              f'{int((sature[e] == vus[e]).sum())} | {p_on.mean():.1%} | {n50} |')
    tous = allume.sum(0)
    print(f'\nNeurones jamais allumés dans AUCUN expert : {int((tous == 0).sum())} sur {H}')
    pw = np.abs(net.ow).max(0)
    print(f'Neurones au poids de sortie nul dans tous les seaux : {int(((pw[:H] == 0) & (pw[H:] == 0)).sum())}')
    # neurones presque identiques (redondants) : correlation des colonnes de poids de l'expert le plus vu
    e = int(np.argmax(vus))
    W = net.fw[e * C.BASE_INPUTS:(e + 1) * C.BASE_INPUTS].astype(np.float64)
    W -= W.mean(0)
    nrm = np.linalg.norm(W, axis=0) + 1e-9
    corr = (W.T @ W) / np.outer(nrm, nrm)
    np.fill_diagonal(corr, 0)
    print(f'Expert {e} : paires de neurones à poids presque identiques (corrélation > 0,95) : {int((corr > 0.95).sum() // 2)}')


if __name__ == '__main__':
    main()
