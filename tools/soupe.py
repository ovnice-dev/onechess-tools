#!/usr/bin/env python3
"""Soupe de reseaux (20-11-2026) : moyenne des poids flottants de reseaux affines depuis le
meme point de depart (meme « bassin »), puis export .nnue. Aucun entrainement.

  python tools/soupe.py --sortie soupe.nnue a.pt b.pt c.pt [--poids 2 1 1]
"""
import argparse, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pts', nargs='+')
    ap.add_argument('--poids', nargs='*', type=float, default=None)
    ap.add_argument('--sortie', required=True)
    ap.add_argument('--buckets', default='mat4')
    ap.add_argument('--hidden', type=int, default=1024)
    ap.add_argument('--out-buckets', type=int, default=4)
    ap.add_argument('--empirique', type=int, default=20000,
                    help="plus grand accumulateur constate (le garde-fou de l'export) ; check_nnue verifie ensuite")
    a = ap.parse_args()
    w = a.poids or [1.0] * len(a.pts)
    assert len(w) == len(a.pts)
    etats = [torch.load(p, map_location='cpu', weights_only=False) for p in a.pts]
    cles = etats[0].keys()
    for e in etats[1:]:
        assert e.keys() == cles, 'reseaux de formes differentes'
    moy = {k: sum(wi * e[k].double() for wi, e in zip(w, etats)) / sum(w) for k in cles}
    fmap = T.FeatureMap(a.buckets, 0, 'cpu')
    model = T.Perspective(fmap.n_inputs, a.hidden, a.out_buckets, True)
    model.load_state_dict({k: v.float() for k, v in moy.items()})
    T.export(model, fmap, a.sortie, a.hidden, empirical=a.empirique)


if __name__ == '__main__':
    main()
