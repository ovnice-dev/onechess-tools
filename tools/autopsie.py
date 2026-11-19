#!/usr/bin/env python3
"""Autopsie d'Onechess (19-11-2026) : ou le moteur perd-il ses demi-points ?

Trois etapes, pensees pour une nuit sur un processeur Kaggle :

  jouer    : parties a la pendule contre un adversaire (Stockfish par defaut), en
             enregistrant TOUT ce que dit Onechess a chaque iteration (profondeur,
             noeuds, vitesse, score, coup en tete, table) et la pendule.
  analyser : chaque coup d'Onechess est rejuge par l'arbitre (Stockfish, en
             profondeur). Les erreurs sont rejouees par Onechess avec 8 fois plus
             de noeuds : s'il corrige, c'etait la recherche (ou le temps) ; s'il
             persiste, c'etait l'evaluation (le reseau).
  rapport  : ou partent les points perdus (phase, materiel, cause, situation),
             comment le temps est depense, vitesse et profondeur reelles.

Un diagnostic dit OU chercher ; seul un SPRT dit si une correction gagne.

  python tools/autopsie.py jouer --moteur onechess --reseau onechess.nnue --adversaire stockfish \
         --parties 100 --tc 40/60 --paralleles 2 --sortie autopsie/parties.jsonl
  python tools/autopsie.py analyser --entree autopsie/parties.jsonl --arbitre stockfish --sortie autopsie/coups.jsonl
  python tools/autopsie.py rapport --parties autopsie/parties.jsonl --coups autopsie/coups.jsonl --sortie autopsie/rapport.md
"""
import argparse, json, math, os, random, statistics, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import chess
import chess.engine

ECHELLE = 400.0   # cp -> score attendu : 1 / (1 + 10^(-cp/400)) (echelle Elo)


def attendu(cp):
    return 1.0 / (1.0 + 10 ** (-max(-3000, min(3000, cp)) / ECHELLE))


def cp_de(score, camp):
    s = score.pov(camp)
    if s.is_mate():
        return 3000 if s.mate() > 0 else -3000
    return s.score()


# ---------------------------------------------------------------- cadences
def lire_cadence(t):
    """'10+0.1' (mort subite + increment) ou '40/60' (40 coups en 60 s, repete)."""
    if '/' in t:
        n, b = t.split('/')
        return float(b), 0.0, int(n)
    b, _, i = t.partition('+')
    return float(b), float(i or 0), 0


# ------------------------------------------------------------------ jouer
def ouvrir(chemin, options):
    e = chess.engine.SimpleEngine.popen_uci(chemin)
    for k, v in options.items():
        try:
            e.configure({k: v})
        except Exception:
            pass
    return e


def une_partie(args):
    """Une partie ; Onechess a la couleur 'couleur'. Rend un dict complet."""
    num, fen, couleur, cfg = args
    one = ouvrir(cfg['moteur'], {'Hash': cfg['hash'], 'Threads': 1, 'EvalFile': cfg['reseau']} if cfg['reseau']
                 else {'Hash': cfg['hash'], 'Threads': 1})
    adv = ouvrir(cfg['adversaire'], {'Hash': cfg['hash'], 'Threads': 1})
    base, inc, mtg = lire_cadence(cfg['tc'])
    board = chess.Board(fen)
    pendule = {chess.WHITE: base, chess.BLACK: base}
    joues = {chess.WHITE: 0, chess.BLACK: 0}
    coups, resultat, raison = [], None, ''
    try:
        while True:
            if board.is_game_over(claim_draw=True):
                resultat, raison = board.result(claim_draw=True), 'regle'
                break
            if board.fullmove_number > 220:
                resultat, raison = '1/2-1/2', 'limite'
                break
            camp = board.turn
            limite = chess.engine.Limit(white_clock=pendule[chess.WHITE], black_clock=pendule[chess.BLACK],
                                        white_inc=inc, black_inc=inc,
                                        remaining_moves=(mtg - joues[camp] % mtg) if mtg else None)
            avant = pendule[camp]
            fen_avant = board.fen()
            iterations = []
            t0 = time.perf_counter()
            if camp == couleur:
                # chaque message « info » d'Onechess est garde : c'est la vie du moteur pendant le coup
                with one.analysis(board, limite, game=num) as an:
                    for info in an:
                        if 'depth' not in info or 'score' not in info:
                            continue
                        pv = info.get('pv') or []
                        iterations.append({
                            'd': info.get('depth'), 'sd': info.get('seldepth'), 'n': info.get('nodes'),
                            'nps': info.get('nps'), 't': info.get('time'), 'hf': info.get('hashfull'),
                            'cp': cp_de(info['score'], camp), 'c': pv[0].uci() if pv else None,
                            'b': info.get('lowerbound', False) or info.get('upperbound', False)})
                    meilleur = an.wait()
                coup = meilleur.move
            else:
                coup = adv.play(board, limite, game=num + 100000).move
            ecoule = time.perf_counter() - t0
            pendule[camp] -= ecoule
            if pendule[camp] < -0.5:
                resultat = '0-1' if camp == chess.WHITE else '1-0'
                raison = 'temps'
                break
            pendule[camp] = max(0.0, pendule[camp]) + inc
            joues[camp] += 1
            if mtg and joues[camp] % mtg == 0:
                pendule[camp] += base
            if coup is None or coup not in board.legal_moves:
                resultat = '0-1' if camp == chess.WHITE else '1-0'
                raison = 'coup illegal'
                break
            coups.append({'fen': fen_avant, 'coup': coup.uci(), 'one': camp == couleur, 'pendule': round(avant, 3),
                          'ecoule': round(ecoule, 3), 'it': iterations})
            board.push(coup)
    finally:
        for e in (one, adv):
            try:
                e.quit()
            except Exception:
                pass
    score_one = {'1-0': 1.0, '0-1': 0.0}.get(resultat, 0.5)
    if couleur == chess.BLACK:
        score_one = 1.0 - score_one
    return {'num': num, 'ouverture': fen, 'one_blancs': couleur == chess.WHITE, 'tc': cfg['tc'],
            'resultat': resultat, 'raison': raison, 'score_one': score_one, 'coups': coups}


def cmd_jouer(a):
    fens = [l.split(';')[0].strip() for l in open(a.ouvertures) if l.strip()]
    random.Random(a.graine).shuffle(fens)
    cfg = {'moteur': a.moteur, 'reseau': a.reseau, 'adversaire': a.adversaire, 'tc': a.tc, 'hash': a.hash}
    taches = []
    for i in range(a.parties):
        fen = fens[i // 2 % len(fens)]
        if len(fen.split()) == 4:
            fen += ' 0 1'
        taches.append((i, fen, chess.WHITE if i % 2 == 0 else chess.BLACK, cfg))
    os.makedirs(os.path.dirname(a.sortie) or '.', exist_ok=True)
    deja = set()
    if os.path.exists(a.sortie):
        deja = {json.loads(l)['num'] for l in open(a.sortie) if l.strip()}
    taches = [t for t in taches if t[0] not in deja]
    fin = time.time() + a.limite_s if a.limite_s else None
    pts = n = 0
    with ProcessPoolExecutor(a.paralleles) as ex, open(a.sortie, 'a') as f:
        en_cours = {}
        it = iter(taches)
        for _ in range(a.paralleles):
            t = next(it, None)
            if t:
                en_cours[ex.submit(une_partie, t)] = t
        while en_cours:
            fut = next(as_completed(en_cours))
            en_cours.pop(fut)
            try:
                r = fut.result()
                f.write(json.dumps(r) + '\n')
                f.flush()
                pts += r['score_one']
                n += 1
                print(f"partie {r['num']:4d} : {r['resultat']:7s} ({r['raison']}), Onechess {pts}/{n}", flush=True)
            except Exception as e:
                print('partie en echec :', e, flush=True)
            if fin and time.time() > fin:
                continue
            t = next(it, None)
            if t:
                en_cours[ex.submit(une_partie, t)] = t


# --------------------------------------------------------------- analyser
ARBITRE = {}


def init_arbitre(chemin, moteur, reseau, hash_mb):
    ARBITRE['sf'] = ouvrir(chemin, {'Hash': hash_mb, 'Threads': 1})
    ARBITRE['one'] = ouvrir(moteur, {'Hash': hash_mb, 'Threads': 1, 'EvalFile': reseau} if reseau
                            else {'Hash': hash_mb, 'Threads': 1})


def juger(tache):
    """Un coup d'Onechess : perte en score attendu selon l'arbitre, puis cause."""
    fen, coup, noeuds_arbitre, noeuds_one, info = tache
    sf, one = ARBITRE['sf'], ARBITRE['one']
    b = chess.Board(fen)
    camp = b.turn
    m = chess.Move.from_uci(coup)
    r = sf.analyse(b, chess.engine.Limit(nodes=noeuds_arbitre), multipv=2)
    meilleur = r[0]['pv'][0]
    cp_best = cp_de(r[0]['score'], camp)
    second = cp_de(r[1]['score'], camp) if len(r) > 1 else None
    if m == meilleur:
        cp_joue = cp_best
    else:
        rj = sf.analyse(b, chess.engine.Limit(nodes=noeuds_arbitre), root_moves=[m])
        cp_joue = cp_de(rj['score'], camp)
    perte = max(0.0, attendu(cp_best) - attendu(cp_joue))
    out = dict(info)
    out.update({'meilleur': meilleur.uci(), 'cp_arbitre': cp_best, 'cp_joue': cp_joue, 'perte': round(perte, 4),
                'unique': second is not None and attendu(cp_best) - attendu(second) > 0.15})
    if perte >= 0.05:
        # meme position, 8 fois plus de noeuds que pendant la partie
        rr = one.analyse(b, chess.engine.Limit(nodes=min(12_000_000, max(200000, noeuds_one * 8))))
        c2 = rr['pv'][0] if rr.get('pv') else None
        if c2 is not None and c2 != m:
            rj2 = sf.analyse(b, chess.engine.Limit(nodes=noeuds_arbitre), root_moves=[c2])
            perte2 = max(0.0, attendu(cp_best) - attendu(cp_de(rj2['score'], camp)))
        else:
            perte2 = perte
        out['recherche_x8'] = c2.uci() if c2 else None
        out['perte_x8'] = round(perte2, 4)
        out['cause'] = 'recherche' if perte2 < perte * 0.5 else 'evaluation'
    return out


def traits(b):
    """Ce qui decrit la situation du coup : phase, materiel, roi, tactique."""
    camp = b.turn
    n = len(b.piece_map())
    mat = {p: len(b.pieces(p, camp)) + len(b.pieces(p, not camp)) for p in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)}
    if mat[chess.QUEEN] + mat[chess.ROOK] + mat[chess.BISHOP] + mat[chess.KNIGHT] == 0:
        famille = 'pions'
    elif mat[chess.QUEEN] == 0 and mat[chess.ROOK] == 0:
        famille = 'pieces mineures'
    elif mat[chess.QUEEN] == 0:
        famille = 'tours' + (' + mineures' if mat[chess.BISHOP] + mat[chess.KNIGHT] else '')
    else:
        famille = 'dames'
    roi = b.king(camp)
    zone = chess.SquareSet(chess.BB_KING_ATTACKS[roi]) | chess.SquareSet.from_square(roi)
    attaques_roi = sum(len(b.attackers(not camp, s)) for s in zone)
    prises = sum(1 for m in b.legal_moves if b.is_capture(m))
    en_prise = sum(1 for s, p in b.piece_map().items() if p.color == camp and p.piece_type != chess.KING
                   and b.is_attacked_by(not camp, s) and not b.is_attacked_by(camp, s))
    return {'pieces': n, 'expert': min(3, max(0, (n - 2) * 4 // 32)), 'famille': famille,
            'echec': b.is_check(), 'attaques_roi': attaques_roi, 'prises': prises, 'en_prise': en_prise,
            'coup_num': b.fullmove_number}


def cmd_analyser(a):
    taches = []
    for l in open(a.entree):
        if not l.strip():
            continue
        g = json.loads(l)
        for i, c in enumerate(g['coups']):
            if not c['one']:
                continue
            b = chess.Board(c['fen'])
            if len(b.piece_map()) <= 5:     # tables de finales : rien a apprendre
                continue
            it = c['it']
            dern = it[-1] if it else {}
            changements = sum(1 for x, y in zip(it, it[1:]) if x['c'] != y['c'])
            info = {'partie': g['num'], 'ply': i, 'fen': c['fen'], 'coup': c['coup'], 'score_one': g['score_one'],
                    'pendule': c['pendule'], 'ecoule': c['ecoule'], 'prof': dern.get('d'), 'noeuds': dern.get('n'),
                    'nps': dern.get('nps'), 'cp_one': dern.get('cp'), 'changements': changements,
                    'hashfull': dern.get('hf')}
            info.update(traits(b))
            taches.append((c['fen'], c['coup'], a.noeuds, dern.get('n') or 100000, info))
    # ordre melange : si la limite de temps coupe, l'echantillon juge reste representatif
    random.Random(7).shuffle(taches)
    print(f'{len(taches)} coups a juger', flush=True)
    os.makedirs(os.path.dirname(a.sortie) or '.', exist_ok=True)
    fin = time.time() + a.limite_s if a.limite_s else None
    t0, k = time.time(), 0
    ex = ProcessPoolExecutor(a.paralleles, initializer=init_arbitre, initargs=(a.arbitre, a.moteur, a.reseau, a.hash))
    with open(a.sortie, 'w') as f:
        it = iter(taches)
        en_cours = set()
        for _ in range(a.paralleles * 2):
            t = next(it, None)
            if t:
                en_cours.add(ex.submit(juger, t))
        while en_cours:
            fut = next(as_completed(en_cours))
            en_cours.discard(fut)
            try:
                f.write(json.dumps(fut.result()) + '\n')
                f.flush()
            except Exception as e:
                print('coup en echec :', e, flush=True)
            k += 1
            if k % 100 == 0:
                print(f'  {k}/{len(taches)} ({(time.time() - t0) / k * a.paralleles:.1f} s par coup et par processus)', flush=True)
            if fin and time.time() > fin:
                continue
            t = next(it, None)
            if t:
                en_cours.add(ex.submit(juger, t))
    # les processus gardent leurs deux moteurs ouverts : un arret ordinaire attendrait
    # sans fin la fin de leurs fils de communication ; on les termine (moteurs compris)
    for p in list(getattr(ex, '_processes', {}).values()):
        p.terminate()
    ex.shutdown(wait=False, cancel_futures=True)
    print(f'{k} coups juges en {time.time() - t0:.0f} s', flush=True)


# ---------------------------------------------------------------- rapport
def tableau(titre, groupes, total_coups, total_parties):
    lignes = [f'### {titre}', '', '| Groupe | Coups | Perte moyenne | Erreurs (≥ 0,05) | Points perdus par partie |',
              '|---|---|---|---|---|']
    for k, v in sorted(groupes.items(), key=lambda kv: -sum(x['perte'] for x in kv[1])):
        p = sum(x['perte'] for x in v)
        err = sum(1 for x in v if x['perte'] >= 0.05)
        lignes.append(f'| {k} | {len(v)} | {p / len(v):.4f} | {err} | {p / total_parties:.3f} |')
    return '\n'.join(lignes) + '\n'


def cmd_rapport(a):
    parties = [json.loads(l) for l in open(a.parties) if l.strip()]
    coups = [json.loads(l) for l in open(a.coups) if l.strip()]
    np_ = len(parties)
    score = sum(g['score_one'] for g in parties) / max(1, np_)
    elo = -400 * math.log10(1 / score - 1) if 0 < score < 1 else float('nan')
    r = [f'# Autopsie d\'Onechess', '',
         f'{np_} parties ({parties[0]["tc"] if parties else "?"}), score {score:.3f} (≈ {elo:+.0f} Elo contre l\'adversaire), '
         f'{len(coups)} coups jugés par l\'arbitre.', '']
    raisons = defaultdict(int)
    for g in parties:
        raisons[g['raison']] += 1
    r.append('Fins de partie : ' + ', '.join(f'{k} {v}' for k, v in raisons.items()) + '\n')
    perte_tot = sum(c['perte'] for c in coups)
    r.append(f'**Perte totale : {perte_tot / max(1, np_):.3f} point par partie** (somme des baisses de score attendu '
             f'selon l\'arbitre ; ce n\'est pas le résultat, c\'est la qualité des coups).\n')

    def grouper(cle):
        g = defaultdict(list)
        for c in coups:
            g[cle(c)].append(c)
        return g

    r.append(tableau('Par cause (erreurs rejouées avec 8 fois plus de nœuds)',
                     grouper(lambda c: c.get('cause', 'pas une erreur')), len(coups), np_))
    r.append(tableau('Par famille de matériel', grouper(lambda c: c['famille']), len(coups), np_))
    r.append(tableau('Par expert du réseau (nombre de pièces)', grouper(lambda c: ['expert 0 (2-9 pièces)', 'expert 1 (10-17 pièces)', 'expert 2 (18-25 pièces)', 'expert 3 (26-32 pièces)'][c['expert']]), len(coups), np_))
    r.append(tableau('Par moment de la partie', grouper(lambda c: f"coups {min(c['coup_num'] // 10 * 10, 80)}-{min(c['coup_num'] // 10 * 10, 80) + 9}"), len(coups), np_))
    r.append(tableau('Par roi sous pression (attaques adverses sur la zone du roi)',
                     grouper(lambda c: '0' if c['attaques_roi'] == 0 else '1-3' if c['attaques_roi'] <= 3 else '4+'), len(coups), np_))
    r.append(tableau('Par tension (pièces à nous en prise)', grouper(lambda c: str(min(c['en_prise'], 2)) + ('+' if c['en_prise'] >= 2 else '')), len(coups), np_))
    r.append(tableau('Coup unique (un seul coup ne perd pas) ou non', grouper(lambda c: 'unique' if c['unique'] else 'plusieurs bons coups'), len(coups), np_))
    r.append(tableau('Par pendule (temps restant avant le coup)', grouper(lambda c: '< 2 s' if c['pendule'] < 2 else '2-10 s' if c['pendule'] < 10 else '10-30 s' if c['pendule'] < 30 else '≥ 30 s'), len(coups), np_))
    r.append(tableau('Par hésitation (changements de coup en tête pendant la recherche)',
                     grouper(lambda c: str(min(c['changements'], 3)) + ('+' if c['changements'] >= 3 else '')), len(coups), np_))

    # evaluation : notre score contre celui de l'arbitre, par expert
    r.append('### Écart d\'évaluation (score d\'Onechess en fin de recherche contre l\'arbitre)\n')
    r.append('| Expert | Positions | Écart absolu moyen (cp) | Onechess trop optimiste (part) |\n|---|---|---|---|')
    ex = defaultdict(list)
    for c in coups:
        if c.get('cp_one') is not None and abs(c['cp_arbitre']) < 1000 and abs(c['cp_one']) < 1000:
            ex[c['expert']].append(c['cp_one'] - c['cp_arbitre'])
    for k in sorted(ex):
        v = ex[k]
        r.append(f'| {k} | {len(v)} | {statistics.mean(abs(x) for x in v):.1f} | {sum(1 for x in v if x > 0) / len(v):.0%} |')
    r.append('')

    # temps : depense par numero de coup
    r.append('### Dépense du temps (Onechess)\n')
    r.append('| Coups | Temps moyen par coup (s) | Part de la pendule utilisée | Profondeur moyenne | Vitesse (k nœuds/s) |\n|---|---|---|---|---|')
    tps = defaultdict(list)
    for g in parties:
        for c in g['coups']:
            if c['one']:
                b = chess.Board(c['fen'])
                k = min(b.fullmove_number // 10 * 10, 80)
                d = c['it'][-1] if c['it'] else {}
                tps[k].append((c['ecoule'], c['ecoule'] / max(0.01, c['pendule']), d.get('d') or 0, d.get('nps') or 0))
    for k in sorted(tps):
        v = tps[k]
        r.append(f'| {k}-{k + 9} | {statistics.mean(x[0] for x in v):.2f} | {statistics.mean(x[1] for x in v):.1%} | '
                 f'{statistics.mean(x[2] for x in v):.1f} | {statistics.mean(x[3] for x in v) / 1000:.0f} |')
    r.append('')

    # les pires erreurs, pour les regarder a la main
    r.append('### Les 25 plus grosses erreurs\n')
    r.append('| Partie | Coup | Joué | Meilleur | Perte | Cause | Famille | Pendule (s) | Profondeur |\n|---|---|---|---|---|---|---|---|---|')
    for c in sorted(coups, key=lambda c: -c['perte'])[:25]:
        r.append(f"| {c['partie']} | {c['coup_num']} | {c['coup']} | {c['meilleur']} | {c['perte']:.2f} | {c.get('cause', '')} | "
                 f"{c['famille']} | {c['pendule']:.1f} | {c['prof']} |")
    r.append('\nPositions complètes (FEN) dans le fichier des coups.\n')
    os.makedirs(os.path.dirname(a.sortie) or '.', exist_ok=True)
    open(a.sortie, 'w', encoding='utf-8').write('\n'.join(r))
    print('\n'.join(r[:12]))


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    j = sp.add_parser('jouer')
    j.add_argument('--moteur', required=True)
    j.add_argument('--reseau', default='')
    j.add_argument('--adversaire', required=True)
    j.add_argument('--ouvertures', default='data/uho_085.epd')
    j.add_argument('--parties', type=int, default=100)
    j.add_argument('--tc', default='40/60')
    j.add_argument('--hash', type=int, default=64)
    j.add_argument('--paralleles', type=int, default=2)
    j.add_argument('--graine', type=int, default=1)
    j.add_argument('--limite-s', type=int, default=0, help='plus de nouvelle partie apres N secondes')
    j.add_argument('--sortie', required=True)
    n = sp.add_parser('analyser')
    n.add_argument('--entree', required=True)
    n.add_argument('--arbitre', required=True)
    n.add_argument('--moteur', required=True)
    n.add_argument('--reseau', default='')
    n.add_argument('--noeuds', type=int, default=2_000_000)
    n.add_argument('--hash', type=int, default=64)
    n.add_argument('--paralleles', type=int, default=4)
    n.add_argument('--limite-s', type=int, default=0, help='plus de nouveau coup apres N secondes')
    n.add_argument('--sortie', required=True)
    r = sp.add_parser('rapport')
    r.add_argument('--parties', required=True)
    r.add_argument('--coups', required=True)
    r.add_argument('--sortie', required=True)
    a = ap.parse_args()
    {'jouer': cmd_jouer, 'analyser': cmd_analyser, 'rapport': cmd_rapport}[a.cmd](a)


if __name__ == '__main__':
    main()
