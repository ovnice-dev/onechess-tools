// Coeur de l'Atlas NNUE : lecture d'un reseau Onechess (.nnue), positions, accumulateur, evaluation.
const NN = (() => {
  const MAGIC3 = 0x334E4E4F, MAGIC4 = 0x344E4E4F, BASE = 768, QA = 255, QB = 64, SCALE = 400;
  const FLAGS = { 256: 'paires de pions', 512: 'menaces', 1024: 'couches cachées', 2048: 'produit par paires', 4096: 'menaces complètes' };
  const TYPES = { p: 1, n: 2, b: 3, r: 4, q: 5, k: 6 };

  function parse(buf) {
    const dv = new DataView(buf);
    const magic = dv.getUint32(0, true);
    if (magic !== MAGIC3 && magic !== MAGIC4) throw new Error("Ce fichier n'est pas un réseau Onechess (.nnue).");
    const hl = dv.getUint32(4, true), nb = dv.getUint32(8, true), mirror = dv.getUint32(12, true), obn = dv.getUint32(16, true);
    let off = 20, act = 0;
    if (magic === MAGIC4) { act = dv.getUint32(20, true); off = 24; }
    const extras = Object.keys(FLAGS).filter(f => act & Number(f)).map(f => FLAGS[f]);
    if (extras.length) throw new Error('Réseau avec ' + extras.join(', ') + " : l'Atlas ne lit que les réseaux simples pour l'instant.");
    const layout = new Uint8Array(buf.slice(off, off + 64)); off += 64;
    const nfeat = nb * BASE;
    const fw = new Int16Array(buf.slice(off, off + nfeat * hl * 2)); off += nfeat * hl * 2;
    const fb = new Int16Array(buf.slice(off, off + hl * 2)); off += hl * 2;
    const ow = new Int16Array(buf.slice(off, off + obn * 2 * hl * 2)); off += obn * 2 * hl * 2;
    const ob = []; for (let i = 0; i < obn; i++) ob.push(dv.getInt32(off + 4 * i, true));
    off += 4 * obn;
    if (off !== buf.byteLength) throw new Error(`Taille inattendue (${buf.byteLength} octets, ${off} attendus) : format non reconnu.`);
    return { hl, nb, obn, mirror: mirror === 1, material: mirror === 2, screlu: (act & 255) === 1, layout, fw, fb, ow, ob, taille: buf.byteLength };
  }

  function fen(f) {   // cases a1 = 0 ... h8 = 63
    const [placement, trait] = f.split(' ');
    const pieces = [];
    let r = 7, c = 0;
    for (const ch of placement) {
      if (ch === '/') { r--; c = 0; continue; }
      if (/\d/.test(ch)) { c += Number(ch); continue; }
      const blanc = ch === ch.toUpperCase();
      pieces.push({ sq: r * 8 + c, type: TYPES[ch.toLowerCase()], blanc, ch });
      c++;
    }
    return { pieces, blancs: trait === 'w', n: pieces.length };
  }

  // expert (bucket) et miroir de la perspective
  function bucket(net, pos, blanc) {
    if (net.material) return { e: Math.min(Math.max(Math.floor((pos.n - 2) * net.nb / 32), 0), net.nb - 1), flip: 0 };
    const k = pos.pieces.find(p => p.type === 6 && p.blanc === blanc);
    let rel = blanc ? k.sq : k.sq ^ 56;
    const flip = net.mirror && (rel & 7) >= 4 ? 7 : 0;
    rel ^= flip;
    return { e: net.layout[rel], flip };
  }

  function features(net, pos, blanc) {
    const { e, flip } = bucket(net, pos, blanc);
    const out = [];
    for (const p of pos.pieces) {
      const pidx = (p.blanc !== blanc ? 6 : 0) + (p.type - 1);
      const rel = (blanc ? p.sq : p.sq ^ 56) ^ flip;
      out.push(e * BASE + pidx * 64 + rel);
    }
    return { e, idx: out };
  }

  function accumulateur(net, pos, blanc) {
    const { e, idx } = features(net, pos, blanc);
    const H = net.hl, a = new Int32Array(H);
    for (let j = 0; j < H; j++) a[j] = net.fb[j];
    for (const f of idx) { const o = f * H; for (let j = 0; j < H; j++) a[j] += net.fw[o + j]; }
    for (let j = 0; j < H; j++) a[j] = (a[j] << 16) >> 16;   // le moteur garde l'accumulateur en int16
    return { e, a };
  }

  const tdiv = (a, b) => Math.trunc(a / b);

  // evaluation du point de vue du trait (cp), activations des deux perspectives
  function evaluer(net, pos) {
    const H = net.hl;
    const W = accumulateur(net, pos, true), B = accumulateur(net, pos, false);
    const us = pos.blancs ? W : B, them = pos.blancs ? B : W;
    const b = Math.min(Math.floor((pos.n - 2) / (32 / net.obn)), net.obn - 1);
    const w = net.ow, o = b * 2 * H;
    let tot = 0;
    const actU = new Uint16Array(H), actT = new Uint16Array(H);
    for (let j = 0; j < H; j++) {
      let u = Math.min(Math.max(us.a[j], 0), QA), t = Math.min(Math.max(them.a[j], 0), QA);
      actU[j] = u; actT[j] = t;
      if (net.screlu) { u *= u; t *= t; }
      tot += u * w[o + j] + t * w[o + H + j];
    }
    if (net.screlu) tot = tdiv(tot, QA);
    const cp = tdiv((tot + net.ob[b]) * SCALE, QA * QB);
    return { cp: Math.max(-12000, Math.min(12000, cp)), seau: b, expert: us.e, actU, actT, eU: us.e, eT: them.e };
  }

  // statistiques par expert et par neurone sur un echantillon de positions
  function statistiques(net, fens) {
    const H = net.hl, nb = net.nb;
    const on = Array.from({ length: nb }, () => new Uint32Array(H));
    const sat = Array.from({ length: nb }, () => new Uint32Array(H));
    const contrib = Array.from({ length: nb }, () => new Float64Array(H));
    const vus = new Uint32Array(nb);
    const evals = [];
    for (const f of fens) {
      const pos = fen(f);
      const r = evaluer(net, pos);
      evals.push(r.cp);
      const o = r.seau * 2 * H;
      for (const [act, e, side] of [[r.actU, r.eU, 0], [r.actT, r.eT, H]]) {
        vus[e]++;
        const ON = on[e], SA = sat[e], CO = contrib[e];
        for (let j = 0; j < H; j++) {
          const v = act[j];
          if (v > 0) { ON[j]++; CO[j] += v * Math.abs(net.ow[o + side + j]); if (v >= QA) SA[j]++; }
        }
      }
    }
    return { on, sat, contrib, vus, evals };
  }

  return { parse, fen, features, accumulateur, evaluer, statistiques, BASE, QA };
})();
if (typeof module !== 'undefined') module.exports = NN;
