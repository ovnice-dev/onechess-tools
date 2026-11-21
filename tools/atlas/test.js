const fs = require('fs');
const NN = require('./core.js');
const buf = fs.readFileSync(__dirname + '/reseau.bin');
const net = NN.parse(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
console.log('hl', net.hl, 'nb', net.nb, 'obn', net.obn, 'material', net.material);
const pos = JSON.parse(fs.readFileSync(__dirname + '/positions.json')).slice(0, 40).map(p => p.fen);
const out = pos.map(f => NN.evaluer(net, NN.fen(f)).cp);
fs.writeFileSync(__dirname + '/js_evals.json', JSON.stringify({ fens: pos, cp: out }));
const t0 = Date.now();
const all = JSON.parse(fs.readFileSync(__dirname + '/positions.json')).map(p => p.fen);
const st = NN.statistiques(net, all);
console.log('stats 1500 positions en', Date.now() - t0, 'ms ; vus', Array.from(st.vus));
for (let e = 0; e < net.nb; e++) {
  let mort = 0; for (let j = 0; j < net.hl; j++) if (st.on[e][j] === 0) mort++;
  console.log('expert', e, 'jamais allumes', mort);
}
