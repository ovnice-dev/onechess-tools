#!/usr/bin/env bash
# Binaires de diffusion d'Onechess (Windows, statiques, Clang + PGO depuis le 18-11 ; MODE_DIFF=pgo pour GCC), un par famille de
# processeurs, dans engine/bin/diffusion/ :
#   onechess-bmi2.exe   AVX2 + PEXT rapide (Intel depuis Haswell, AMD Zen 3 et suivants)
#   onechess-avx2.exe   AVX2 sans PEXT (AMD Zen 1 et 2, ou PEXT est tres lent)
#   onechess-sse41.exe  sans AVX2 (vieux processeurs ; reseau en scalaire, lent)
# Chaque binaire doit donner la signature de reference (ci/signature.txt).
#   bash tools/diffusion.sh
set -e
cd "$(dirname "$0")/../engine"
mkdir -p bin/diffusion
# le reseau voyage a cote des executables (le moteur le cherche la)
cp bin/onechess.nnue bin/diffusion/onechess.nnue
attendu=$(cat ../ci/signature.txt)
# avx512 : le profil PGO et la verification tournent sous Intel SDE (le PC n'a pas l'AVX-512)
SDE="${SDE:-$(ls -d "$(pwd)"/../sortie/outils_sde/sde-external-*/sde.exe 2>/dev/null | head -1)}"
for cible in "avx512|-march=x86-64-v4 -mavx512vnni -mtune=generic -DUSE_PEXT" \
             "bmi2|-march=x86-64-v3 -mtune=generic -DUSE_PEXT" \
             "avx2|-march=x86-64-v3 -mtune=generic" \
             "sse41|-march=x86-64-v2 -mtune=generic"; do
    nom=${cible%%|*}; arch=${cible#*|}
    run=""
    if [ "$nom" = avx512 ]; then
        [ -n "$SDE" ] || { echo "onechess-avx512.exe : SDE introuvable, saute"; continue; }
        run="$SDE -icl --"
    fi
    PGO_RUN="$run" ARCH="$arch" OUT="bin/diffusion/onechess-$nom.exe" ./build.sh ${MODE_DIFF:-clang} > "bin/diffusion/$nom.log" 2>&1
    n=$(printf 'bench 12\nquit\n' | $run "./bin/diffusion/onechess-$nom.exe" | grep "Noeuds totaux" | awk '{print $4}')
    nps=$(printf 'bench 12\nquit\n' | $run "./bin/diffusion/onechess-$nom.exe" | grep "^NPS" | awk '{print $3}')
    [ "$n" = "$attendu" ] && etat=OK || etat="SIGNATURE DIFFERENTE ($n)"
    echo "onechess-$nom.exe : $etat, $nps NPS"
done
