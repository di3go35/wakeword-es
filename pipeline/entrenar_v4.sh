#!/bin/bash
# Modelo v4: positivos originales + aumentados de canal, negativos reales.
#
# Une cuatro fuentes de positivos respetando los splits ya hechos:
#   - Kokoro original (ef_dora/em_alex train, em_santa validación)
#   - Chatterbox original limpio (split por clip en cbv5_val.txt)
#   - Kokoro aumentado (mismo split por voz)
#   - Chatterbox aumentado (hereda el split de cbv5_val.txt)
# y mide al final sobre: voz humana real, Chatterbox no visto, em_santa y las 53 h.

set -u
set -o pipefail
REPO="$HOME/wakeword-es"
P="$HOME/.hermes/tools/venv-wakeword/.venv311/bin/python"
BASE="$HOME/datos-cv"
E="$BASE/entrenamiento"

info() { echo "[$(date '+%H:%M:%S')] $*"; }

info "arrays de los aumentados"
"$P" "$REPO/pipeline/agregar_motor.py" --features "$BASE/features_aug_kokoro" \
     --manifiesto "$BASE/features_aug_kokoro/manifiesto_aug_kokoro.jsonl" \
     --audio "$BASE/positivos/aug_kokoro" --salida "$E" --prefijo augk --voz-val em_santa \
     || { info "ABORTO en aug kokoro"; exit 2; }

"$P" "$REPO/pipeline/agregar_motor.py" --features "$BASE/features_aug_cb" \
     --manifiesto "$BASE/features_aug_cb/manifiesto_aug_cb.jsonl" \
     --audio "$BASE/positivos/aug_cb" --salida "$E" --prefijo augc \
     --val-list "$E/cbv5_val.txt" \
     || { info "ABORTO en aug chatterbox"; exit 3; }

info "combinando"
"$P" - <<'PY'
import numpy as np
from pathlib import Path
E = Path('/home/di3gor35/datos-cv/entrenamiento')
partes = {'pos_train': ['pos_train_kokoro.npy', 'cbv5_pos_train.npy',
                        'augk_pos_train.npy', 'augc_pos_train.npy'],
          'pos_val':   ['pos_val_kokoro.npy', 'cbv5_pos_val.npy',
                        'augk_pos_val.npy', 'augc_pos_val.npy']}
for destino, fuentes in partes.items():
    arrays = []
    for f in fuentes:
        ruta = E / f
        if ruta.exists():
            a = np.load(ruta); arrays.append(a); print(f'  {f}: {a.shape[0]}')
    total = np.concatenate(arrays)
    np.save(E / f'{destino}.npy', total)
    print(f'{destino}: {total.shape[0]} ventanas')
PY
[ $? -eq 0 ] || { info "ABORTO combinando"; exit 4; }

info "entrenando v4"
"$P" "$REPO/pipeline/entrenar_cabeza.py" --datos "$E" --epochs 30 2>&1 | tail -8 \
  || { info "ABORTO entrenando"; exit 5; }

info "midiendo sobre voz humana real"
"$P" "$REPO/pipeline/medir_recall.py" --modelo "$E/cabeza.onnx" \
     --features "$BASE/features_diego" --manifiesto "$BASE/positivos/diego/manifiesto.jsonl" \
     --salida "$REPO/benchmark/recall_voz_humana.json" 2>&1 | tail -14

info "midiendo sobre voz humana normalizada"
"$P" "$REPO/pipeline/medir_recall.py" --modelo "$E/cabeza.onnx" \
     --features "$BASE/features_diego_norm" \
     --manifiesto "$BASE/positivos/diego_norm/manifiesto.jsonl" \
     --salida "$REPO/benchmark/recall_voz_humana_norm.json" 2>&1 | tail -14

info "midiendo sobre Chatterbox no visto"
"$P" "$REPO/pipeline/medir_recall.py" --modelo "$E/cabeza.onnx" \
     --features "$BASE/features_cb_v5" --manifiesto /tmp/cbv5_val.jsonl 2>&1 | tail -8

info "midiendo sobre em_santa (Kokoro, voz no vista)"
"$P" "$REPO/pipeline/medir_recall.py" --modelo "$E/cabeza.onnx" \
     --features "$BASE/features_positivos_pad" --manifiesto /tmp/es_manifest.jsonl 2>&1 | tail -8

info "midiendo FA/h en las 53 h"
for U in 0.50 0.70; do
  echo "--- umbral $U ---"
  "$P" "$REPO/pipeline/medir_test.py" --modelo "$E/cabeza.onnx" --features "$BASE/features" \
       --tsv "$BASE/validado_con_duraciones.tsv" --clips "$BASE/test/clips.txt" --umbral "$U" \
       2>&1 | grep -E "^TOTAL"
done
info "LISTO v4"
