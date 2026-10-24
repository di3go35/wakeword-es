#!/bin/bash
# Línea base completa del candidato C, en cadena y con verificación entre pasos.
#
#   1. espera a que termine la generación de positivos con relleno
#   2. calcula características de todos los positivos (reanudable: completa las que falten)
#   3. arma el conjunto (positivos por voz, negativos por hablante, validación disjunta)
#   4. entrena la cabeza y exporta ONNX
#
# Cualquier paso que falle detiene la cadena: no se entrena sobre datos incompletos.

set -u
set -o pipefail
REPO="$HOME/wakeword-es"
P="$HOME/.hermes/tools/venv-wakeword/.venv311/bin/python"
POS="$HOME/datos-cv/positivos/kokoro_pad"
FEATPOS="$HOME/datos-cv/features_positivos_pad"
DATOS="$HOME/datos-cv/entrenamiento"
UNIDAD=wakeword-positivos2

info() { echo "[$(date '+%H:%M:%S')] $*"; }

info "esperando a que termine $UNIDAD"
while systemctl --user is-active --quiet "$UNIDAD"; do sleep 20; done

generados=$(ls "$POS"/*.wav 2>/dev/null | wc -l)
info "positivos generados: $generados"
if [ "$generados" -lt 1800 ]; then
  info "ABORTO: se esperaban 1800 positivos y hay $generados"
  exit 2
fi

ls "$POS"/*.wav | xargs -n1 basename > "$POS/lista.txt"
info "características de los positivos"
"$P" "$REPO/pipeline/extraer_caracteristicas.py" --lista "$POS/lista.txt" --audio-dir "$POS" \
     --salida "$FEATPOS" --lote-id kokoro > "$HOME/datos-cv/logs/features_positivos.log" 2>&1 || {
  info "ABORTO: falló la extracción de características"; exit 3; }
hechos=$(find "$FEATPOS" -maxdepth 1 -name '*.npy' | wc -l)
info "características listas: $hechos"
if [ "$hechos" -lt 1800 ]; then
  info "ABORTO: faltan características ($hechos de 1800)"; exit 4
fi

info "armando el conjunto de entrenamiento"
"$P" "$REPO/pipeline/preparar_entrenamiento.py" \
     --split "$HOME/datos-cv/split.json" --tsv "$HOME/datos-cv/validado_con_duraciones.tsv" \
     --features "$HOME/datos-cv/features" --features-positivos "$FEATPOS" \
     --salida "$DATOS" --neg-train 150000 --neg-val 25000 2>&1 | tail -20 || {
  info "ABORTO: falló la preparación del conjunto"; exit 5; }

info "entrenando la cabeza"
"$P" "$REPO/pipeline/entrenar_cabeza.py" --datos "$DATOS" --epochs 30 2>&1 | tail -40 || {
  info "ABORTO: falló el entrenamiento"; exit 6; }

info "LISTO"
