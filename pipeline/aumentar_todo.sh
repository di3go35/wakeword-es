#!/bin/bash
# Aumentación de canal de los dos conjuntos de positivos + características.
# Se ejecuta como unidad de systemd para sobrevivir al final del turno.

set -u
set -o pipefail
REPO="$HOME/wakeword-es"
P="$HOME/.hermes/tools/venv-wakeword/.venv311/bin/python"
BASE="$HOME/datos-cv/positivos"
MUSAN="$HOME/datos/musan/musan"

info() { echo "[$(date '+%H:%M:%S')] $*"; }

aumentar() {
  local origen="$1" destino="$2" nombre="$3"
  if [ ! -d "$origen" ]; then info "AVISO: no existe $origen"; return 0; fi
  info "aumentando $nombre"
  "$P" "$REPO/pipeline/aumentar_positivos.py" --audio "$origen" --salida "$destino" \
       --variantes 3 --musan "$MUSAN" 2>&1 | tail -3
  ls "$destino"/*.wav 2>/dev/null | xargs -n1 basename > "$destino/lista.txt" || true
  "$P" "$REPO/pipeline/extraer_caracteristicas.py" --lista "$destino/lista.txt" \
       --audio-dir "$destino" --salida "$HOME/datos-cv/features_$nombre" --lote-id "$nombre" \
       2>&1 | tail -3
  info "$nombre: $(find "$HOME/datos-cv/features_$nombre" -name '*.npy' | wc -l) características"
}

aumentar "$BASE/kokoro_pad" "$BASE/aug_kokoro" "aug_kokoro"
aumentar "$BASE/cb_v5"     "$BASE/aug_cb"     "aug_cb"
info "LISTO"
