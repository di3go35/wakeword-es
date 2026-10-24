#!/bin/bash
# Espera a que termine la extracción del test, VERIFICA, y solo entonces borra el archivo
# del corpus (52 GB). Si las cuentas no cuadran, no borra nada: el archivo es la única copia.
#
# Regla: nunca se libera espacio a costa de perder la única fuente de audio.

set -u
BASE="$HOME/datos-cv"
ARCH="$BASE/common-voice-scripted-speech-27-0-spanis.tar.gz"
DESTINO="$BASE/test/clips"
ESPERADOS=$(wc -l < "$BASE/test/clips.tar.txt")
UNIDAD=wakeword-test-audio
MAX_ESPERA=14400   # 4 h

info() { echo "[$(date '+%H:%M:%S')] $*"; }

inicio=$(date +%s)
while systemctl --user is-active --quiet "$UNIDAD"; do
  ahora=$(date +%s)
  if [ $((ahora - inicio)) -gt "$MAX_ESPERA" ]; then
    info "ABORTO: la extracción no terminó en $MAX_ESPERA s. No borro nada."
    exit 2
  fi
  sleep 20
done

resultado=$(systemctl --user show "$UNIDAD" -p Result --value 2>/dev/null)
total=$(find "$DESTINO" -maxdepth 1 -name '*.mp3' | wc -l)
truncados=$(find "$DESTINO" -maxdepth 1 -name '*.mp3' -size -1k | wc -l)
info "extracción: result=$resultado | clips=$total de $ESPERADOS | truncados=$truncados"

if [ "$resultado" != "success" ] || [ "$total" -ne "$ESPERADOS" ] || [ "$truncados" -ne 0 ]; then
  info "ABORTO: la verificación no cuadra. NO se borra el archivo."
  exit 3
fi

antes=$(df --output=avail -BG / | tail -1 | tr -d ' G')
info "verificación correcta: borrando el archivo del corpus ($(du -h "$ARCH" | cut -f1))"
rm -f "$ARCH"
despues=$(df --output=avail -BG / | tail -1 | tr -d ' G')
info "archivo borrado. Disco libre: ${antes} GB -> ${despues} GB"
info "el test queda en $DESTINO ($(du -sh "$DESTINO" | cut -f1))"
info "LISTO"
