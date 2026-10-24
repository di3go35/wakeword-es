#!/bin/bash
# Extrae del archivo del corpus SOLO el audio del test congelado.
#
# Es la última pasada sobre los 52 GB: después el archivo se puede borrar, porque las
# características de las 620 h ya están calculadas y el test queda en disco como audio.
# Se ejecuta como unidad transitoria de systemd (ver docs), con cortesía de CPU y disco.

set -u
BASE="$HOME/datos-cv"
ARCH="$BASE/common-voice-scripted-speech-27-0-spanis.tar.gz"
LISTA="$BASE/test/clips.tar.txt"
DESTINO="$BASE/test/clips"

info() { echo "[$(date '+%H:%M:%S')] $*"; }

[ -f "$ARCH" ] || { echo "ERROR: no existe $ARCH" >&2; exit 2; }
[ -f "$LISTA" ] || { echo "ERROR: no existe $LISTA" >&2; exit 3; }
mkdir -p "$DESTINO"

info "extrayendo $(wc -l < "$LISTA") clips del test"
ionice -c2 -n7 nice -n 10 tar -xzf "$ARCH" -C "$DESTINO" -T "$LISTA" \
    --strip-components=3 --skip-old-files

total=$(find "$DESTINO" -maxdepth 1 -name '*.mp3' | wc -l)
vacios=$(find "$DESTINO" -maxdepth 1 -name '*.mp3' -size -1k | wc -l)
info "extracción terminada: $total clips, $vacios sospechosos de estar truncados (<1 KB)"
du -sh "$DESTINO"
info "LISTO"
