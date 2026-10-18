#!/bin/bash
# Procesa un lote: extrae sus clips del archivo del corpus y calcula características.
#
# Diseño: el archivo es un gzip de 52 GB, así que no se puede buscar dentro. Cada lote se
# extrae en UNA pasada sobre el archivo, se procesa en paralelo (N procesos) y al final se
# borran los clips para no acumular decenas de GB. Todo es reanudable: si se corta, se
# relanza y salta lo ya hecho.
#
# Uso:
#   pipeline/procesar_lote.sh <indice> [n_procesos]

set -u

LOTE="${1:?uso: procesar_lote.sh <indice_lote> [n_procesos]}"
NP="${2:-3}"
BASE="$HOME/datos-cv"
ARCH="$BASE/common-voice-scripted-speech-27-0-spanis.tar.gz"
DIR_CORPUS="$BASE/corpus/cv-corpus-27.0-2026-09-11/es"
RAIZ="$(basename "$(dirname "$DIR_CORPUS")")"       # cv-corpus-27.0-2026-09-11
LISTA="$BASE/lotes/lote_${LOTE}.txt"
LISTA_TAR="$BASE/lotes/lote_${LOTE}.tar.txt"
AUDIO="$DIR_CORPUS/clips"
FEATURES="$BASE/features"
LOGS="$BASE/logs"
VENV="$HOME/.hermes/tools/venv-wakeword/.venv311/bin/python"
PY="$(command -v python3)"

info() { echo "[$(date '+%H:%M:%S')] lote $LOTE: $*"; }
mkdir -p "$AUDIO" "$FEATURES" "$LOGS"

[ -f "$LISTA" ] || { echo "ERROR: falta $LISTA" >&2; exit 2; }
TOTAL=$(wc -l < "$LISTA")
info "clips en el lote: $TOTAL | procesos: $NP"

# --- 1. Lista con rutas completas dentro del archivo (tar -T compara nombres exactos)
if [ ! -f "$LISTA_TAR" ] || [ "$(wc -l < "$LISTA_TAR")" -ne "$TOTAL" ]; then
  sed "s|^|${RAIZ}/es/clips/|" "$LISTA" > "$LISTA_TAR"
fi

# --- 2. Extracción (una sola pasada sobre el archivo; salta si ya están)
if [ "$(ls -1 "$AUDIO" 2>/dev/null | wc -l)" -lt "$TOTAL" ]; then
  info "extrayendo clips (una pasada sobre los 52 GB, 5-10 min)..."
  tar -xzf "$ARCH" -C "$AUDIO" -T "$LISTA_TAR" --strip-components=3
  info "extracción terminada: $(ls -1 "$AUDIO" | wc -l) archivos en clips/"
else
  info "los clips ya estaban extraídos"
fi

# --- 3. Características en paralelo
mkdir -p "$BASE/trabajo"
split -n l/"$NP" -d --additional-suffix=.txt "$LISTA" "$BASE/trabajo/parte_"
pids=()
for parte in "$BASE/trabajo/parte_"*.txt; do
  sufijo="$(basename "$parte" .txt | sed 's/parte_//')"
  log="$LOGS/lote_${LOTE}_${sufijo}.log"
  "$VENV" "$(dirname "$0")/extraer_caracteristicas.py" \
      --lista "$parte" --audio-dir "$AUDIO" --salida "$FEATURES" \
      --lote-id "${LOTE}${sufijo}" > "$log" 2>&1 &
  pids+=($!)
  info "proceso $sufijo lanzado (log: $log)"
done
for pid in "${pids[@]}"; do wait "$pid"; done
info "los $NP procesos terminaron"

# --- 4. Resumen
"$PY" - "$FEATURES" "$LOTE" <<'PY'
import json, sys
from pathlib import Path
features, lote = Path(sys.argv[1]), sys.argv[2]
filas, errores, horas = 0, 0, 0.0
for man in sorted(features.glob(f"manifiesto_{lote}*.jsonl")):
    for linea in man.read_text(encoding="utf-8").splitlines():
        d = json.loads(linea)
        filas += 1
        horas += d["segundos"] / 3600.0
for log in sorted(Path(features).parent.joinpath("logs").glob(f"lote_{lote}*.log")):
    errores += log.read_text(errors="ignore").count("ERROR ")
print(json.dumps({"lote": lote, "clips_con_caracteristicas": filas,
                  "horas": round(horas, 2), "errores": errores}, ensure_ascii=False))
PY
info "características en $FEATURES"

# --- 5. Borrar los clips del lote (el archivo se queda; los clips no hacen falta ya)
antes=$(df --output=avail -BM / | tail -1 | tr -d ' M')
# sin -I: rm se invoca por lotes, no una vez por archivo (con 170.000 clips la diferencia
# es entre segundos y horas)
sed "s|^|$AUDIO/|" "$LISTA" | xargs -d '\n' rm -f
despues=$(df --output=avail -BM / | tail -1 | tr -d ' M')
info "clips borrados; disco libre: ${antes} MB -> ${despues} MB"
info "LISTO"
