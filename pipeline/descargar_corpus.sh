#!/bin/bash
# Descarga reanudable y auto-vigilada del corpus de Common Voice (Mozilla Data Collective).
#
# Por qué existe: un `curl --retry` no sobrevive a todos los cortes — si el proceso muere,
# la descarga se queda a medias y nadie la retoma. Este script es el supervisor: reanuda
# desde el byte donde iba, refresca la URL prefirmada cuando caduca o cuando no hay avance,
# y verifica el sha256 contra el que publica la API antes de dar el trabajo por bueno.
#
# Uso (no necesita argumentos si usas el corpus del proyecto):
#   MDC_API_KEY=... pipeline/descargar_corpus.sh [destino] [dataset_id]
#
# La clave se lee de la variable de entorno o de ~/.hermes/.env (nunca se imprime).

set -u

DESTINO="${1:-$HOME/datos-cv}"
DATASET="${2:-cmu5kitx700pemi073feoiqqu}"
API="https://mozilladatacollective.com/api"
ARCHIVO="$DESTINO/common-voice-scripted-speech-27-0-spanis.tar.gz"

mkdir -p "$DESTINO"
if [ -z "${MDC_API_KEY:-}" ] && [ -f "$HOME/.hermes/.env" ]; then
  MDC_API_KEY=$(grep -m1 '^MDC_API_KEY=' "$HOME/.hermes/.env" | cut -d= -f2-)
fi
if [ -z "${MDC_API_KEY:-}" ]; then
  echo "ERROR: falta MDC_API_KEY (variable de entorno o ~/.hermes/.env)" >&2
  exit 2
fi

info() { echo "[$(date '+%H:%M:%S')] $*"; }

# Pide a la API una URL prefirmada nueva y devuelve también el tamaño esperado.
obtener_url() {
  curl -sS -X POST -H "Authorization: Bearer $MDC_API_KEY" \
       "$API/datasets/$DATASET/download" -o "$DESTINO/.respuesta_api.json" -w '%{http_code}' \
  | grep -q '^200$' || return 1
  python3 - "$DESTINO/.respuesta_api.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(d["downloadUrl"])
print(d["sizeBytes"])
print(d.get("checksum", ""))
PY
}

read -r URL TOTAL CHECKSUM < <(obtener_url | tr '\n' ' ') || true
TOTAL="${TOTAL:-0}"
if [ "$TOTAL" = "0" ]; then
  echo "ERROR: la API no devolvió tamaño. ¿Aceptaste los términos en la web?" >&2
  exit 3
fi
info "dataset $DATASET | esperado: $TOTAL bytes ($((TOTAL/1048576)) MB)"
info "checksum publicado: ${CHECKSUM:0:16}..."

intento=0
sin_avance=0
while :; do
  tam=$(stat -c%s "$ARCHIVO" 2>/dev/null || echo 0)
  if [ "$tam" -ge "$TOTAL" ]; then
    info "descarga completa: $tam bytes"
    break
  fi
  intento=$((intento + 1))
  antes=$tam
  info "intento $intento desde $((tam/1048576)) MB de $((TOTAL/1048576)) MB"

  if [ "$sin_avance" -ge 1 ]; then
    info "refrescando URL prefirmada"
    read -r URL _T _C < <(obtener_url | tr '\n' ' ') || true
  fi

  curl -sS -L -C - --retry 3 --retry-delay 5 --retry-all-errors \
       --speed-limit 1000 --speed-time 30 --connect-timeout 20 \
       -o "$ARCHIVO" "$URL" 2>>"$DESTINO/descarga.log"

  despues=$(stat -c%s "$ARCHIVO" 2>/dev/null || echo 0)
  info "en $((despues/1048576)) MB de $((TOTAL/1048576)) MB"
  if [ "$despues" -le "$antes" ]; then
    sin_avance=$((sin_avance + 1))
    info "sin avance (intento $sin_avance)"
    sleep 10
  else
    sin_avance=0
  fi
  if [ "$intento" -ge 200 ]; then
    echo "ERROR: demasiados intentos, me detengo" >&2
    exit 4
  fi
done

info "verificando sha256..."
real=$(sha256sum "$ARCHIVO" | cut -d' ' -f1)
if [ -n "$CHECKSUM" ] && [ "$real" = "$CHECKSUM" ]; then
  info "VERIFICADO: $real"
  echo "$real  $(basename "$ARCHIVO")" > "$DESTINO/VERIFICADO.sha256"
  exit 0
fi
info "ATENCION: sha256 no coincide. esperado=$CHECKSUM real=$real"
exit 5
