#!/bin/bash
# Envía un mensaje a Diego por el bridge de WhatsApp del servidor.
# Uso: avisar.sh "texto"   (o sin argumentos: resumen automático del estado)
set -u

DESTINO="51912997055@s.whatsapp.net"
BRIDGE="http://127.0.0.1:3000/send"

MARCA="$HOME/.hermes/wakeword_ultimo_aviso"
SI_VIEJO=0
if [ "${1:-}" = "--si-viejo" ]; then
  SI_VIEJO="${2:-50}"
  shift 2 || true
fi
if [ "$SI_VIEJO" -gt 0 ] && [ -f "$MARCA" ]; then
  ULTIMO=$(stat -c %Y "$MARCA" 2>/dev/null || echo 0)
  AHORA=$(date +%s)
  if [ $((AHORA - ULTIMO)) -lt $((SI_VIEJO * 60)) ]; then
    echo "sin aviso: el anterior fue hace $(( (AHORA - ULTIMO) / 60 )) min"
    exit 0
  fi
fi

if [ $# -gt 0 ]; then
  MENSAJE="$*"
else
  BASE="$HOME/datos-cv"
  FEAT=$(find "$BASE/features" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)
  AUGK=$(find "$BASE/features_aug_kokoro" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)
  AUGC=$(find "$BASE/features_aug_cb" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)
  CARGA=$(cut -d' ' -f1-3 /proc/loadavg)
  MENSAJE="wakeword-es · $(date '+%H:%M' UTC)
Sigo trabajando. Estado:
· características reales: $FEAT
· augmentadas: $AUGK kokoro / $AUGC chatterbox
· carga del servidor: $CARGA
Último hito: $(grep -m1 '^| [0-9]' "$HOME/wakeword-es/PROGRESO.md" 2>/dev/null | tail -1 | cut -c3-80)"
fi

python3 - "$DESTINO" "$MENSAJE" "$BRIDGE" <<'PY'
import json, sys, urllib.request
destino, mensaje, bridge = sys.argv[1], sys.argv[2], sys.argv[3]
datos = json.dumps({"chatId": destino, "message": mensaje}).encode()
req = urllib.request.Request(bridge, data=datos, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=20) as r:
        print(r.read().decode()[:200])
        open(__import__("os").path.expanduser("~/.hermes/wakeword_ultimo_aviso"), "w").write("ok")
except Exception as e:
    print("ERROR al enviar:", e)
PY
