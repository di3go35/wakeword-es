# Banco de evaluación — wakeword-es

Aquí vive el criterio de éxito del proyecto convertido en código ejecutable. No es
decoración: `--criterio` devuelve código de salida 1 cuando el modelo no alcanza el
listón, y `make autotest` demuestra los dos caminos (cumple / no cumple) con fixtures
sintéticos.

## Criterio congelado

- **≥95% de recall a ≤0,5 falsos positivos por hora**, y ese recall se exige **en cada
  condición** por separado (limpio, TV/música a ~10 dB SNR, campo lejano a 3 m).
- **0 activaciones en el test de silencio digital.**
- El **manifiesto del test** debe coincidir con el hash congelado; si alguien cambia el
  test después de publicarlo, la evaluación se aborta con código 2.
- Se publica la **curva DET completa** (recall a 0,2 / 0,5 / 1,0 FA/h), no un punto único.

Los umbrales de estos valores están en `evaluar.py` (`RECALL_MINIMO`, `FA_H_CRITERIO`) y
no se cambian sin reescribir `PLAN.md`.

## Formato de entrada

JSONL, un registro por línea. Ver el docstring de `evaluar.py` para el detalle.
Resumen: `{"id", "tipo": positivo|negativo|silencio, "frames": [[t, score], ...]}`,
con `dur_s` **obligatorio** en negativos y silencio (sin duración explícita no se pueden
contar falsos positivos por hora, y el banco se niega a adivinar).

## Uso

```bash
python3 benchmark/evaluar.py --puntajes puntajes.jsonl --criterio
python3 benchmark/evaluar.py --puntajes puntajes.jsonl \
        --manifiesto benchmark/manifiesto.json --json informe.json
```

## Autoprueba (fixtures sintéticos, nunca resultados)

```bash
make test       # pruebas unitarias de las métricas y del criterio
make autotest   # perfil 'bueno' debe cumplir; perfil 'malo' debe fallar
```

El perfil `malo` existe para probar que el criterio puede fallar. Un proyecto de wake
words que nunca puede fallar su propio criterio no está midiendo nada.
