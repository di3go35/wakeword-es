# Descartes de clips

Un clip que no se puede procesar **no se borra en silencio**: se registra aquí con su motivo.
El verificador (`pipeline/verificar_features.py`) trata un clip ausente **con** motivo como
una exclusión consciente y uno ausente **sin** motivo como un fallo del lote.

## Lote 001 — 179.924 clips, 250,00 h

| Clips | Motivo | Causa real |
|---|---|---|
| 5 | `audio demasiado corto: el modelo mel recibió rango 2 y ONNX abortó` | Clips de pocas decenas de milisegundos. El extractor no tenía guarda de duración mínima; se añadió `MIN_SEGUNDOS = 0,30` |
| 1 | `forma inesperada (96,)`: un solo vector sin dimensión de ventanas | Caso degenerado del mismo problema; el `.npy` se eliminó y su fila del manifiesto también |

**Cobertura: 179.918 clips útiles de 179.924 (99,997%).** Los 5 clips cortos y el degenerado
requerirían otra pasada de 70 minutos sobre los 52 GB del archivo para recuperarlos: no
compensa, y su aporte a un conjunto de 178.000 negativos es indistinguible de cero.

## Regla para los lotes siguientes

El extractor descarta automáticamente, y escribe el motivo en
`descartes_<lote>.jsonl`:

- audio de menos de **0,30 s** (no produce una ventana de embedding válida);
- features con forma distinta de `(n, 96)` o con `n = 0`.

Cualquier lote que presente descartes por **otra** causa se investiga antes de darlo por
bueno: un motivo nuevo puede ser un bug, no un dato malo.
