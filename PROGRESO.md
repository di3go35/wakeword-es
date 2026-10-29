# Bitácora de avance

Registro cronológico del trabajo autónomo. Horas en UTC (tu hora = UTC-5).

## 2026-10-26

| Hora UTC | Qué pasó |
|---|---|
| 17:10 | Llegan 10 grabaciones reales (micrófono de laptop). Se convierten a 16 kHz mono. |
| 17:14 | Nivel medido: pico -16 a -28 dBFS, RMS -38 a -50 (muy bajo). |
| 17:18 | **Recall con voz humana real: 0,000** (máx 0,0019). Normalizado a -6 dBFS: 0,000 (máx 0,0123). |
| 17:22 | ASR `small` sobre audio sin normalizar alucina ("¡Hasta la próxima!"). |
| 17:28 | ASR `medium` sobre normalizado transcribe "¡Hey, Celeste!" en 5/10 y variantes en el resto: la frase está. |
| 17:34 | Estirar el tiempo ×1,4 y ×1,8 no mejora nada (máx 0,0006): no es velocidad. |
| 17:40 | Minado de negativos difíciles: solo 24 ventanas de ~1,1 M superan 0,10 → el modelo no confunde habla real, la rechaza. |
| 17:45 | Escrito `docs/hallazgo-voz-humana.md` y decisión 37. |
| 17:50 | Lanzada aumentación de canal (nivel, EQ de micrófono, reverberación) sobre 5.400 positivos Kokoro y 2.892 de Chatterbox. |
| 18:05 | Prueba few-shot (dejando una grabación fuera) en curso: 10 pliegues. |
| 18:33 | Características de augmentados: 4.203 de 5.400. MUSAN descargando (4,7 de ~11 GB). |

## Pendiente inmediato

1. Terminar características de augmentados → entrenar v4 → medir voz humana otra vez.
2. Few-shot: ¿bastan 10 grabaciones del usuario?
3. Con MUSAN completo: segunda pasada de aumentación con ruido real.
4. Si ninguna palanca levanta el recall humano: cambiar de arquitectura (candidato A/B).
