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

| 18:36 | **Few-shot: 87,5% de recall** dejando una grabación fuera (9 en entrenamiento + TTS). Zero-shot: 0%. |
| 18:48 | Lanzado v4 (TTS + aumentación de canal) bajo systemd. |
| 19:03 | v4: la probabilidad con voz humana sube 5× (0,0123 → 0,0634) pero el recall sigue en 0. |
| 19:12 | Modelo few-shot final: recall 10/10 en muestra, pero **4,469 FA/h** (el few-shot dispara en habla ajena). |
| 19:18 | Minados 2.844 negativos difíciles → **0,094 FA/h** con recall intacto. |
| 19:20 | Validación cruzada con negativos difíciles en curso. |

| 19:30 | Few-shot con ruido (MUSAN): recall en ruido no visto 32,5% → **100%**. FA/h sube a 0,867. |
| 19:50 | Segundo minado de negativos → FA/h **0,396** @0,5 y 0,151 @0,9. |
| 20:40 | Validación cruzada del recetario completo, con ruido no visto: **100% (10 de 10)**. |
| 21:10 | Curva de FA/h completa y punto de operación fijado en 0,9. |

| 21:35 | Silencio: 0 activaciones en 30 min (ceros y ruido térmico -70 dBFS). |
| 21:45 | Campo lejano simulado (3 m, RIR 0,5 s): 100% de recall. |
| 21:50 | Consolidado `docs/resultados.md` con todas las cifras y los enfoques descartados. |

| 22:30 | Especificidad: 4 de 30 frases con "celeste" disparaban; añadidas como negativos → **0 de 30**, y el FA/h baja a **0,113**. |
| 23:00 | Validación final dejando un audio fuera con el recetario completo: **100% (10/10)**. |

## 2026-10-27 (continuación autónoma)

| Hora UTC | Qué pasó |
|---|---|
| — | Fase 6.1 implementada: `pipeline/antidisparo.py` (compuerta RMS, frames consecutivos, enfriamiento). |
| — | Compuerta a −45 dBFS: limpio 100%, lejano 3 m **100%**, ruido 66% (igual que sin compuerta). A −35 el lejano cae a 40%: descartado por medición. |
| — | **Corrección importante**: el 100% de recall en ruido era la mediana de puntuación, no la regla operativa. Con 3 frames sobre 0,9: **66%**. Sin compuerta también 66%: no es la compuerta. No depende del SNR (fallos hasta +19 dB). |
| — | `docs/resultados.md`, `README.md` y `MODEL_CARD.md` corregidos con las dos cifras y el desglose por SNR. Decisiones 43 y 44. |
| 21:38 | Job `wakeword-compuerta` terminado: FA/h **0,113 → 0,094** con compuerta a −45 dBFS (6 → 5 eventos en 53,03 h). A −35 baja a 0,075 pero rompe el campo lejano: descartado. Sin costo de recall. |
| — | Fase 6.2: `pipeline/calibrar.py`. Sobre 8,19 h de habla real como «entorno» recomienda umbral **0,30** con 0,244 FA/h para un objetivo de 0,5. Elige el umbral **más bajo** que cumple el objetivo, para maximizar recall. Ejemplo en `benchmark/calibracion_ejemplo.json`. |

| — | Escrito `ESTADO.md`: documento de traspaso con fase actual, números, comandos, convención systemd y las 10 trampas que costaron tiempo. |

## Pendiente inmediato

1. **Recall en ruido (66%)**: es la única condición que no cumple el criterio. Necesita más
   variedad de canal en los positivos reales (más grabaciones, más condiciones), no más ruido sintético.
2. `calibrar` (fase 6.2): umbral por entorno. El ruido no mejora bajando el umbral (66% → 69% al pasar
   de 0,9 a 0,5), así que un umbral adaptativo no arregla esta condición: hace falta dato.
3. Test público 100% sintético con disjunción por motor (fase 5.1), para que un tercero reproduzca.
4. Demo Android con paridad numérica Python ↔ Android (fase 9).
