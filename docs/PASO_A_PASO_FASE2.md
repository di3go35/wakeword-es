# Fase 2 — qué hacer tú, paso a paso

Lo que ya está hecho desde aquí: banco de evaluación ejecutable, criterio congelado,
manifiesto con hash, repo público y licencias. **Lo que falta es la línea base**, y eso
requiere una cuenta de Kaggle tuya, así que es tu turno.

El objetivo de esta fase no es un modelo bueno: es un **número malo, reproducido y
publicado**. Si el baseline sale bien, es señal de que el test está mal construido.

---

## Paso 1 — Cuenta en Mozilla Data Collective (10 min)

1. Alta en `mozilladatacollective.com` (gratis).
2. Busca el dataset *Common Voice Scripted Speech — Spanish* (CC0). La versión 26.0 son
   2.278 h, de las cuales ~595 h están validadas: **usa las validadas**.
3. Descarga solo para tu uso. **No lo reempaquetes ni lo subas a ningún lado** —
   los términos de Common Voice piden no replicar el dataset en otras plataformas.
   Lo que sí se publica es el **script de descarga** y el manifiesto con hashes.

> El corpus pesa varios GB y no hace falta entero para la fase 2: con 30–60 min de audio
> negativo y 20 clips de la frase provisional ya puedes correr el baseline y validar el
> banco completo.

## Paso 2 — Puntajes del baseline en Kaggle

Restricciones verificadas que te van a morder si las ignoras:

| Restricción | Detalle |
|---|---|
| Python | El pipeline estándar (piper-sample-generator, torchaudio) exige **3.10–3.12**. Kaggle trae 3.11: sirve. Tu VPS tiene 3.14: no sirve para esto |
| Disco | El pipeline completo pide **15–25 GB** |
| Tiempo | 1–2 h con GPU; **12–24 h solo CPU** por modelo |
| Cuota | 30 h/semana de GPU, sesiones de 9–12 h: si te pasas, se corta a mitad |
| Salida | Un `.onnx` de ~200 KB, más los `puntajes.jsonl` de la evaluación |

Lo que necesitas producir con la línea base (openWakeWord preentrenado, o el modelo
comunitario en español más cercano, o tu intento previo con livekit-wakeword):

- Un `puntajes.jsonl` con el formato documentado en `benchmark/README.md`.
- Los positivos del test con **~3 s de silencio líder** antes de la frase (los detectores
  de streaming no emiten hasta llenar su ventana; sin cebado mides falsos rechazos
  artificiales).
- Los negativos con `dur_s` explícito y sus frames puntuados.
- El bloque de silencio digital (30 min si puedes; se acepta 1.800 s para el primer pase).

Y después, aquí mismo:

```bash
make test        # 12/12, ya en verde
make benchmark   # evalúa tu puntajes.jsonl contra el criterio congelado
```

## Paso 3 — Consentimientos (en paralelo, no urge hasta la fase 4)

Plantilla lista en `docs/consentimiento-voz.md`. Necesitas 10–15 personas. Sin
consentimiento escrito no se clona ninguna voz para generar datos.

---

## Qué me pegas de vuelta

Cuando algo falle en Kaggle: el traceback completo y la celda donde murió. Iteramos ahí.
Cuando tengas el `puntajes.jsonl` del baseline: dámelo y lo evaluamos, se publica el
`benchmark/RESULTADOS.md` y se cierra la fase 2.

## Qué NO hacer todavía

- No entrenes nada propio antes de tener el baseline medido: no habría contra qué comparar.
- No generes datos sintéticos masivos antes de congelar el test (fase 5): el riesgo de
  contaminar el test con la misma voz es exactamente lo que el split por hablante y motor
  existe para evitar.
- No reempaquetes audio de Common Voice ni de ninguna fuente, ni siquiera en un dataset
  privado de Kaggle que puedas volver público por error.
