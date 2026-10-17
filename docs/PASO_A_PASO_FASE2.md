# Fase 2 — qué hacer tú, paso a paso

Lo que ya está hecho desde aquí: banco de evaluación ejecutable, criterio congelado,
manifiesto con hash, repo público y licencias. **Lo que falta es la línea base**, y eso
requiere una cuenta de Kaggle tuya, así que es tu turno.

El objetivo de esta fase no es un modelo bueno: es un **número malo, reproducido y
publicado**. Si el baseline sale bien, es señal de que el test está mal construido.

---

## Paso 1 — Cuenta y descarga en Mozilla Data Collective (paso a paso)

Datos verificados de la ficha del dataset (consultada el 2026-10-17):

| Campo | Valor |
|---|---|
| Nombre | Common Voice Scripted Speech 26.0 — Spanish |
| Dataset ID | `cmqim2spa00synr071fcp7av0` |
| Licencia | **CC0-1.0** |
| Formato | MP3 |
| Tamaño | **48,30 GB** |
| Contenido | 1.680.810 clips · 2.278,66 h grabadas · **595,45 h validadas** · 26.940 hablantes |
| Fecha de publicación | 2026-06-17 |
| **Forbidden usage** | "se prohíbe intentar determinar la identidad de los hablantes" y **"se prohíbe re-hospedar o re-compartir este dataset"** |

Pasos:

1. Entra en `mozilladatacollective.com` y crea una cuenta (es gratuita; es una plataforma
   nueva, incubada por la Mozilla Foundation).
2. Abre la ficha del dataset por ID: `mozilladatacollective.com/datasets/cmqim2spa00synr071fcp7av0`
3. Comprueba que la ficha diga `cv-corpus-26.0-2026-06-12`, CC0-1.0 y 48,30 GB. Si te
   ofrece variantes (por ejemplo solo el split validado), **elige la más pequeña que
   incluya `validated.tsv`**; si solo hay una opción, son los 48,30 GB completos.
4. Pulsa **Download** y acepta los términos si te los pide. Descarga al servidor, no a tu
   laptop: `/home/di3gor35/datos-cv/` (73 GB libres allí y el `.gitignore` ya bloquea todo
   audio).
5. Verifica la descarga: que `validated.tsv` exista, que el número de clips tenga sentido
   y que no haya archivos truncados (`find . -name '*.mp3' -size -1k | head`).
6. **No lo re-hospedes en ningún sitio**: ni Kaggle, ni Hugging Face, ni un torrent. La
   prohibición es explícita en la propia ficha. Lo único que puede salir de aquí son
   **características derivadas** (embeddings), y en dataset privado; el audio no.

> Nota de disco: el corpus completo pesa casi 48 GB. Si el espacio aprieta, basta con las
> 595 h validadas (~12-13 GB de MP3) para todo el proyecto; el resto son clips sin validar
> que no aportan negativos fiables.

### Hallazgo que cambia la calidad del proyecto

La ficha trae el desglose por acento, con etiquetas por clip. Lo relevante para ti:

| Acento | Clips | Hablantes |
|---|---|---|
| mexicano | 879.480 (52,3%) | 1.380 |
| surpeninsular | 178.950 (10,6%) | 261 |
| nortepeninsular | 66.653 (4,0%) | 638 |
| **andino (Colombia, Perú, Ecuador, oeste de Bolivia y Venezuela andina)** | **38.728 (2,3%)** | **947** |
| rioplatense | 23.828 (1,4%) | 579 |
| caribe | 21.835 (1,3%) | 582 |

Es decir: existe un subconjunto etiquetado del **acento andino-pacífico, el tuyo**, con 947
hablantes distintos. Eso permite entrenar y evaluar **estratificado por acento**, que es
algo que ningún proyecto abierto hace, y publicar la curva DET desglosada por acento. Es un
diferenciador medible y gratis.


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
