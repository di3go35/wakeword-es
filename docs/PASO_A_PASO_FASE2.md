# Fase 2 — qué hacer tú, paso a paso

Lo que ya está hecho desde aquí: banco de evaluación ejecutable, criterio congelado,
manifiesto con hash, repo público, licencias y **el entorno del pipeline funcionando y
medido en tu servidor**. Lo que falta para la línea base es el **audio**: el corpus español
está detrás de una cuenta en Mozilla Data Collective (paso 1), y la frase provisional la
elegimos en el paso 3.

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
3. **Acepta los términos en la web.** Este es el único paso que no se puede hacer por API:
   la propia documentación dice que "el usuario debe haber aceptado previamente los
   términos de uso del dataset mediante la interfaz web" y que la aceptación por API no
   está soportada. Sin ese clic, la API responde `403 Access to this dataset has not been
   granted` — comprobado el 2026-10-17 con la clave ya cargada en el servidor.
4. Confirma que la ficha diga `cv-corpus-26.0-2026-06-12`, CC0-1.0 y 48,30 GB.
5. Avisa y el resto lo hago yo por API (ver más abajo): la descarga no la tienes que
   lanzar tú.

### La API de MDC, ya verificada desde el servidor

| Dato | Valor |
|---|---|
| Base URL | `https://mozilladatacollective.com/api` |
| Autenticación | cabecera `Authorization: Bearer <clave>` (clave en Profile → API) |
| `GET /health` | 200 sin autenticación — comprobado |
| `GET /datasets/:id` | metadatos del dataset — comprobado (403 hasta aceptar términos) |
| `POST /datasets/:id/download` | devuelve una **URL prefirmada** de storage, con `sizeBytes`, `filename` y `checksum`; válida 12 h |
| Endpoint antiguo | `GET /datasets/:id/download/:token` **retirado** (devuelve 410): si ves un tutorial que lo usa, está obsoleto |

La clave está guardada en `~/.hermes/.env` como `MDC_API_KEY`, con permisos 600 y fuera del
repositorio. **Rótala cuando termine la descarga**: la enviaste por chat, así que está
expuesta en el historial y conviene reemplazarla por una nueva en Profile → API.

### Plan de descarga (decidido por el tamaño real, no estimado)

1. Con los términos aceptados, `POST /download` da la URL prefirmada y el `sizeBytes` exacto.
2. Disco disponible: 72 GB. El archivo es un `.tar.gz` y el audio MP3 apenas comprime, así
   que el plan es: **descargar el archivo una sola vez (≈45-48 GB), extraer solo lo
   necesario y borrar el archivo después**. Nunca extraer todo más el archivo a la vez, que
   serían ~96 GB y no caben.
3. Extracción selectiva: primero los `.tsv` (pequeños), parsear `validated.tsv` para
   quedarnos con el subconjunto validado (~595 h), y extraer esos clips en lugar de los
   1,68 millones.
4. Registro de procedencia actualizado con los hashes reales de lo descargado.
5. El audio **no sale del servidor**: a Kaggle solo irían características derivadas, en
   dataset privado.


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


## Paso 2 — Entorno local ya preparado (no tienes que hacer nada aquí)

El pipeline ya corre en tu servidor, sin root:

```bash
~/.hermes/tools/venv-wakeword/.venv311/bin/python   # Python 3.11 + openwakeword 0.6.0 + onnxruntime
```

Datos medidos el 2026-10-17 en este host:

| Medición | Valor |
|---|---|
| Extracción de características (mel + embedding) | **28,7× tiempo real** (120 s de audio en 4,18 s) |
| Vector de embedding | 96 dimensiones, ~12,4 ventanas/s |
| 595 h de audio | ≈ **20,7 h de CPU en serie**, ~5 h con 4 procesos |
| 100 h de audio | ≈ 3,5 h de CPU |
| Descarga de los modelos base | 1,7 s |

Y una trampa de dependencias que ya está resuelta: **`openwakeword==0.6.0` exige
`tflite-runtime`, cuyos wheels solo existen hasta Python 3.11**. Por eso el entorno es 3.11
—que además es el que usa Kaggle— y no 3.12 ni 3.14. Si algún día ves ese error, la causa es
esta, no tu código.

Consecuencia práctica: **la línea base y el candidato C se pueden entrenar y evaluar aquí**,
sin gastar cuota de Kaggle. Kaggle queda para el candidato A (CNN con GPU) y para alojar el
dataset privado de características derivadas.

## Paso 3 — Puntajes del baseline

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

## Paso 4 — Consentimientos (en paralelo, no urge hasta la fase 4)

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
