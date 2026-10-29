# wakeword-es

Pipeline abierto para entrenar **wake words en español latinoamericano** con falsos positivos
**medidos** y procedencia de datos auditable. Código Apache-2.0; los datos y modelos llevan su
propia licencia declarada.

Nace de un problema concreto: los modelos abiertos de wake word usan un backbone de embeddings cuyo
idioma dominante es el inglés y cuyos datos negativos tienen licencias restrictivas. En la práctica
eso produce dos síntomas que cualquiera reconoce: **se activan sin que nadie hable** y **no reconocen
a la persona que sí dijo la frase**.

## Resultados medidos (no prometidos)

Test de desarrollo: **37.106 clips / 53,03 h de habla real** de 3.247 hablantes que el modelo nunca
vio, en 10 condiciones de acento.

| Enfoque | Recall con voz humana real | Recall TTS no visto | Falsos positivos |
|---|---|---|---|
| Solo TTS (Kokoro + Chatterbox) | **0 / 10** | 99,5% | 0,019 FA/h |
| + aumentación de canal | **0 / 10** | — | 0,019 FA/h |
| Few-shot (9 audios del usuario + TTS) | **87,5%** (dejando uno fuera) | — | 4,469 FA/h |
| Few-shot + negativos minados | 87,5% | — | **0,094 FA/h** |
| **Few-shot + ruido MUSAN + reminado** | **100% (10/10, dejando uno fuera, con ruido no visto)** | — | **0,151 FA/h @0,9** |

Criterio del proyecto: ≥95% de recall con ≤0,5 FA/h. **El recetario few-shot lo cumple a umbral 0,9
(100% de recall, 0,151 FA/h); el zero-shot no cumple ninguno de los dos porque no reconoce a una
persona.** Curva completa de falsos positivos: 0,396 @0,5 · 0,302 @0,7 · 0,245 @0,8 · 0,151 @0,9 ·
0,113 @0,95.

## Los tres hallazgos que ordenan todo

**1. El etiquetado pesa más que la arquitectura.** Etiquetar como positiva toda ventana de un clip
positivo enseña que media frase basta: 0,773 FA/h. Exigiendo que la frase completa quepa en la
ventana: 0,019. Ver `DECISIONES.md` (35).

**2. Un solo motor TTS produce un modelo que reconoce ese motor.** Con positivos solo de Kokoro,
recall 1,000 con sus tres voces —incluida una que nunca vio— y **0,067 con Chatterbox**. Con dos
motores: 0,995. Y aun así, con voz humana real: **0,000**. Ver `docs/hallazgo-voz-humana.md`.

**3. El TTS genera basura silenciosa.** De 434 clips de Chatterbox, 56% inservibles (frase repetida,
fragmentos, repetición truncada tipo "hey celeste … hey cel"). Y lo peor: 170 de 434 aportaban cero
ventanas de entrenamiento sin que el recuento de clips lo delatara.

## Qué hay dentro

| Pieza | Qué hace |
|---|---|
| `benchmark/` | Criterio de éxito ejecutable: curva DET, FA/h por condición, prueba de silencio, manifiesto verificado por hash |
| `pipeline/planificar_lotes.py` | Lotes por horas con normalización de acentos (el campo es texto libre) |
| `pipeline/extraer_caracteristicas.py` | Mel + embedding de openWakeWord → `(n, 96)`, reanudable, con descartes motivados |
| `pipeline/partir_hablantes.py` | Partición disjunta por hablante, congelada con hash |
| `pipeline/preparar_test.py` | Selección del audio de test por condición de acento |
| `pipeline/elegir_frase.py` | Elección de la frase por medición del corpus (frecuencia + colisiones fonéticas) |
| `pipeline/generar_positivos.py` | Positivos con Kokoro (Apache-2.0), voz `es-419` |
| `pipeline/aumentar_positivos.py` | Aumentación de canal: nivel, EQ de micrófono, reverberación, ruido MUSAN |
| `pipeline/validar_positivos.py` | Filtro de clips TTS degradados (una sola frase, 0,9-2,0 s, rachas finas) |
| `pipeline/preparar_entrenamiento.py` | Tensores train/val por voz y por hablante |
| `pipeline/entrenar_cabeza.py` | Cabeza 1536→128→1 sobre embedding congelado, exporta ONNX |
| `pipeline/entrenar_fewshot.py` | Modelo con grabaciones del usuario + TTS + negativos difíciles |
| `pipeline/minar_negativos.py` | Minado de falsos positivos sobre hablantes de entrenamiento |
| `pipeline/medir_test.py` | FA/h por condición sobre el test congelado |
| `pipeline/medir_recall.py` | Curva de recall por umbral, con descarte motivado |
| `pipeline/diagnostico_tts.py` | Control negativo: ¿detecta la frase o "esto es TTS"? |

## Reproducir

```bash
make test        # 12 pruebas del banco de evaluación
make autotest    # demuestra que el criterio puede cumplirse y puede fallar
```

El corpus no se redistribuye: cada usuario descarga Common Voice (CC0) desde Mozilla Data Collective
y este repositorio aporta los scripts, las particiones congeladas y los manifiestos.

## Límites declarados

- El test de 53 h se ha mirado ~10 veces durante el desarrollo: **es un test de desarrollo**, no un
  test final. La cifra publicable debe salir de grabaciones nunca usadas.
- Las 10 grabaciones de voz real son de una sola persona, una sola sesión y un micrófono de laptop.
- El recall del recetario completo está medido en muestra; la cifra honesta es la validación dejando
  un clip fuera.
- Todo el modelado usa el embedding congelado de Google. Si el techo aparece ahí, el siguiente paso
  es entrenar una CNN sobre mel en vez de una cabeza sobre características (candidatos A/B en
  `DECISIONES.md`).

## Documentación

[`PLAN.md`](PLAN.md) · [`DECISIONES.md`](DECISIONES.md) (39 decisiones con criterio y descarte) ·
[`ARQUITECTURA.md`](ARQUITECTURA.md) · [`INVENTARIO.md`](INVENTARIO.md) ·
[`FRASE.md`](FRASE.md) · [`PROGRESO.md`](PROGRESO.md) ·
[`docs/hallazgo-voz-humana.md`](docs/hallazgo-voz-humana.md)

## Licencia

Apache-2.0 (ver `LICENSE` y `NOTICE`). Este repositorio **no redistribuye** audio de Common Voice ni
de otras fuentes.
