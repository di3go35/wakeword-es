# Frase de activación — congelada

**Frase:** `hey celeste`

**Fecha de congelación:** 2026-10-24 · **Decisión:** 33

---

## Por qué esta y no otra

La frase no se eligió por gusto: se midió sobre las **1.082.797 frases / 10,9 millones de palabras**
de Common Voice es 27.0, porque una wake word tiene que ser **rara en habla normal** y
**fonéticamente distintiva**. Las tres métricas:

1. **`freq_bigrama`** — veces por millón que aparece la frase completa.
2. **`freq_nombre`** — veces por millón que aparece la palabra distintiva sola.
3. **`colisiones`** — palabras del corpus que comparten el **esqueleto consonántico** del nombre
   (`celeste` → `slst`), ponderadas por frecuencia. Es un proxy de "suena parecido a algo común".

Hallazgo que ordenó la decisión: **ningún "oye X" ni "hey X" aparece ni una vez** en 10,9 millones
de palabras. La frase completa nunca es el problema; lo que decide es el nombre.

| Frase | Sílabas | Nombre (por millón) | Colisiones por millón |
|---|---|---|---|
| **hey celeste** | **4** | 16,61 | **127,34** |
| oye celeste | 5 | 16,61 | 127,34 |
| oye vega | 4 | 32,75 | 120,00 |
| oye alma | 4 | 39,63 | 410,73 |
| oye emma | 4 | 17,52 | 754,57 |
| oye luna | 4 | 51,38 | 811,82 |
| oye maya | 4 | 18,16 | 1.746,85 |
| oye aurora | 5 | 15,14 | 4.147,63 |
| oye lía | 3 | 1,19 | 83.645,59 |
| oye cóndor | 4 | 4,95 | 9,45 |
| oye wayra | 4 | 0,00 | 0,00 |

`celeste` es la mejor de las propuestas de nombre propio común: su esqueleto `slst` es raro, con
**6× menos colisiones que `emma`** (`m`, una sola consonante) y **6,4× menos que `luna`** (`ln`).
`lía` (`l`) y `zoe` (`s`) quedan descartados por medición: suenan consonánticamente igual que el
8% del idioma.

## Tres decisiones dentro de la decisión

**1. Con disparador, no el nombre solo.** Una sola palabra de dos sílabas da la mitad de evidencia
acústica al detector y, medido, empeora: `condor` sola puntúa 8,32 y `wayra` sola 4,00, **peor que
cualquier frase de dos palabras** de la tabla, aun con esqueletos rarísimos.

**2. `hey` en vez de `oye`.** `oye` es 14,13 por millón **en texto leído** (como verbo, *oye la
música*), y en habla real es mucho más frecuente todavía porque es el vocativo más usado del
español. `hey` aparece 6,15 por millón y suena natural en la calle. Además deja la frase en **4
sílabas** en vez de 5: se activa antes.

**3. Nombre del español común, no quechua.** `wayra` y `sumaq` puntúan 0,00 colisiones, pero ese
cero viene justamente de sus grafías (`w`, `q`) ajenas al español: es el mismo motivo por el que un
TTS puede pronunciarlas mal y contaminar los positivos de entrenamiento con audio defectuoso. Un
0,00 medido sobre texto no compensa un sintetizador que falla.

## Regla de congelación

- `hey celeste` **no se cambia** después de congelar el test. Si se cambia, el test se rehace.
- La evaluación de candidatas es reproducible: `pipeline/elegir_frase.py` con
  `--disparadores "oye,hey" --solo-nombres`.
- Los números en bruto están en `benchmark/candidatas_frase.json` y `benchmark/candidatas2.json`.
- Hash del acta: `benchmark/FRASE_HASH`.

## Limitación declarada

Las colisiones son un **proxy sobre texto**, no una medición acústica. La comprobación real es el
test congelado: activaciones por hora sobre 53 h de audio negativo real en 10 condiciones de acento.
Si `hey celeste` sale peor de lo que predice esta tabla, manda el test, no esta tabla.
