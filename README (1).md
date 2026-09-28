# INSM 7020 - Asignacion 2

Analisis reproducible sobre selectividad institucional e ingresos posteriores de graduados de Computer Science usando College Scorecard.

## Reproducir el analisis

```bash
python -m pip install -r requirements.txt
python Joel_Vega_INSM7020_Asignacion2_Analisis.py
```

Los dos CSV incluidos son extractos reproducibles de los archivos oficiales `FieldOfStudyData2122_2223_PP.csv` y `MERGED2022_23_PP.csv` de College Scorecard, limitados a las columnas y unidades necesarias para este analisis. La fuente oficial se documenta en el informe.

El workflow `.github/workflows/run-analysis.yml` ejecuta el analisis automaticamente en GitHub Actions y guarda las salidas como artifact.
