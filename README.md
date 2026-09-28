# Predictive Health Risk Modeling

Academic heart disease screening application using a calibrated SVM pipeline.

## Files

- `heart_disease_model.py`: model, preprocessing, thread-safe cache, and terminal UI.
- `main.py`: Streamlit interface.
- `heart_disease.csv`: local UCI Heart Disease dataset.
- `requirements.txt`: Python dependencies.

## Run

```powershell
python heart_disease_model.py
streamlit run main.py
```

The terminal interface uses numbered menus for categorical values. The Streamlit interface uses dropdowns, toggles, and bounded numeric inputs. Outputs are educational screening estimates and are not clinical diagnoses.
