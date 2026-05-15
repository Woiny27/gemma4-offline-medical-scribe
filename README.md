# Gemma 4 Offline Medical Scribe

A privacy-focused medical scribe that runs locally with Ollama + Gemma and supports OCR for paper charts.

## Setup

1. Install Ollama from [ollama.com](https://ollama.com)
2. Pull the model:
   ```bash
   ollama pull gemma4:e4b
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Example usage

```python
from scribe import run_agent

prompt = "Analyze the term 'Dyspnea' and include its definition in the patient's summary."
result = run_agent(prompt)
print(f"\nFinal SOAP Note:\n{result}")
```

## OCR chart transcription

```python
from scribe import process_chart

result = process_chart("patient_chart.jpg")
print(result)
```

## Streamlit app

```bash
streamlit run app.py
```
