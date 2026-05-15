# Gemma 4 Offline Medical Scribe

A local, privacy-focused medical scribe application that uses **Ollama** and **Gemma 4** to transcribe and summarize medical consultations offline.

## Setup

1. **Install Ollama**: Follow instructions at [ollama.com](https://ollama.com).
2. **Pull Gemma 4**:
   ```bash
   ollama pull gemma:7b  # or whichever version you prefer
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
python scribe.py
```

## Agentic Skills Logic

The scribe includes a lightweight local tool registry for clinical reasoning support:

- `medical_lookup(term)`: local/offline lookup for common medical terms
- `tools`: registry for available agent tools
- `run_agent(prompt)`: enriches prompts with dictionary definitions before SOAP generation

Example:

```python
from scribe import run_agent

prompt = "Analyze the term 'Dyspnea' and include its definition in the patient's summary."
result = run_agent(prompt)
print(f"\nFinal SOAP Note:\n{result}")
```
