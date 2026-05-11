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
