# Gemma 4 Offline Medical Scribe

A local, privacy-focused medical scribe application that uses **Ollama** and **Gemma 4 E4B** to transcribe and summarize medical consultations offline.

## Setup

1. **Install Ollama**: Follow instructions at [ollama.com](https://ollama.com).
2. **Pull Gemma 4 E4B**:
   ```bash
   ollama pull gemma4:e4b
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the command-line example:

```bash
python scribe.py
```

Or launch the Streamlit app:

```bash
streamlit run app.py
```

The scribe can use a built-in offline `medical_lookup` tool to resolve common medical terms before producing the final SOAP note.
