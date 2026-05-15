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

## Raspberry Pi Deployment for TFLite Experiments

If you export a `quantized_model.tflite` from `/home/runner/work/gemma4-offline-medical-scribe/gemma4-offline-medical-scribe/the_gemma_4_good_hackathon.ipynb`, you can deploy it to a Raspberry Pi as follows:

1. **Prepare the Raspberry Pi**
   - Install Raspberry Pi OS.
   - Update the system with `sudo apt update && sudo apt upgrade`.
   - Install TensorFlow Lite Runtime with the wheel that matches your Pi and Python version, or use:
     ```bash
     pip install tflite-runtime
     ```

2. **Transfer the model**
   - Copy the model to the device with `scp`, `rsync`, or similar tools.
   - Example:
     ```bash
     scp quantized_model.tflite pi@<raspberry_pi_ip_address>:/home/pi/
     ```

3. **Run inference on the Pi**
   - Create a small Python script that loads `quantized_model.tflite` with `tflite_runtime.interpreter`.
   - Run it with:
     ```bash
     python3 run_inference.py
     ```

See `/home/runner/work/gemma4-offline-medical-scribe/gemma4-offline-medical-scribe/the_gemma_4_good_hackathon.ipynb` for the fuller example workflow and sample inference code.
