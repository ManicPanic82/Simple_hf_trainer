# Ollama Export

The **Ollama Export** tab converts a local, merged Hugging Face model into GGUF with
[llama.cpp](https://github.com/ggml-org/llama.cpp), optionally quantises it, then imports and
verifies it with the local Ollama CLI.

## Prerequisites

1. Install Ollama for Windows and make sure its local service is running.
2. Obtain/build llama.cpp for Windows. The workflow needs:
   - `convert_hf_to_gguf.py`
   - `llama-quantize.exe` for Q4_K_M, Q5_K_M or Q8_0
3. Select a Python interpreter that has the dependencies required by the llama.cpp converter.
   Keep this separate from the lightweight application environment if it requires PyTorch.

## Supported source models

The selected folder must contain a Hugging Face `config.json` and full model weights.
A LoRA adapter folder is not a standalone model and is rejected. In Training, choose
**Merged model (16-bit)** when exporting if the resulting model is intended for Ollama.
The original model folder is read-only throughout this workflow.

## Conversion

1. Select the merged model folder.
2. Select the Python interpreter, llama.cpp converter script, quantizer executable and
   `ollama.exe` paths.
3. Choose an output directory and a lowercase Ollama model name.
4. Choose Q4_K_M, Q5_K_M, Q8_0 or F16.
5. Start conversion. The tab runs each external process asynchronously and shows its output:
   - Hugging Face to F16 GGUF
   - Optional GGUF quantisation
   - `ollama create` using a generated Modelfile
   - `ollama show` verification

The output directory retains the intermediate F16 GGUF (when quantising), final GGUF and
Modelfile. Nothing is automatically deleted. If a stage fails, its output is shown in the
conversion log and partial files are retained for inspection.

## Quantisation notes

- **Q4_K_M** is the suggested starting point for smaller local deployments.
- **Q5_K_M** trades a larger file for additional precision.
- **Q8_0** is larger and usually needs more memory.
- **F16** skips quantisation and keeps the full-precision GGUF.

Quantisation size is not a guarantee of runtime memory use. Ollama context length, GPU layer
offload, KV cache and available system/GPU memory also affect whether a model runs well.
