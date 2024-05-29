# Model merging for improved language translation

**Potential models**
- [Polish - LLaMa2 7B](https://huggingface.co/Azurro/llama-2-7b-qlora-polish-instruct) 
- [Polish - LLaMa2 7B](https://huggingface.co/Aspik101/Llama-2-7b-hf-instruct-pl-lora_unload)
- [Turkish - LLaMa2 7B](https://huggingface.co/Trendyol/Trendyol-LLM-7b-chat-v0.1)
- [Italian - Mistral 7B](https://huggingface.co/scribis/Fantastica-7b-Instruct-0.2-Italian)

**Datsets**
- https://tatoeba.org/en/downloads
- https://www.statmt.org/europarl/


## Merging
Here is an implementation of TIES: https://github.com/arcee-ai/mergekit/blob/main/mergekit/merge_methods/generalized_task_arithmetic.py
