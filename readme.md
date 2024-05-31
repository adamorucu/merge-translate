# Model merging for improved language translation

## Methodology

`<language>` = [Italian, Polish]

### Hypothesis

1. The merged model, combining instruction fine-tuned monolingual models for `<language>` and English, can perform translation tasks between these languages better than the component models.
2. The merged model, combining instruction fine-tuned monolingual models for `<language>` and English, can perform translation tasks between these languages as effectively as, or better than, multilingual models like mT5.

### Experiments

Evaluate the following models (in both directions `<language>`->English and vice
versa):
- Instruction Fine-Tuned Monolingual `<language>` Model (prompt-based translation)
- Instruction Fine-Tuned Monolingual English Model (prompt-based translation)
- Merged Fine-Tuned Models 1 and 2 (prompt-based translation)
- Multilingual translation model (out-of-the-box translation, text-to-text generation)

Metrics:
- [BLEU](https://huggingface.co/spaces/evaluate-metric/bleu)
- [METEOR](https://huggingface.co/spaces/evaluate-metric/meteor)
- [CHRF](https://huggingface.co/spaces/evaluate-metric/chrf)

Datasets:
- [OPUS-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
- [Europarl](https://www.statmt.org/europarl/)
- [WMT21](https://huggingface.co/datasets/wmt/wmt21) (or previous years)

Potential models:
- [Multilingual translator IT->EN](https://huggingface.co/Helsinki-NLP/opus-mt-it-en)
- [Multilingual translator EN->IT](https://huggingface.co/Helsinki-NLP/opus-mt-en-it)
- [Multilingual translator PL->EN](https://huggingface.co/Helsinki-NLP/opus-mt-pl-en)
- [Multilingual translator EN->PL](https://huggingface.co/gsarti/opus-mt-tc-en-pl)
- [Monolingual instruction fine-tuned IT]() (TBD)
- [Monolingual instruction fine-tuned PL](https://huggingface.co/Azurro/llama-2-7b-qlora-polish-instruct)

## Merging

Here is an implementation of TIES: https://github.com/arcee-ai/mergekit/blob/main/mergekit/merge_methods/generalized_task_arithmetic.py
