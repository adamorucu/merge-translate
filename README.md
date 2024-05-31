# Model merging for improved language translation

## Install

```
conda activate <env>
poetry install
```

If you do not have `poetry`, see the [official documentation](https://python-poetry.org/docs/#installation) for the installation.

## Usage

Interactive translation:
```
python scripts/interact_en-it.py
```

Evaluate:
```
python scripts/evaluate_it-tr.py
```

## Hypothesis

Given two LLMs:
1. Instruction fine-tuned LLM on non-English `<language1>`
2. Instruciton fine-tuned LLM on non-English `<language2>`

Merging LLMs 1 and 2 can perform machine translation between `<language1>` and `<language2>` (both directions) better than the individual LLMs 1 and 2.

## Experiments

`<language1>` = Turkist (TR)  
`<language2>` = Italian (IT)

Models:
1. [Instruction fine-tuned LM on `<language1>`](https://huggingface.co/malhajar/Llama-2-7b-chat-tr) (fine-tuning: llama-2-7b-hf -> Turkist instructions)
2. [Instruction fine-tuned LM on `<language2>`](https://huggingface.co/swap-uniba/LLaMAntino-2-chat-7b-hf-UltraChat-ITA) (fine-tuning: llama-2-7b-chat-hf -> Italian -> Italian instructions)

Metrics:
- [BLEU](https://huggingface.co/spaces/evaluate-metric/bleu)
- [METEOR](https://huggingface.co/spaces/evaluate-metric/meteor)
- [CHRF](https://huggingface.co/spaces/evaluate-metric/chrf)

Datasets:
- [Tatoeba](https://huggingface.co/datasets/Helsinki-NLP/tatoeba)
- [Europarl](https://huggingface.co/datasets/Helsinki-NLP/europarl)

## Merging

Here is an implementation of TIES: https://github.com/arcee-ai/mergekit/blob/main/mergekit/merge_methods/generalized_task_arithmetic.py
