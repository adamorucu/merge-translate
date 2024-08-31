# Model merging for improved language translation

## Install

Clone:

```bash
git pull --recurse-submodules git@github.com:adamorucu/merge-translate.git
```

```bash
poetry install
poetry shell
```

If you do not have `poetry`, see the [official documentation](https://python-poetry.org/docs/#installation) for the installation.

## Usage

Merge model:

```bash
bash ./scripts/merge.sh
```

Interactive translation (Italian -> Swedish):

```bash
python scripts/interact_it_sv.py
```

Evaluate (Italian-Swedish, both ways):

```bash
python scripts/evaluate_it_sv.py
```

Note: The scripts require an access token (needed to use [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)).

## Hypothesis

Given these LLMs:

1. Pre-trained LLM
2. Fine-tuned version of (1) on English instruction-following
3. Fine-tuned version of (1) on non-English `<language1>`
4. Fine-tuned version of (1) on non-English `<language2>`

Merging LLMs (2)-(4) can perform prompt-based machine translation between `<language1>` and `<language2>` (both directions) better than the individual LLMs (2)-(4).

## Experiments

`<language1>` = Italian (IT)  
`<language2>` = Sweden (SV)

Models:

1. [Pre-trained LLM](https://huggingface.co/mistralai/Mistral-7B-v0.1)
2. [Fine-tuned version of (1) on English instruction-following](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
3. [Fine-tuned version of (1) on non-English `<language1>`](https://huggingface.co/DeepMount00/Mistral-Ita-7b)
4. [Fine-tuned version of (1) on non-English `<language2>`](https://huggingface.co/timpal0l/Mistral-7B-v0.1-flashback-v2)

Metrics:

- [BLEU](https://huggingface.co/spaces/evaluate-metric/bleu)
- [METEOR](https://huggingface.co/spaces/evaluate-metric/meteor)
- [CHRF](https://huggingface.co/spaces/evaluate-metric/chrf)

Datasets:

- [Tatoeba](https://huggingface.co/datasets/Helsinki-NLP/tatoeba)
- [Europarl](https://huggingface.co/datasets/Helsinki-NLP/europarl)
