from evaluate import combine
from datasets import load_dataset


def my_evaluate(translator):
    dataset = load_dataset("Helsinki-NLP/opus-100", name="en-it", split="test")
    metrics = combine(["bleu", "meteor"])
    predictions = translator(dataset)

    metrics.compute(predictions=predictions, references=references)
