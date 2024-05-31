from tqdm import tqdm
import transformers as hf_transformers
import evaluate as hf_evaluate
import datasets as hf_datasets


def evaluate(translator, dataset):
    metrics = hf_evaluate.combine(["bleu", "meteor", "chrf"])
    progress_bar = tqdm(total=len(dataset), desc="Translating")
    batch_size = 32

    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch = dataset[batch_start:batch_end]

        x = [translation["en"] for translation in batch]
        y_true = [translation["it"] for translation in batch]
        y_pred = translator(x)
        y_pred = [translation["translation_text"] for translation in y_pred]

        metrics.add_batch(predictions=y_pred, references=y_true)

        progress_bar.update(len(batch))
    progress_bar.close()

    return metrics.compute()


def main():
    dataset = hf_datasets.load_dataset("Helsinki-NLP/opus-100", name="en-it", split="test")
    dataset = dataset["translation"]

    translator = hf_transformers.pipeline("translation", model="Helsinki-NLP/opus-mt-en-it", device_map="auto")

    results = evaluate(translator, dataset)
    print(results)


if __name__ == "__main__":
    main()
