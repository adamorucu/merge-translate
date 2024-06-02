from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
import transformers as hf_transformers
import datasets as hf_datasets
import evaluate as hf_evaluate


class Translator(ABC):
    """Prompt-based translator."""

    @abstractmethod
    def __call__(self, text: str, from_language: str, to_language: str) -> str:
        raise NotImplementedError

    def evaluate(self, from_language: str, to_language: str, output_path: str | Path | None = None) -> dict[str, float]:
        lang1, lang2 = (from_language, to_language) if from_language < to_language else (to_language, from_language)
        dataset = hf_datasets.load_dataset(
            "Helsinki-NLP/tatoeba",
            lang1=lang1,
            lang2=lang2,
            split="train",
            trust_remote_code=True,
        )["translation"]

        dataset = dataset[:100]  # TODO: for draft, a subset is ok... later we will use the full dataset

        def data_generator():
            for translation in dataset:
                yield translation[from_language], translation[to_language]

        metrics = hf_evaluate.combine(["bleu", "sacrebleu", "meteor", "chrf"])
        for x, y_true in tqdm(data_generator(), total=len(dataset), desc="Translating"):
            y_pred = self(text=x, from_language=from_language, to_language=to_language)
            metrics.add(prediction=y_pred, reference=y_true)
        results = metrics.compute()

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            hf_evaluate.save(output_path, **results)

        return results


class MistralTranslator(Translator):
    code_to_language = {
        "it": "Italian",
        "pl": "Polish",
        "en": "English",
    }

    prompt_template = (
        "Translate the following sentence from {from_language} to {to_language} without adding any contextual "
        "information: '{text}'"
    )

    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        super().__init__()
        self.model_id = model_id
        self.model = hf_transformers.AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = hf_transformers.AutoTokenizer.from_pretrained(self.model_id)

    def __call__(self, text: str, from_language: str, to_language: str) -> str:
        prompt = self.prompt_template.format(
            from_language=self.code_to_language[from_language],
            to_language=self.code_to_language[to_language],
            text=text,
        )
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0, input_ids.shape[1] :]
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        translation = decoded.replace("'", "").replace('"', "")
        return translation
