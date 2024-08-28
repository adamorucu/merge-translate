from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
import torch as th
import transformers as hf_transformers
import datasets as hf_datasets
import evaluate as hf_evaluate
import nltk


class Translator(ABC):
    """Prompt-based translator."""

    def __init__(self):
        nltk.download("punkt_tab")

    @abstractmethod
    def __call__(self, text: str, source_language: str, target_language: str) -> str:
        raise NotImplementedError

    def evaluate(
        self, source_language: str, target_language: str, output_path: str | Path | None = None
    ) -> dict[str, float]:
        lang1, lang2 = (
            (source_language, target_language)
            if source_language < target_language
            else (target_language, source_language)
        )
        dataset = hf_datasets.load_dataset(
            "Helsinki-NLP/tatoeba",
            lang1=lang1,
            lang2=lang2,
            split="train",
            trust_remote_code=True,
        )["translation"]

        def data_generator():
            for translation in dataset:
                yield translation[source_language], translation[target_language]

        metrics = hf_evaluate.combine(["bleu", "sacrebleu", "meteor", "chrf"])
        for x, y_true in tqdm(data_generator(), total=len(dataset), desc="Translating"):
            y_pred = self(text=x, source_language=source_language, target_language=target_language)
            metrics.add(prediction=y_pred, reference=y_true)
        results = metrics.compute()

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            hf_evaluate.save(output_path, **results)

        return results


class MistralTranslator(Translator):
    code_target_language = {
        "it": "Italian",
        "sv": "Swedish",
        "en": "English",
    }

    prompt_template = (
        "Translate the following sentence from {source_language} to {target_language} without adding any contextual "
        "information: '{text}'"
    )

    def __init__(self):
        super().__init__()
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        self.device = "cuda" if th.cuda.is_available() else "cpu"
        self.model = hf_transformers.AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device).eval()
        self.tokenizer = hf_transformers.AutoTokenizer.from_pretrained(self.model_id)

    def __call__(self, text: str, source_language: str, target_language: str) -> str:
        prompt = self.prompt_template.format(
            source_language=self.code_target_language[source_language],
            target_language=self.code_target_language[target_language],
            text=text,
        )
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0, input_ids.shape[1] :]
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        translation = decoded.strip("\"'")
        return translation


class MistralMergedTranslator(MistralTranslator):
    def __init__(self, model_id: str):
        # Due to an issue in mergekit, the merged model doesn't have the correct chat template.
        # We need to manually set the chat template by using the one from Mistral-7B-Instruct-v0.1.
        super().__init__()  # this will load the Mistral-7B-Instruct-v0.1 model
        chat_template = self.tokenizer.chat_template  # cache the chat template

        # Now, we can load the correct model and tokenizer
        self.model_id = model_id
        self.model = hf_transformers.AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device).eval()
        self.tokenizer = hf_transformers.AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.chat_template = chat_template  # set the correct chat template
