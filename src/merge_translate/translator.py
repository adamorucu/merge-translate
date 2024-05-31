from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
import transformers as hf_transformers
import datasets as hf_datasets
import evaluate as hf_evaluate


class Translator(ABC):
    """Prompt-based translator."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.pipeline = hf_transformers.pipeline(
            task="text-generation",
            model=model_id,
            max_new_tokens=64,
            device_map="auto",
        )

    @abstractmethod
    def _get_prompt(self, text: str, from_language: str, to_language: str) -> str:
        raise NotImplementedError

    def __call__(self, text: str, from_language: str, to_language: str) -> str:
        prompt = self._get_prompt(text, from_language, to_language)
        output = self.pipeline(prompt, return_full_text=False)
        return output[0]["generated_text"]

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

        metrics = hf_evaluate.combine(["bleu", "meteor", "chrf"])
        for x, y_true in tqdm(data_generator(), total=len(dataset), desc="Translating"):
            y_pred = self(text=x, from_language=from_language, to_language=to_language)
            metrics.add(prediction=y_pred, reference=y_true)
        results = metrics.compute()

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            hf_evaluate.save(output_path, **results)

        return results


class ItalianTranslator(Translator):
    code_to_language = {
        "it": "Italiano",
        "tr": "Turco",
        "en": "Inglese",
    }

    def _get_prompt(self, text: str, from_language: str, to_language: str) -> str:
        from_language = self.code_to_language[from_language]
        to_language = self.code_to_language[to_language]
        instruction = f"Traduci la seguente frase da {from_language} a {to_language}.\n\n{text}\n"

        return (
            "[INST]"
            "<<SYS>>\n"
            "Sei un assistente disponibile, rispettoso e onesto. Rispondi sempre nel modo piu' utile possibile, pur "
            "essendo sicuro. Le risposte non devono includere contenuti dannosi, non etici, razzisti, sessisti, "
            "tossici, pericolosi o illegali. Assicurati che le tue risposte siano socialmente imparziali e positive. "
            "Se una domanda non ha senso o non e' coerente con i fatti, spiegane il motivo invece di rispondere in "
            "modo non corretto. Se non conosci la risposta a una domanda, non condividere informazioni false.\n"
            "<</SYS>>\n\n"
            f"{instruction}"
            "[/INST]"
        )


class TurkishTranslator(Translator):
    code_to_language_from = {
        "tr": "Türkçe'den",
        "it": "İtalyanca'dan",
        "en": "İngilizce'den",
    }
    code_to_language_to = {
        "tr": "Türkçe'ye",
        "it": "İtalyanca'ya",
        "en": "İngilizce'ye",
    }

    def _get_prompt(self, text: str, from_language: str, to_language: str) -> str:
        from_language = self.code_to_language_from[from_language]
        to_language = self.code_to_language_to[to_language]
        return f"<s>[INST] Bu cümleyi {from_language} {to_language} çevir.\n\n{text} [/INST]"
