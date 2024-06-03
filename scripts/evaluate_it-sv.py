from pathlib import Path
import huggingface_hub as hf_hub
from merge_translate.translator import MistralTranslator, MistralMergedTranslator


def evaluate(translator, source_language, target_language):
    output_path = (
        Path.cwd()
        / "results"
        / translator.model_id.split("/")[-1].lower()
        / f"{source_language}-{target_language}.json"
    )
    translator.evaluate(source_language=source_language, target_language=target_language, output_path=output_path)


def main():
    hf_hub.login()

    translator = MistralTranslator()
    evaluate(translator=translator, source_language="sv", target_language="it")
    evaluate(translator=translator, source_language="it", target_language="sv")
    del translator

    translator = MistralMergedTranslator(model_id="./models/Mistral-7B-Merged-v0.1")
    evaluate(translator=translator, source_language="sv", target_language="it")
    evaluate(translator=translator, source_language="it", target_language="sv")
    del translator


if __name__ == "__main__":
    main()
