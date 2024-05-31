from pathlib import Path
from merge_translate.translator import ItalianTranslator, TurkishTranslator


def evaluate(translator):
    output_dir = Path.cwd() / "results" / translator.__class__.__name__
    # translator.evaluate(from_language="en", to_language="it", output_path=output_dir / "it-en.json")
    translator.evaluate(from_language="it", to_language="tr", output_path=output_dir / "it-tr.json")
    translator.evaluate(from_language="tr", to_language="it", output_path=output_dir / "tr-it.json")


def main():
    evaluate(ItalianTranslator(model_id="swap-uniba/LLaMAntino-2-chat-7b-hf-UltraChat-ITA"))
    evaluate(TurkishTranslator(model_id="malhajar/Llama-2-7b-chat-tr"))
    # evaluate(MergedTranslator(model_id="")) # TODO: Add merged model ID


if __name__ == "__main__":
    main()
