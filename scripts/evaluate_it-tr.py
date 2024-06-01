from pathlib import Path
from merge_translate.translator import ItalianTranslator, TurkishTranslator


def evaluate(translator, from_language, to_language):
    output_path = Path.cwd() / "results" / translator.__class__.__name__.lower() / f"{from_language}-{to_language}.json"
    translator.evaluate(from_language=from_language, to_language=to_language, output_path=output_path)


def main():
    translator = ItalianTranslator()
    evaluate(translator=translator, from_language="tr", to_language="it")
    evaluate(translator=translator, from_language="it", to_language="tr")
    del translator

    translator = TurkishTranslator()
    evaluate(translator=translator, from_language="tr", to_language="it")
    evaluate(translator=translator, from_language="it", to_language="tr")
    del translator

    # TODO: complete for merged model
    # translator = MergedTranslator(model_id="")
    # evaluate(translator=translator, from_language="tr", to_language="it")
    # evaluate(translator=translator, from_language="it", to_language="tr")


if __name__ == "__main__":
    main()
