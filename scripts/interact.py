import huggingface_hub as hf_hub
from merge_translate.translator import MistralTranslator, MistralMergedTranslator

USE_MERGED = True  # Set to False to use the base model


def main():
    hf_hub.login()

    translator = MistralMergedTranslator("./models/Mistral-7B-Merged-v0.1") if USE_MERGED else MistralTranslator()
    print("Welcome to the Italian-Swedish translator!")
    while True:
        text = input("Enter text to translate: ")
        translation = translator(text=text, source_language="it", target_language="sv")
        print(f"Translation: {translation}")


if __name__ == "__main__":
    main()
