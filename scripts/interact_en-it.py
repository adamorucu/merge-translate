from merge_translate.translator import ItalianTranslator


def main():
    translator = ItalianTranslator(model_id="swap-uniba/LLaMAntino-2-chat-7b-hf-UltraChat-ITA")
    while True:
        text = input("Enter text to translate: ")
        translation = translator(text=text, from_language="en", to_language="it")
        print(f"Translation: {translation}")


if __name__ == "__main__":
    main()
