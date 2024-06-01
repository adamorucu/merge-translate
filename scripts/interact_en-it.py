from merge_translate.translator import ItalianTranslator


def main():
    translator = ItalianTranslator()
    while True:
        text = input("Enter text to translate: ")
        translation = translator(text=text, from_language="en", to_language="it")
        print(f"Translation: {translation}")


if __name__ == "__main__":
    main()
