from merge_translate.translator import TurkishTranslator


def main():
    translator = TurkishTranslator()
    while True:
        text = input("Enter text to translate: ")
        translation = translator(text=text, from_language="en", to_language="tr")
        print(f"Translation: {translation}")


if __name__ == "__main__":
    main()
