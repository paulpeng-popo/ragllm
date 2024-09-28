import deepl


def translate(
    text: str,
    source_language: str = "ZH",
    target_language: str = "EN"
) -> str:
    return deepl.translate(
        source_language=source_language,
        target_language=target_language,
        text=text
    )


if __name__ == "__main__":
    print(translate("根管治療"))
