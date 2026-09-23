from src.data.text import CharacterTokenizer, normalize_text


def test_normalization_and_tokenization() -> None:
    tokenizer = CharacterTokenizer()
    assert normalize_text("  Hello,   Wörld!  ") == "hello, w rld!"
    ids = tokenizer.encode("Hello!")
    assert ids
    assert max(ids) < tokenizer.vocab_size
    assert tokenizer.decode(ids) == "hello!"

