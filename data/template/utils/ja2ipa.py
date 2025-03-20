# ja2ipa.py
import argparse
import json
import sys
from collections import OrderedDict
from typing import Tuple, Optional

from tqdm import tqdm
import pykakasi.kakasi as kakasi

# 1) Attempt optional imports
try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# ========== Kakasi Converter Setup ==========
kks = kakasi()
kks.setMode('J', 'H')  # Kanji -> Hiragana
kks.setMode('H', 'H')  # Hiragana -> Hiragana
kks.setMode('K', 'H')  # Katakana -> Hiragana
conv = kks.getConverter()

kana_mapper = OrderedDict([
    ("ゔぁ","bˈa"),
    ("ゔぃ","bˈi"),
    ("ゔぇ","bˈe"),
    ("ゔぉ","bˈo"),
    ("ゔゃ","bˈʲa"),
    ("ゔゅ","bˈʲɯ"),
    ("ゔゃ","bˈʲa"),
    ("ゔょ","bˈʲo"),

    ("ゔ","bˈɯ"),

    ("あぁ","aː"),
    ("いぃ","iː"),
    ("いぇ","je"),
    ("いゃ","ja"),
    ("うぅ","ɯː"),
    ("えぇ","eː"),
    ("おぉ","oː"),
    ("かぁ","kˈaː"),
    ("きぃ","kˈiː"),
    ("くぅ","kˈɯː"),
    ("くゃ","kˈa"),
    ("くゅ","kˈʲɯ"),
    ("くょ","kˈʲo"),
    ("けぇ","kˈeː"),
    ("こぉ","kˈoː"),
    ("がぁ","gˈaː"),
    ("ぎぃ","gˈiː"),
    ("ぐぅ","gˈɯː"),
    ("ぐゃ","gˈʲa"),
    ("ぐゅ","gˈʲɯ"),
    ("ぐょ","gˈʲo"),
    ("げぇ","gˈeː"),
    ("ごぉ","gˈoː"),
    ("さぁ","sˈaː"),
    ("しぃ","ɕˈiː"),
    ("すぅ","sˈɯː"),
    ("すゃ","sˈʲa"),
    ("すゅ","sˈʲɯ"),
    ("すょ","sˈʲo"),
    ("せぇ","sˈeː"),
    ("そぉ","sˈoː"),
    ("ざぁ","zˈaː"),
    ("じぃ","dʑˈiː"),
    ("ずぅ","zˈɯː"),
    ("ずゃ","zˈʲa"),
    ("ずゅ","zˈʲɯ"),
    ("ずょ","zˈʲo"),
    ("ぜぇ","zˈeː"),
    ("ぞぉ","zˈeː"),
    ("たぁ","tˈaː"),
    ("ちぃ","tɕˈiː"),
    ("つぁ","tsˈa"),
    ("つぃ","tsˈi"),
    ("つぅ","tsˈɯː"),
    ("つゃ","tɕˈa"),
    ("つゅ","tɕˈɯ"),
    ("つょ","tɕˈo"),
    ("つぇ","tsˈe"),
    ("つぉ","tsˈo"),
    ("てぇ","tˈeː"),
    ("とぉ","tˈoː"),
    ("だぁ","dˈaː"),
    ("ぢぃ","dʑˈiː"),
    ("づぅ","dˈɯː"),
    ("づゃ","zˈʲa"),
    ("づゅ","zˈʲɯ"),
    ("づょ","zˈʲo"),
    ("でぇ","dˈeː"),
    ("どぉ","dˈoː"),
    ("なぁ","nˈaː"),
    ("にぃ","nˈiː"),
    ("ぬぅ","nˈɯː"),
    ("ぬゃ","nˈʲa"),
    ("ぬゅ","nˈʲɯ"),
    ("ぬょ","nˈʲo"),
    ("ねぇ","nˈeː"),
    ("のぉ","nˈoː"),
    ("はぁ","hˈaː"),
    ("ひぃ","çˈiː"),
    ("ふぅ","ɸˈɯː"),
    ("ふゃ","ɸˈʲa"),
    ("ふゅ","ɸˈʲɯ"),
    ("ふょ","ɸˈʲo"),
    ("へぇ","hˈeː"),
    ("ほぉ","hˈoː"),
    ("ばぁ","bˈaː"),
    ("びぃ","bˈiː"),
    ("ぶぅ","bˈɯː"),
    ("ふゃ","ɸˈʲa"),
    ("ぶゅ","bˈʲɯ"),
    ("ふょ","ɸˈʲo"),
    ("べぇ","bˈeː"),
    ("ぼぉ","bˈoː"),
    ("ぱぁ","pˈaː"),
    ("ぴぃ","pˈiː"),
    ("ぷぅ","pˈɯː"),
    ("ぷゃ","pˈʲa"),
    ("ぷゅ","pˈʲɯ"),
    ("ぷょ","pˈʲo"),
    ("ぺぇ","pˈeː"),
    ("ぽぉ","pˈoː"),
    ("まぁ","mˈaː"),
    ("みぃ","mˈiː"),
    ("むぅ","mˈɯː"),
    ("むゃ","mˈʲa"),
    ("むゅ","mˈʲɯ"),
    ("むょ","mˈʲo"),
    ("めぇ","mˈeː"),
    ("もぉ","mˈoː"),
    ("やぁ","jˈaː"),
    ("ゆぅ","jˈɯː"),
    ("ゆゃ","jˈaː"),
    ("ゆゅ","jˈɯː"),
    ("ゆょ","jˈoː"),
    ("よぉ","jˈoː"),
    ("らぁ","ɽˈaː"),
    ("りぃ","ɽˈiː"),
    ("るぅ","ɽˈɯː"),
    ("るゃ","ɽˈʲa"),
    ("るゅ","ɽˈʲɯ"),
    ("るょ","ɽˈʲo"),
    ("れぇ","ɽˈeː"),
    ("ろぉ","ɽˈoː"),
    ("わぁ","ɯˈaː"),
    ("をぉ","oː"),

    ("う゛","bˈɯ"),
    ("でぃ","dˈi"),
    ("でぇ","dˈeː"),
    ("でゃ","dˈʲa"),
    ("でゅ","dˈʲɯ"),
    ("でょ","dˈʲo"),
    ("てぃ","tˈi"),
    ("てぇ","tˈeː"),
    ("てゃ","tˈʲa"),
    ("てゅ","tˈʲɯ"),
    ("てょ","tˈʲo"),
    ("すぃ","sˈi"),
    ("ずぁ","zˈɯa"),
    ("ずぃ","zˈi"),
    ("ずぅ","zˈɯ"),
    ("ずゃ","zˈʲa"),
    ("ずゅ","zˈʲɯ"),
    ("ずょ","zˈʲo"),
    ("ずぇ","zˈe"),
    ("ずぉ","zˈo"),
    ("きゃ","kˈʲa"),
    ("きゅ","kˈʲɯ"),
    ("きょ","kˈʲo"),
    ("しゃ","ɕˈʲa"),
    ("しゅ","ɕˈʲɯ"),
    ("しぇ","ɕˈʲe"),
    ("しょ","ɕˈʲo"),
    ("ちゃ","tɕˈa"),
    ("ちゅ","tɕˈɯ"),
    ("ちぇ","tɕˈe"),
    ("ちょ","tɕˈo"),
    ("とぅ","tˈɯ"),
    ("とゃ","tˈʲa"),
    ("とゅ","tˈʲɯ"),
    ("とょ","tˈʲo"),
    ("どぁ","dˈoa"),
    ("どぅ","dˈɯ"),
    ("どゃ","dˈʲa"),
    ("どゅ","dˈʲɯ"),
    ("どょ","dˈʲo"),
    ("どぉ","dˈoː"),
    ("にゃ","nˈʲa"),
    ("にゅ","nˈʲɯ"),
    ("にょ","nˈʲo"),
    ("ひゃ","çˈʲa"),
    ("ひゅ","çˈʲɯ"),
    ("ひょ","çˈʲo"),
    ("みゃ","mˈʲa"),
    ("みゅ","mˈʲɯ"),
    ("みょ","mˈʲo"),
    ("りゃ","ɽˈʲa"),
    ("りぇ","ɽˈʲe"),
    ("りゅ","ɽˈʲɯ"),
    ("りょ","ɽˈʲo"),
    ("ぎゃ","gˈʲa"),
    ("ぎゅ","gˈʲɯ"),
    ("ぎょ","gˈʲo"),
    ("ぢぇ","dʑˈe"),
    ("ぢゃ","dʑˈa"),
    ("ぢゅ","dʑˈɯ"),
    ("ぢょ","dʑˈo"),
    ("じぇ","dʑˈe"),
    ("じゃ","dʑˈa"),
    ("じゅ","dʑˈɯ"),
    ("じょ","dʑˈo"),
    ("びゃ","bˈʲa"),
    ("びゅ","bˈʲɯ"),
    ("びょ","bˈʲo"),
    ("ぴゃ","pˈʲa"),
    ("ぴゅ","pˈʲɯ"),
    ("ぴょ","pˈʲo"),
    ("うぁ","ɯˈa"),
    ("うぃ","ɯˈi"),
    ("うぇ","ɯˈe"),
    ("うぉ","ɯˈo"),
    ("うゃ","ɯˈʲa"),
    ("うゅ","ɯˈʲɯ"),
    ("うょ","ɯˈʲo"),
    ("ふぁ","ɸˈa"),
    ("ふぃ","ɸˈi"),
    ("ふぅ","ɸˈɯ"),
    ("ふゃ","ɸˈʲa"),
    ("ふゅ","ɸˈʲɯ"),
    ("ふょ","ɸˈʲo"),
    ("ふぇ","ɸˈe"),
    ("ふぉ","ɸˈo"),

    ("あ","a"),
    ("い","i"),
    ("う","ɯ"),
    ("え","e"),
    ("お","o"),
    ("か","kˈa"),
    ("き","kˈi"),
    ("く","kˈɯ"),
    ("け","kˈe"),
    ("こ","kˈo"),
    ("さ","sˈa"),
    ("し","ɕˈi"),
    ("す","sˈɯ"),
    ("せ","sˈe"),
    ("そ","sˈo"),
    ("た","tˈa"),
    ("ち","tɕˈi"),
    ("つ","tsˈɯ"),
    ("て","tˈe"),
    ("と","tˈo"),
    ("な","nˈa"),
    ("に","nˈi"),
    ("ぬ","nˈɯ"),
    ("ね","nˈe"),
    ("の","nˈo"),
    ("は","hˈa"),
    ("ひ","çˈi"),
    ("ふ","ɸˈɯ"),
    ("へ","hˈe"),
    ("ほ","hˈo"),
    ("ま","mˈa"),
    ("み","mˈi"),
    ("む","mˈɯ"),
    ("め","mˈe"),
    ("も","mˈo"),
    ("ら","ɽˈa"),
    ("り","ɽˈi"),
    ("る","ɽˈɯ"),
    ("れ","ɽˈe"),
    ("ろ","ɽˈo"),
    ("が","gˈa"),
    ("ぎ","gˈi"),
    ("ぐ","gˈɯ"),
    ("げ","gˈe"),
    ("ご","gˈo"),
    ("ざ","zˈa"),
    ("じ","dʑˈi"),
    ("ず","zˈɯ"),
    ("ぜ","zˈe"),
    ("ぞ","zˈo"),
    ("だ","dˈa"),
    ("ぢ","dʑˈi"),
    ("づ","zˈɯ"),
    ("で","dˈe"),
    ("ど","dˈo"),
    ("ば","bˈa"),
    ("び","bˈi"),
    ("ぶ","bˈɯ"),
    ("べ","bˈe"),
    ("ぼ","bˈo"),
    ("ぱ","pˈa"),
    ("ぴ","pˈi"),
    ("ぷ","pˈɯ"),
    ("ぺ","pˈe"),
    ("ぽ","pˈo"),
    ("や","jˈa"),
    ("ゆ","jˈɯ"),
    ("よ","jˈo"),
    ("わ","ɯˈa"),
    ("ゐ","i"),
    ("ゑ","e"),
    ("ん","ɴ"),
    ("っ","ʔ"),
    ("ー","ː"),

    ("ぁ","a"),
    ("ぃ","i"),
    ("ぅ","ɯ"),
    ("ぇ","e"),
    ("ぉ","o"),
    ("ゎ","ɯˈa"),
    ("ぉ","o"),

    ("を","o")
])

nasal_sound = OrderedDict([
    # before m, p, b
    ("ɴm","mm"),
    ("ɴb","mb"),
    ("ɴp","mp"),

    # before k, g
    ("ɴk","ŋk"),
    ("ɴg","ŋg"),

    # before t, d, n, s, z, ɽ
    ("ɴt","nt"),
    ("ɴd","nd"),
    ("ɴn","nn"),
    ("ɴs","ns"),
    ("ɴz","nz"),
    ("ɴɽ","nɽ"),

    ("ɴɲ","ɲɲ"),
])

# ========== Basic Conversions ==========
def to_hiragana(text: str) -> str:
    """Convert JP text to Hiragana via Kakasi."""
    return conv.do(text)

def hiragana_to_ipa(text: str) -> str:
    """Convert Hiragana to IPA using kana_mapper + nasal_sound."""
    for k, v in kana_mapper.items():
        text = text.replace(k, v)
    for k, v in nasal_sound.items():
        text = text.replace(k, v)
    return text


# ========== 2) MeCab Morphological Tokenization ==========
def mecab_spaced_reading(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Use MeCab for morphological analysis. Return four strings:
      1) spaced_original: original surface forms joined by spaces.
      2) spaced_hira_subbed: token text with the "は" particle overridden to "わ" where applicable, then converted to Hiragana.
      3) spaced_hira_original: the Hiragana conversion of the original spaced text.
      4) pos_tags: part-of-speech tags for each token (joined by spaces).
    If MeCab is not available, return (None, None, None, None).
    """
    if not MECAB_AVAILABLE:
        return None, None, None, None

    tagger = MeCab.Tagger()
    node = tagger.parseToNode(text)

    tokens_original = []
    tokens_for_hira = []
    pos_tokens = []

    while node:
        surface = node.surface
        features = node.feature.split(",")
        if len(features) >= 1:
            pos = features[0]  # e.g. 助詞, 名詞, 動詞...
            tokens_original.append(surface)
            pos_tokens.append(pos)
            # Override if particle "は" (助詞)
            if pos == "助詞" and surface == "は":
                tokens_for_hira.append("わ")
            else:
                tokens_for_hira.append(surface)
        else:
            tokens_original.append(surface)
            tokens_for_hira.append(surface)
            pos_tokens.append("UNK")
        node = node.next

    spaced_original = " ".join(tokens_original)
    spaced_for_hira = " ".join(tokens_for_hira)
    pos_tags = " ".join(pos_tokens)
    spaced_hira_subbed = to_hiragana(spaced_for_hira)
    # spaced_hira_original is computed here if needed later.
    spaced_hira_original = to_hiragana(spaced_original)

    return spaced_original, spaced_hira_subbed, spaced_hira_original, pos_tags


# ========== 3) spaCy Morphological Tokenization ==========
_spacy_nlp = None
def load_spacy_japanese():
    """
    Lazy-load the spaCy model. Requires 'ja_core_news_sm' or similar to be installed.
    """
    global _spacy_nlp
    if _spacy_nlp is None:
        _spacy_nlp = spacy.load("ja_core_news_sm")
    return _spacy_nlp

def spacy_spaced_reading(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Use spaCy morphological analysis. Return four strings:
      1) spaced_original: original token texts joined by spaces.
      2) spaced_hira_subbed: token texts (with "は" overridden to "わ" when pos_ is ADP) converted to Hiragana.
      3) spaced_hira_original: Hiragana conversion of the original spaced token texts.
      4) pos_tags: part-of-speech tags (using token.pos_) joined by spaces.
    If spaCy is not available, return (None, None, None, None).
    """
    if not SPACY_AVAILABLE:
        return None, None, None, None

    nlp = load_spacy_japanese()
    doc = nlp(text)

    tokens_original = []
    tokens_for_hira = []
    pos_tokens = []

    for token in doc:
        tokens_original.append(token.text)
        pos_tokens.append(token.pos_)
        if token.text == "は" and token.pos_ == "ADP":
            tokens_for_hira.append("わ")
        else:
            tokens_for_hira.append(token.text)

    spaced_original = " ".join(tokens_original)
    spaced_for_hira = " ".join(tokens_for_hira)
    pos_tags = " ".join(pos_tokens)
    spaced_hira_subbed = to_hiragana(spaced_for_hira)
    spaced_hira_original = to_hiragana(spaced_original)

    return spaced_original, spaced_hira_subbed, spaced_hira_original, pos_tags


# ========== 4) Unified "get spaced reading" function ==========
def get_spaced_reading(text: str, method: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Return (spaced_original, spaced_hira_subbed, spaced_hira_original, pos_tags) using the chosen method.
    """
    if method == "mecab":
        return mecab_spaced_reading(text)
    elif method == "spacy":
        return spacy_spaced_reading(text)
    else:
        return None, None, None, None


# ========== 5) Main Processing Logic ==========
def process_japanese_text(
    input_file: str,
    output_file: str,
    json_inplace_update: bool = False,
    use_mecab: bool = False,
    use_spacy: bool = False,
):
    """
    Processes Japanese text to IPA.
    
    Whether the input is a JSON array (with each object having a "sentence" field) or plain text (one sentence per line),
    the output is a valid JSON array. Each output object has the following fields:
      - sentence             : original sentence
      - unspaced_ipa         : IPA conversion of the unspaced (raw) Hiragana reading
      - spaced_original      : original sentence with morphological tokenization (tokens joined by spaces)
      - spaced_hira_subbed   : tokenized sentence (with "は" overridden to "わ") in Hiragana
      - pos_tags             : space-separated part-of-speech tags
      - spaced_ipa           : IPA conversion of the spaced reading
    """
    # Decide morphological method:
    morph_method = None
    if use_mecab and use_spacy:
        print("Error: Please choose either MeCab or spaCy, not both.")
        sys.exit(1)
    elif use_mecab:
        morph_method = "mecab"
    elif use_spacy:
        morph_method = "spacy"
    else:
        morph_method = None

    # If input is JSON, process as JSON array.
    if json_inplace_update:
        try:
            with open(input_file, "r", encoding="utf-8") as fin:
                data = json.load(fin)
            out_array = []
            for entry in tqdm(data, desc="Processing JSON entries"):
                if "sentence" not in entry:
                    continue
                original_text = entry["sentence"]
                # Compute unspaced IPA:
                hira_unspaced = to_hiragana(original_text)
                ipa_unspaced = hiragana_to_ipa(hira_unspaced)
                out_obj = {
                    "sentence": original_text,
                    "unspaced_ipa": ipa_unspaced,
                    "spaced_original": "",
                    "spaced_hira_subbed": "",
                    "pos_tags": "",
                    "spaced_ipa": ""
                }
                if morph_method is not None:
                    spaced_original, spaced_hira_subbed, _, pos_tags = get_spaced_reading(original_text, morph_method)
                    out_obj["spaced_original"]    = spaced_original if spaced_original is not None else ""
                    out_obj["spaced_hira_subbed"]   = spaced_hira_subbed if spaced_hira_subbed is not None else ""
                    out_obj["pos_tags"]             = pos_tags if pos_tags is not None else ""
                    ipa_spaced = ""
                    if spaced_hira_subbed:
                        ipa_spaced = hiragana_to_ipa(spaced_hira_subbed)
                    out_obj["spaced_ipa"] = ipa_spaced
                out_array.append(out_obj)
            with open(output_file, "w", encoding="utf-8") as fout:
                json.dump(out_array, fout, ensure_ascii=False, indent=4)
        except FileNotFoundError:
            print(f"Error: Input file '{input_file}' not found.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{input_file}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        # Plain text input: each non-blank line is treated as a sentence.
        try:
            with open(input_file, "r", encoding="utf-8") as fin:
                lines = fin.readlines()
            out_array = []
            for line in tqdm(lines, desc="Processing lines"):
                line = line.strip()
                if not line:
                    continue
                original_text = line
                hira_unspaced = to_hiragana(original_text)
                ipa_unspaced = hiragana_to_ipa(hira_unspaced)
                out_obj = {
                    "sentence": original_text,
                    "unspaced_ipa": ipa_unspaced,
                    "spaced_original": "",
                    "spaced_hira_subbed": "",
                    "pos_tags": "",
                    "spaced_ipa": ""
                }
                if morph_method is not None:
                    spaced_original, spaced_hira_subbed, _, pos_tags = get_spaced_reading(original_text, morph_method)
                    out_obj["spaced_original"]    = spaced_original if spaced_original is not None else ""
                    out_obj["spaced_hira_subbed"]   = spaced_hira_subbed if spaced_hira_subbed is not None else ""
                    out_obj["pos_tags"]             = pos_tags if pos_tags is not None else ""
                    ipa_spaced = ""
                    if spaced_hira_subbed:
                        ipa_spaced = hiragana_to_ipa(spaced_hira_subbed)
                    out_obj["spaced_ipa"] = ipa_spaced
                out_array.append(out_obj)
            with open(output_file, "w", encoding="utf-8") as fout:
                json.dump(out_array, fout, ensure_ascii=False, indent=4)
        except FileNotFoundError:
            print(f"Error: Input file '{input_file}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")


# ========== 6) Command-Line Entry Point ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert JP text to IPA with morphological spacing and POS tagging. "
                    "Output is a JSON array with fields: sentence, unspaced_ipa, spaced_original, spaced_hira_subbed, pos_tags, spaced_ipa."
    )
    parser.add_argument("input_file", nargs="?", default="input.txt",
                        help="Path to the input file (JSON array with 'sentence' fields or plain text).")
    parser.add_argument("output_file", nargs="?", default="output.json",
                        help="Path to the output JSON file.")

    parser.add_argument("-j", "--json_inplace_update", action="store_true",
                        help="Treat input file as JSON and update each entry with IPA and POS fields.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use_mecab", action="store_true",
                       help="Use MeCab for morphological tokenization (and forcing 'は' => 'わ').")
    group.add_argument("--use_spacy", action="store_true",
                       help="Use spaCy for morphological tokenization (and forcing 'は' => 'わ').")

    args = parser.parse_args()

    process_japanese_text(
        input_file=args.input_file,
        output_file=args.output_file,
        json_inplace_update=args.json_inplace_update,
        use_mecab=args.use_mecab,
        use_spacy=args.use_spacy
    )

