"""Reference overlap and optional semantic similarity, not factuality or proof."""
import re
from rouge_score import rouge_scorer


def sentence_lines(text):
    # Deterministic, no external sentence-tokenizer download. Declare this
    # segmentation with ROUGE-Lsum results rather than implying another recipe.
    return "\n".join(s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+",text) if s.strip())


def score(prediction,reference):
    scorer=rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL","rougeLsum"],use_stemmer=True)
    result=scorer.score(sentence_lines(reference),sentence_lines(prediction))
    scores={k:100*v.fmeasure for k,v in result.items()}
    words=re.findall(r"\b\w+\b",prediction.lower())
    grams=[tuple(words[i:i+3]) for i in range(max(0,len(words)-2))]
    scores.update(prediction_words=len(words),reference_words=len(re.findall(r"\b\w+\b",reference)),
        repeated_trigram_fraction=(1-len(set(grams))/len(grams)) if grams else 0.0)
    return scores
