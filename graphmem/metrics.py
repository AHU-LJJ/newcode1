"""HotpotQA answer/support/joint metrics, aligned with the official v1 evaluator.

Reference: https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
"""
from collections import Counter
import re
import string


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def harmonic(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def score(answer, evidence, gold_answer, gold_evidence):
    predicted, gold = normalize(answer), normalize(gold_answer)
    overlap = sum((Counter(predicted.split()) & Counter(gold.split())).values())
    if predicted != gold and ({predicted, gold} & {"yes", "no", "noanswer"}):
        overlap = 0
    precision = overlap / max(1, len(predicted.split()))
    recall = overlap / max(1, len(gold.split()))
    actual, expected = set(map(tuple, evidence)), set(map(tuple, gold_evidence))
    matches = len(actual & expected)
    sp_precision = matches / len(actual) if actual else 0.0
    sp_recall = matches / len(expected) if expected else 0.0
    jp, jr = precision * sp_precision, recall * sp_recall
    return dict(em=float(predicted == gold), f1=harmonic(precision, recall),
                prec=precision, recall=recall, sp_em=float(actual == expected),
                sp_f1=harmonic(sp_precision, sp_recall), sp_prec=sp_precision, sp_recall=sp_recall,
                joint_em=float(predicted == gold and actual == expected),
                joint_f1=harmonic(jp, jr), joint_prec=jp, joint_recall=jr)


def locomo_normalize(text):
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the|and)\b", " ", text).split())


def locomo_f1(prediction, answer):
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    left = [stemmer.stem(t) for t in locomo_normalize(prediction).split()]
    right = [stemmer.stem(t) for t in locomo_normalize(answer).split()]
    matched = sum((Counter(left) & Counter(right)).values())
    return harmonic(matched / max(1, len(left)), matched / max(1, len(right)))


def locomo_bleu1(prediction, answer, category):
    """Sentence BLEU-1, then macro-average over questions (CAPS protocol v1).

    Use LoCoMo normalization without stemming, clipped unigram precision and
    brevity penalty. Category 3 uses the same first reference segment as F1;
    multi-hop lists are evaluated as whole answers, not per-item maxima.
    This explicitly defined protocol is not every paper's BLEU implementation.
    """
    from nltk.translate.bleu_score import sentence_bleu
    if int(category) == 3:
        answer = answer.split(";")[0].strip()
    reference = locomo_normalize(answer).split()
    candidate = locomo_normalize(prediction).split()
    if not reference or not candidate:
        return 0.0
    return float(sentence_bleu([reference], candidate, weights=(1.0,)))


def score_locomo(prediction, example):
    """Official category-aware answer score; citation scores are auxiliary, not official."""
    answer = example.answer.split(";")[0].strip() if example.kind == "3" else example.answer
    text = prediction["answer"]
    category = int(example.kind)
    if category == 1:
        guesses, references = text.split(","), answer.split(",")
        f1 = sum(max(locomo_f1(p, g) for p in guesses) for g in references) / len(references)
    elif category == 5:
        f1 = float("no information available" in text.lower() or "not mentioned" in text.lower())
    else:
        f1 = locomo_f1(text, answer)
    result = dict(f1=f1, em=float(set(locomo_normalize(text).split()) == set(locomo_normalize(answer).split())))
    if not example.unresolved_evidence and example.supporting_facts and category != 5:
        aux = score(text, prediction["supporting_facts"], answer, example.supporting_facts)
        result.update({key: value for key, value in aux.items() if key.startswith("sp_")})
    return result
