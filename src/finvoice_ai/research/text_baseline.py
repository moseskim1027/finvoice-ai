from dataclasses import dataclass

from finvoice_ai.research.contracts import ResearchCase


@dataclass(frozen=True)
class PredictionSet:
    case_ids: tuple[str, ...]
    labels: tuple[str, ...]
    predictions: tuple[str, ...]
    probabilities: tuple[tuple[float, ...], ...]
    classes: tuple[str, ...]


class MajorityIntentBaseline:
    def __init__(self) -> None:
        self._label = ""
        self._classes: tuple[str, ...] = ()

    def fit(self, cases: list[ResearchCase]) -> "MajorityIntentBaseline":
        counts: dict[str, int] = {}
        for case in cases:
            counts[case.intent.intent_id] = counts.get(case.intent.intent_id, 0) + 1
        if not counts:
            raise ValueError("training cases must not be empty")
        self._classes = tuple(sorted(counts))
        self._label = min(counts, key=lambda label: (-counts[label], label))
        return self

    def predict(self, cases: list[ResearchCase]) -> PredictionSet:
        if not self._label:
            raise RuntimeError("baseline must be fitted before prediction")
        probability = tuple(float(label == self._label) for label in self._classes)
        return PredictionSet(
            case_ids=tuple(case.case_id for case in cases),
            labels=tuple(case.intent.intent_id for case in cases),
            predictions=(self._label,) * len(cases),
            probabilities=(probability,) * len(cases),
            classes=self._classes,
        )


class TfidfIntentBaseline:
    def __init__(self, *, c: float = 1.0, seed: int = 17) -> None:
        self._c = c
        self._seed = seed
        self._pipeline = None

    def fit(
        self,
        cases: list[ResearchCase],
        *,
        transcript_source: str = "asr",
    ) -> "TfidfIntentBaseline":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline

        word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        character = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        classifier = LogisticRegression(
            C=self._c,
            max_iter=1_000,
            random_state=self._seed,
            class_weight="balanced",
        )
        self._pipeline = Pipeline(
            [
                ("features", FeatureUnion([("word", word), ("character", character)])),
                ("model", classifier),
            ]
        )
        self._pipeline.fit(_texts(cases, transcript_source), _labels(cases))
        return self

    def predict(
        self,
        cases: list[ResearchCase],
        *,
        transcript_source: str = "asr",
    ) -> PredictionSet:
        if self._pipeline is None:
            raise RuntimeError("baseline must be fitted before prediction")
        probabilities = self._pipeline.predict_proba(_texts(cases, transcript_source))
        predictions = self._pipeline.predict(_texts(cases, transcript_source))
        classes = tuple(str(value) for value in self._pipeline.classes_)
        return PredictionSet(
            case_ids=tuple(case.case_id for case in cases),
            labels=tuple(_labels(cases)),
            predictions=tuple(str(value) for value in predictions),
            probabilities=tuple(tuple(float(value) for value in row) for row in probabilities),
            classes=classes,
        )


def _texts(cases: list[ResearchCase], source: str) -> list[str]:
    if source not in {"asr", "reference"}:
        raise ValueError("transcript source must be 'asr' or 'reference'")
    return [getattr(case, f"{source}_transcript") for case in cases]


def _labels(cases: list[ResearchCase]) -> list[str]:
    return [case.intent.intent_id for case in cases]
