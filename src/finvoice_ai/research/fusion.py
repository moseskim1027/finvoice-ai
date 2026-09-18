from finvoice_ai.research.contracts import ResearchCase
from finvoice_ai.research.text_baseline import PredictionSet


def late_fuse(
    text: PredictionSet,
    acoustic: PredictionSet,
    text_weight: float,
) -> PredictionSet:
    if not 0.0 <= text_weight <= 1.0:
        raise ValueError("text weight must be between zero and one")
    if text.case_ids != acoustic.case_ids or text.labels != acoustic.labels:
        raise ValueError("prediction sets must contain aligned cases")
    if text.classes != acoustic.classes:
        raise ValueError("prediction sets must use identical class order")
    probabilities = tuple(
        tuple(
            text_weight * text_value + (1.0 - text_weight) * acoustic_value
            for text_value, acoustic_value in zip(text_row, acoustic_row, strict=True)
        )
        for text_row, acoustic_row in zip(text.probabilities, acoustic.probabilities, strict=True)
    )
    predictions = tuple(
        text.classes[max(range(len(row)), key=row.__getitem__)] for row in probabilities
    )
    return PredictionSet(
        case_ids=text.case_ids,
        labels=text.labels,
        predictions=predictions,
        probabilities=probabilities,
        classes=text.classes,
    )


class ConcatenatedIntentBaseline:
    def __init__(self, *, c: float = 1.0, seed: int = 17) -> None:
        self._c = c
        self._seed = seed
        self._vectorizer = None
        self._scaler = None
        self._model = None

    def fit(
        self,
        cases: list[ResearchCase],
        acoustic_features: list[tuple[float, ...]],
        *,
        transcript_source: str = "asr",
    ) -> "ConcatenatedIntentBaseline":
        from scipy.sparse import csr_matrix, hstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            analyzer="char_wb",
            min_df=1,
            sublinear_tf=True,
        )
        self._scaler = StandardScaler()
        text_matrix = self._vectorizer.fit_transform(_texts(cases, transcript_source))
        acoustic_matrix = self._scaler.fit_transform(acoustic_features)
        features = hstack([text_matrix, csr_matrix(acoustic_matrix)])
        self._model = LogisticRegression(
            C=self._c,
            max_iter=1_000,
            random_state=self._seed,
            class_weight="balanced",
        )
        self._model.fit(features, [case.intent.intent_id for case in cases])
        return self

    def predict(
        self,
        cases: list[ResearchCase],
        acoustic_features: list[tuple[float, ...]],
        *,
        transcript_source: str = "asr",
    ) -> PredictionSet:
        if self._model is None or self._vectorizer is None or self._scaler is None:
            raise RuntimeError("baseline must be fitted before prediction")
        from scipy.sparse import csr_matrix, hstack

        text_matrix = self._vectorizer.transform(_texts(cases, transcript_source))
        acoustic_matrix = self._scaler.transform(acoustic_features)
        features = hstack([text_matrix, csr_matrix(acoustic_matrix)])
        probabilities = self._model.predict_proba(features)
        predictions = self._model.predict(features)
        return PredictionSet(
            case_ids=tuple(case.case_id for case in cases),
            labels=tuple(case.intent.intent_id for case in cases),
            predictions=tuple(str(value) for value in predictions),
            probabilities=tuple(tuple(float(value) for value in row) for row in probabilities),
            classes=tuple(str(value) for value in self._model.classes_),
        )


def _texts(cases: list[ResearchCase], source: str) -> list[str]:
    if source not in {"asr", "reference"}:
        raise ValueError("transcript source must be 'asr' or 'reference'")
    return [getattr(case, f"{source}_transcript") for case in cases]
