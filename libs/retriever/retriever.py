import typing as t

from qdrant_client.http import models as qm

from .qdrant_db import search_with_scores


def build_filters(
    must: t.Optional[dict] = None,
    should: t.Optional[dict] = None,
    must_not: t.Optional[dict] = None,
) -> t.Optional[qm.Filter]:
    """
    Упрощённый билдер фильтров Qdrant по равенству полей.

    Аргументы принимают словари {field: value|[values]} и конвертируются в
    Term/Any/Match фильтры с логикой must/should/must_not.
    Возвращает qm.Filter или None (если фильтров нет).
    """

    def make_conditions(mapping: dict | None) -> list[qm.FieldCondition]:
        if not mapping:
            return []
        conditions: list[qm.FieldCondition] = []
        for field, value in mapping.items():
            if isinstance(value, (list, tuple, set)):
                conditions.append(
                    qm.FieldCondition(
                        key=str(field),
                        match=qm.MatchAny(any=list(value)),
                    )
                )
            else:
                conditions.append(
                    qm.FieldCondition(
                        key=str(field),
                        match=qm.MatchValue(value=value),
                    )
                )
        return conditions

    must_conds = make_conditions(must)
    should_conds = make_conditions(should)
    must_not_conds = make_conditions(must_not)

    if not (must_conds or should_conds or must_not_conds):
        return None

    return qm.Filter(must=must_conds or None, should=should_conds or None, must_not=must_not_conds or None)


def retrieve(
    collection: str,
    query: str,
    *,
    top_k: int = 5,
    must: t.Optional[dict] = None,
    should: t.Optional[dict] = None,
    must_not: t.Optional[dict] = None,
    with_scores: bool = False,
) -> list[dict]:
    """
    Выполняет семантический поиск по `collection` с опциональными фильтрами.

    Возвращает список чанков payload (и score, если with_scores=True).
    """
    qf = build_filters(must=must, should=should, must_not=must_not)
    results = search_with_scores(collection, query, limit=top_k, filters=qf)

    if with_scores:
        # Вернём объединённую структуру
        return [{**item["payload"], "score": item["score"]} for item in results]

    return [item["payload"] for item in results]



