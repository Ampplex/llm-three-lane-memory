"""Strict typed schemas for GSW-style semantic extraction."""

from typing import TypedDict, List, Optional


class EntityRole(TypedDict):
    entity: str
    role: str


class ActionItem(TypedDict):
    actor: str
    verb: str
    object: Optional[str]


class StateItem(TypedDict):
    entity: str
    attribute: str
    value: str


class SemanticExtraction(TypedDict):
    summary: str
    emotion: str
    importance: float
    entities: List[str]
    roles: List[EntityRole]
    actions: List[ActionItem]
    states: List[StateItem]
    location: Optional[str]
    time: Optional[str]
