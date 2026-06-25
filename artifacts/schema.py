"""
Canonical schema for threaded conversation data.

Every connector emits Conversation objects. Every pipeline stage consumes them.
This is the only data structure the toolkit guarantees stability for — Connectors
and Models can change shape, the schema cannot break without a version bump.

Design decisions worth knowing before extending this module:

  * Pydantic v2 with `validate_assignment=True` on the mutable types — every field
    write goes through validation. The runtime cost is worth the safety net for
    students attaching enrichment fields from many different pipeline stages.

  * `Speaker` is frozen; `Utterance` and `Conversation` are not. Pipeline stages
    attach enrichment fields (discourse_act, quality_score, embedding) in place
    on Utterance rather than constructing parallel EnrichedUtterance objects.

  * Speakers are stored once per Conversation in `speakers: dict[str, Speaker]`,
    and each Utterance references them by ID. Denormalizing onto every Utterance
    would waste memory in threads where one author posts many times.

  * IDs are source-prefixed strings: 'reddit:t1_abc123', 'hn:item:8675309'.
    Cheap collision resistance across connectors, no UUID overhead, debuggable.

  * Anything connector-specific that doesn't fit the canonical schema goes in
    the `metadata: dict[str, Any]` escape hatch. The canonical fields stay clean.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"


class Speaker(BaseModel):
    """A unique participant in a single conversation.

    Speaker identity is scoped to one Conversation. The same handle appearing
    in two different conversations is two different Speaker objects.
    Cross-thread speaker matching is a downstream concern, not a schema concern.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Canonical ID, source-prefixed: 'reddit:u/spez'")
    handle: str = Field(description="Display handle as it appears in the source")
    source: str = Field(description="Source platform: 'reddit', 'hackernews', etc.")
    is_deleted: bool = Field(
        default=False,
        description="True when the source marks the account as deleted but utterances remain",
    )


class Utterance(BaseModel):
    """A single comment, post, or message within a conversation.

    Mutable by design. Pipeline stages write enrichment fields (`discourse_act`,
    `quality_score`, `embedding`) in place. `validate_assignment=True` catches
    bad writes as they happen.
    """

    model_config = ConfigDict(validate_assignment=True)

    # --- Identity ---
    id: str = Field(description="Canonical ID, source-prefixed: 'reddit:t1_abc123'")
    source_id: str = Field(description="Original platform identifier, unprefixed")
    conversation_id: str = Field(description="Canonical ID of the parent Conversation")
    parent_id: str | None = Field(
        default=None,
        description="Canonical ID of the immediate parent Utterance; None for the root",
    )
    speaker_id: str = Field(description="References Speaker.id in the parent Conversation")

    # --- Content ---
    text: str = Field(description="Raw text as ingested, including markdown")
    text_clean: str | None = Field(
        default=None,
        description="Cleaned variant, populated by preprocessing; None until then",
    )

    # --- Position in thread ---
    created_at: datetime = Field(description="UTC timestamp of creation")
    depth: int = Field(ge=0, description="Distance from root; 0 for the root utterance itself")

    # --- Source-platform signals ---
    score: int | None = Field(
        default=None,
        description="Upvotes minus downvotes or equivalent platform signal; None if not provided",
    )
    is_deleted: bool = Field(
        default=False,
        description="True when content was removed but thread structure was preserved",
    )

    # --- Connector-specific extras ---
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific fields outside the canonical schema",
    )

    # --- Enrichment slots (populated by pipeline stages) ---
    discourse_act: str | None = Field(
        default=None,
        description="Classified discourse role: 'question', 'answer', 'agreement', etc.",
    )
    quality_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Argument quality score in [0.0, 1.0]; higher is stronger",
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Sentence embedding vector; typically 384 or 768 dimensions",
    )


class Conversation(BaseModel):
    """A complete threaded discussion.

    Exactly one utterance has `parent_id=None` — the root. All other utterances
    have a `parent_id` that resolves to another utterance in the same conversation.
    Connectors are responsible for guaranteeing this invariant; the pipeline assumes it.
    """

    model_config = ConfigDict(validate_assignment=True)

    # --- Identity ---
    id: str = Field(description="Canonical ID, source-prefixed")
    source_id: str = Field(description="Original platform identifier")
    source: str = Field(description="Source platform: 'reddit', 'hackernews', etc.")
    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Schema version this object conforms to",
    )

    # --- Metadata ---
    title: str | None = Field(default=None, description="Thread title if the platform has one")
    url: str | None = Field(default=None, description="Canonical URL where the thread can be viewed")
    created_at: datetime = Field(description="UTC timestamp of root creation")

    # --- Content ---
    utterances: list[Utterance] = Field(description="All utterances, ideally in DFS traversal order")
    speakers: dict[str, Speaker] = Field(description="Map of Speaker.id to Speaker")

    # --- Connector-specific extras ---
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def root(self) -> Utterance:
        """Return the single root utterance.

        Raises:
            ValueError: if zero or more than one utterance has parent_id=None.
        """
        roots = [u for u in self.utterances if u.parent_id is None]
        if len(roots) != 1:
            raise ValueError(
                f"Conversation {self.id} expected exactly one root, found {len(roots)}"
            )
        return roots[0]

    def replies_to(self, utterance_id: str) -> list[Utterance]:
        """Return direct replies to the given utterance, in their existing list order."""
        return [u for u in self.utterances if u.parent_id == utterance_id]

    def speaker_of(self, utterance: Utterance) -> Speaker:
        """Look up the Speaker who authored a given Utterance."""
        return self.speakers[utterance.speaker_id]

    def traverse(self, utterance_id: str | None = None) -> Iterator[Utterance]:
        """Depth-first traversal from the given utterance, defaulting to root.

        Iterative implementation — handles arbitrarily deep threads without
        hitting Python's recursion limit. Reddit threads in the wild can be
        300+ deep on a bad day.
        """
        by_id = {u.id: u for u in self.utterances}
        children: dict[str, list[Utterance]] = {}
        for u in self.utterances:
            if u.parent_id is not None:
                children.setdefault(u.parent_id, []).append(u)

        start = self.root() if utterance_id is None else by_id[utterance_id]
        stack: list[Utterance] = [start]
        while stack:
            current = stack.pop()
            yield current
            # Reverse so DFS visits children in their original list order
            for child in reversed(children.get(current.id, [])):
                stack.append(child)

    def __len__(self) -> int:
        """Number of utterances in this conversation."""
        return len(self.utterances)
