"""Bootstrap script — populate cluster_history with synthetic scenarios.

Run once:
    python bootstrap.py

Safe to re-run: uses upsert so existing entries are not duplicated.
"""
from __future__ import annotations

from datetime import datetime

from app.agents.embed_agent import _HISTORY_COLLECTION, get_chroma_client, get_embeddings
from app.agents.parser_agent import parse_line
from app.agents.pattern_agent import cluster_events
from app.generator.factory import generate_logs
from app.models.schemas import Scenario

_SCENARIO_META: dict[str, dict] = {
    Scenario.OOM_CRASH: {"scenario": "oom_crash", "root_cause_type": "OOM"},
    Scenario.DB_TIMEOUT: {"scenario": "db_timeout", "root_cause_type": "DBTimeout"},
    Scenario.SILENT_FAIL: {"scenario": "silent_fail", "root_cause_type": "SilentFail"},
}


def bootstrap() -> None:
    client = get_chroma_client()

    # Skip if already bootstrapped
    try:
        existing = client.get_collection(_HISTORY_COLLECTION)
        count = existing.count()
        if count > 0:
            print(f"Already bootstrapped ({count} entries in '{_HISTORY_COLLECTION}'). Nothing to do.")
            return
    except Exception:
        pass

    collection = client.get_or_create_collection(
        _HISTORY_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    now = datetime.utcnow().isoformat()

    for scenario in [Scenario.OOM_CRASH, Scenario.DB_TIMEOUT, Scenario.SILENT_FAIL]:
        raw_logs = generate_logs(scenario, n_logs=50)
        events = [e for line in raw_logs if (e := parse_line(line)) is not None]
        clusters = cluster_events(events)

        meta_base = _SCENARIO_META[scenario]
        inserted = 0

        for cluster in clusters:
            doc_id = f"bootstrap_{scenario.value}_{cluster.cluster_id}"
            embedding = get_embeddings([cluster.representative])[0]
            collection.upsert(
                documents=[cluster.representative],
                embeddings=[embedding],
                ids=[doc_id],
                metadatas=[{
                    **meta_base,
                    "source": "bootstrap",
                    "created_at": now,
                    "cluster_size": cluster.size,
                }],
            )
            inserted += 1

        print(f"  {scenario.value}: {inserted} clusters inserted")

    print(f"\nBootstrap complete — {collection.count()} total entries in '{_HISTORY_COLLECTION}'.")


if __name__ == "__main__":
    bootstrap()
