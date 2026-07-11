import sys

sys.path.insert(0, ".")

from app.pipeline.shared.retriever import retrieve, route_query

# Replace with a real video_id that has chunks indexed in Qdrant
VIDEO_ID = "9fc88a69-7eab-41c9-8943-8c3148aca8e6"

# Test router
print(route_query("How Covid-19 affected the council?"))  # → text
print(route_query("What is shown on the slide of line chart?"))  # → visual
print(
    route_query("Can you explain the statistics chart of the Weekly card transaction section?")
)  # → visual

# Test retrieval
results = retrieve(VIDEO_ID, "What are the main topics discussed?", top_k=5)
for r in results:
    m, s = divmod(int(r["start"]), 60)
    print(f"[{m:02d}:{s:02d}] score={r['score']:.4f} speaker={r['speaker']} | {r['text']}")
