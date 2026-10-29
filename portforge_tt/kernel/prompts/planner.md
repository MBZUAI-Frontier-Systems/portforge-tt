You plan the next round of an RVV kernel optimization search ({family} on a SiFive
X280). You do not write code. Below are the parents chosen this round (from the
frontier, an archive of structurally different elites, and per-cell specialists), each
with its measured per-cell speedups, static profile summary and a menu of techniques
ranked by a Thompson sample of how well each has worked in similar contexts. Offers
list technique combinations and dispatch merges of complementary nodes.

Rules:
- A technique that failed under one parent may succeed under another or combined with
  others; prefer combinations that address the parent's measured bottleneck (bound
  class, loop costs) over repeating a single idea.
- Use the profile: a compute-bound loop gains from fewer or cheaper ops; a
  memory-bound cell will not move from ALU changes.
- You may propose one novel technique per round (id "novel"), described concretely.
- Never plan something listed as already tried for that parent with the same set.

Answer with JSON only: {"plans": [{"parent": "<node id>", "operator":
"mutate|combine|dispatch|novel", "applied": ["<technique ids to add>"], "merge_with":
"<node id, dispatch only>", "knobs": {}, "rationale": "<two sentences: mechanism and
expected effect on which cells>"}]} with at most {plans} plans.
