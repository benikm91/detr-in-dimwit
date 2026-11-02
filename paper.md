# The paper: framing and plans

A working document: the directions the paper can take, what the results so far support, and what is
still missing. Results live in `vitruvion-results.md` (sketches) and, once complete, a floor plan
counterpart; this file only points at them.

## Where it started, and why that changed

The plan was a paper about **d2po**: d2s, the remaining-node set predictor, writing nodes class by
class. It was our best method on the sketches until **d2tok**, a plain language model over
coordinate tokens trained as the naive baseline, overtook it (61.6 % of Vitruvion test sketches
exactly right, against 58.1 % for d2po). A paper selling d2po as the way to predict sets would be
contradicted by its own baseline.

## The framing: a controlled comparison of how to predict a set from an image

Not "here is our new set predictor" but a grounded study: one codebase, one encoder, one training
budget and decoding scheme, and existing and new ways of writing a set down compared head to head on
document data where the target is known exactly.

**Working title ideas**
- *Sets, sequences or tokens? A controlled comparison of image-conditioned set prediction*
- *What should a model write down? Predicting structured records from document images*

**Research questions**
1. **Set or sequence:** does treating the output as a set (DETR, d2s) beat writing it as a sequence
   (d2seq, d2tok)?
2. **Order:** when a model writes one element at a time, which order should it learn — a fresh
   random one, a fixed one (class, position, the corpus's own), or none (d2s's remaining-node loss)?
3. **Factorisation:** one prediction per element (d2seq, d2s) or one per coordinate (d2tok)?
4. **Scale:** what happens as records grow from at most 16 elements (sketches) to 73 (floor plans)?
5. **Relations:** how far do the approaches carry from sets to graphs (floor plans with `joined` and
   `in`)?

**Contributions as they stand**
- **Two benchmarks, each with data, code and baselines:**
  - *Sketches:* the Vitruvion split of SketchGraphs, which PICASSO and DAVINCI are measured on, is
    no longer publicly available. We recreated it and published it on Hugging Face
    ([benikm91/sketch-graph-vitruvion](https://huggingface.co/datasets/benikm91/sketch-graph-vitruvion)),
    with every model of this paper as a baseline on it.
  - *Floor plans:* a new synthetic dataset of exact records with relationships, each drawing drawn
    from its record alone. The generator code (*link to add*) and the dataset
    ([benikm91/synthetic-floorplans-for-transcription](https://huggingface.co/datasets/benikm91/synthetic-floorplans-for-transcription))
    are public, with the same baselines.
- The comparison itself: seven model families under one budget, with seeds, on both benchmarks.
- Findings, see below; and a cost analysis (training time, sequential decoding steps).

## What the results support so far

| claim | evidence | status |
|---|---|---|
| A naive token model is a strong baseline for image-to-set | d2tok random is the best of all on Vitruvion (61.6 % sketches exactly right, PF1 92.4, above DAVINCI's published 91.7) | **supported** (3 seeds) |
| A random order beats fixed orders for token models | d2tok: random 61.6, class 61.1, position 56.2, kept 53.7; consistent with Pix2Seq's object order ablation | supported for kept (3 seeds); class and position one seed each |
| One prediction per element fails where the next element is ambiguous | d2seq random 0.1 % against d2tok random 61.6 % (same order, same budget); d2seq kept 23.8 against d2tok kept 53.7 | **supported** (3 seeds); compute per sketch differs (113 decoder positions against 17), see `vitruvion-results.md` |
| Set predictors perform alike on sketches | d2s free 52.9, d2s class 57.7, DETR 54.1, DETR with class buckets 52.6 (validation) | supported; buckets one seed |
| Which approach wins depends on the record's structure | floor plans, validation at 100k steps: d2s walls first 91.6, DETR 76.5, d2tok 72.7, d2s free 52.9 | **pending**: one seed, validation only; 200k runs and test scores outstanding |
| An order of the set matters most where many elements compete | d2s: class order +5 points on sketches, +39 on floor plans | pending, as above |
| d2s does not simply need more training | d2s free floor plans: 52.9 → 67.4 from 100k to 200k steps, still behind the others at 100k | pending: the others at 200k |
| Token models fail at the end, set models in the middle | floor plans: d2tok's errors are almost all one surplus node after a complete plan; d2s's are omitted nodes, worst on doors | one seed each |

The open question for the story is the floor plans. If the 200k runs and seeds keep d2s walls-first
clearly ahead, the paper has two contrasting datasets and a clear lesson: **an ordered set predictor
for many structured elements, tokens for few loosely structured ones.** If DETR or d2tok catch up,
the lesson is closer to "it barely matters which, as long as the order is right".

## What is still needed

- Floor plans at 200k for DETR, d2tok and d2s walls first (running), and their test scores.
- Seeds on the floor plans, at least for the leading two or three.
- The bucketed DETR's test score; d2tok class and position with further seeds.
- Relations (RQ5): DETR plus a second decoder for the edge set (see below), d2tok writing relation
  words, and the rule baseline from geometry.
- A fairness section: equal budget and recipe for all, no per-model tuning beyond what is stated,
  the greedy matching of the metric, decoding latency next to accuracy.

**Reviewer risks to answer in advance**
- "The baselines were not tuned": the recipe is shared and stated; show convergence curves.
- "The differences are small": report seeds and say which differences are not results.
- "Synthetic data": the sketches are real (SketchGraphs); the floor plans are synthetic by design,
  since exact records are what makes whole-record exactness measurable.

## An idea that may carry a section: a graph as two set predictions

A graph is a set of nodes and a set of edges. A DETR decoder writes the nodes; a second DETR-style
decoder with learned edge queries cross-attends the image and the node embeddings and writes the
edges, each naming its ends among the nodes. The closest published design is SGTR (CVPR 2022),
which reaches its entities only indirectly (indicator queries assembled afterwards). With class
buckets on both sides (`in` edges point from opening queries to wall queries, `joined` from walls to
walls), type constraints hold by construction; we found that nowhere.

## Further datasets: brainstorm

Requirements: the target is a set (or graph) conditioned on an image, labels exact or nearly so, a
known benchmark, ideally in document analysis for ICDAR. Sizes and details below are from memory
and need checking before any decision.

| dataset | what is predicted | why it fits | caveats |
|---|---|---|---|
| **Chemical structures** (molecule images rendered from PubChem/USPTO; benchmarks such as USPTO, CLEF, JPO) | atoms (a set) and bonds (an edge set) | exact labels from the molecule; renderable at any scale, as our floor plans; a graph, so it also tests RQ5; established sequence (SMILES/InChI) and graph (atoms then bonds) approaches to compare | real-world test sets are scanned and noisy; chemistry is a niche at ICDAR |
| **Table structure** (PubTables-1M) | rows, columns and cells as boxes, or as an HTML sequence | DETR is the published baseline (Table Transformer), and image-to-HTML sequence models are the rival: the set-versus-sequence question exactly; large and fairly clean | boxes, not exact geometry; a cell grid is very regular |
| **Mathematical expressions** (CROHME) | symbols and their spatial relations (a symbol layout graph), or a LaTeX sequence | the ICDAR competition; set/graph against sequence is a live debate there | small; LaTeX models dominate; handwriting adds noise |
| **Music notation** (DeepScoresV2; MUSCIMA++ for the notation graph) | notation symbols, and in MUSCIMA++ their relations | rendered scores give exact labels and very many small objects per page, beyond our 73 | DeepScores is detection with boxes; MUSCIMA++ is small |
| **Document layout** (DocLayNet) | regions as boxes with classes | human-annotated, cleaner than PubLayNet; well known at ICDAR | boxes only; layout is loosely defined, inter-annotator disagreement |
| **CLEVR object properties** | a set of objects with position, shape, colour, size, material | the canonical set-prediction benchmark of machine learning (DSPN, Slot Attention); synthetic, exact labels | not a document; fits an ML venue better than ICDAR |
| **Object detection** (COCO, VOC) | boxes with classes | the best-known image-to-set task | noisy labels, boxes only, as discussed; not a fit |

For ICDAR, the strongest additions look like **chemical structures** (exact, synthetic, a graph,
and it would extend the floor plan story) or **table structure** (the clearest set-against-sequence
debate in document analysis, with DETR as the incumbent). CLEVR is the one to add if the paper
moves towards a machine learning venue.
