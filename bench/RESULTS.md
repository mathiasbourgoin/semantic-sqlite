# Million-vector SQLite benchmark

Oonnxx 0.2.0's Vec1-backed IVF search, now part of Semantic SQLite, was measured against its exact linear
search on two reproducible synthetic corpora, each with 1,000,000 normalized
vectors of 384 float32 components. The results show the recall and latency
trade-off for these corpora; they do not predict performance on a different
embedding distribution.

| Corpus | `nprobe` | Recall@10 | Approximate median | Exact median |
|---|---:|---:|---:|---:|
| Clustered | 0.05 | 1.00 | 61.9 ms | 4,310.3 ms |
| Uniform | 0.05 | 0.17 | 26.3 ms | 4,248.9 ms |
| Uniform | 0.50 | 0.79 | 271.7 ms | 4,221.6 ms |

The clustered corpus uses 256 random unit centroids and adds independent
uniform noise to each component before normalization. The uniform corpus uses
independent random components. Each query perturbs a vector drawn from the
corpus. Recall@10 is the mean overlap with exact top-10 results across ten
queries. Search order alternates between approximate-first and exact-first;
the medians are warm-cache measurements. The exact path decodes each stored
float32 vector and ranks it in OCaml. The approximate path probes Vec1's
persistent IVF index. Both use the same SQLite database and original vectors.
The corpus and query generator has a fixed seed, but the SQL training sample
uses `ORDER BY RANDOM()`; a fresh build can therefore produce a different IVF
model and different recall.

Both databases used 256 buckets, 16,384 sampled training vectors, no product
quantization, and 16 training threads. Initial ingestion took 28.2 s
(clustered) and 29.0 s (uniform); training and index reconstruction took
39.1 s and 82.3 s respectively. Each database occupied 4,148,523,008 bytes.
These are one-run wall-clock figures, not throughput guarantees.

Measurements were made on Linux x86-64 with an AMD Ryzen 9 7950X (16 cores,
32 threads), SQLite 3.54.0, and Vec1 0.8. The benchmark used a scratch
database on an external SSD, with SQLite journaling and synchronous writes
disabled for data generation only. Its source is [`million_vectors.ml`](million_vectors.ml).
Set `SEMANTIC_SQLITE_BENCH_DB` to a **new scratch path**, then run:

```sh
SEMANTIC_SQLITE_VEC1_EXTENSION=_build/default/sqlite3/vec1.so \
SEMANTIC_SQLITE_BENCH_DB=/path/to/scratch.sqlite \
SEMANTIC_SQLITE_BENCH_N=1000000 SEMANTIC_SQLITE_BENCH_DIM=384 \
dune exec bench/million_vectors.exe
```

Set `SEMANTIC_SQLITE_BENCH_CORPUS=uniform` for the second corpus. To vary only the
query probe fraction against an already built database, set
`SEMANTIC_SQLITE_BENCH_REUSE=1` and `SEMANTIC_SQLITE_BENCH_NPROBE=0.5`. The generator's fixed
seed recreates the same query vectors in reuse mode.
