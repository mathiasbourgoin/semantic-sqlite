# Semantic SQLite

Semantic SQLite helps OCaml applications build **semantic indexes in SQLite**: store text passages with their embeddings, then retrieve passages by meaning, by keyword, or with both rankings combined. A query in English can retrieve a relevant passage in French. The library computes multilingual E5 embeddings locally through ONNX Runtime and works with an existing `Sqlite3.db` or Caqti SQLite connection. Its bundled SQLite Vec1 extension provides a persistent approximate nearest-neighbor index for larger collections. You can also supply vectors computed elsewhere through `upsert` or `upsert_many`.

The built-in inference path supports the tested multilingual E5 model contract: `input_ids`, `attention_mask`, and `token_type_ids` inputs and a float32 `last_hidden_state` output. It is not a general ONNX model runner. Applications supply passage splitting and stable IDs; this library indexes the resulting passages and searches them.

## Install and supply a model

Semantic SQLite currently supports Linux and builds without ONNX Runtime installed. From a source checkout:

```sh
opam pin add semantic-sqlite .
```

At run time, supply ONNX Runtime 1.23.2's `libonnxruntime.so` on the linker path or set `ONNXRUNTIME_LIBRARY` to its path. Also download these matching files from the model author:

- [`model_O4.onnx`](https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model_O4.onnx) — tested SHA-256 `4654c156f3e4171abc9c716cdb771bf9116455d15ac1aab364aeeede0e3205b0`
- [`tokenizer.json`](https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/tokenizer.json) — tested SHA-256 `0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39`

The model and runtime binaries are supplied separately; opam does not download them. Check the asset hashes because the URLs follow a movable branch. `Semantic_sqlite.open_model` raises an exception if it cannot load the model, tokenizer, or runtime.

By default ONNX Runtime sizes its thread pool to the machine and its threads spin while waiting, so a single embedding call can occupy every core. Use `Semantic_sqlite.open_model_with ~threads:1 ~model ~tokenizer ()` to run each inference on the calling thread, or a count up to 1024 to bound the pool; it sets the session's intra-op thread count. Combine `~threads:1` with `Semantic_sqlite_eio.embed_passages` to choose the parallelism with `max_fibers` instead. Semantic SQLite does not expose ONNX Runtime's `session.intra_op.allow_spinning` session option, which keeps the default pool but stops its idle spinning; on a loaded 32-thread machine with ONNX Runtime 1.23.2, setting it to `0` through the C API's `AddSessionConfigEntry` reduced the default pool's CPU use from about 24 cores to about 7.

## Index passages and search

Add `semantic-sqlite`, `semantic-sqlite.sqlite3`, and `sqlite3` to the application's Dune dependencies. This example opens a writable SQLite database and assumes the model files are at the paths shown:

```ocaml
let engine =
  Semantic_sqlite.open_model ~model:"model_O4.onnx" ~tokenizer:"tokenizer.json"

let matches =
  let db = Sqlite3.db_open "passages.sqlite" in
  let index =
    Semantic_sqlite_sqlite3.attach
      ~model_id:"e5-small-o4:<model-sha256>:<tokenizer-sha256>" db
    |> Result.get_ok in
  let passages = [ "passage-1", "un chat noir"; "passage-2", "database schema" ] in
  let rows =
    List.map (fun (id, text) -> id, text, Semantic_sqlite.embed_passage engine text)
      passages in
  Semantic_sqlite_sqlite3.upsert_many index rows |> Result.get_ok;
  let results =
    Semantic_sqlite_sqlite3.search_hybrid index ~text:"black cat"
      ~vector:(Semantic_sqlite.embed_query engine "black cat") ~limit:10
    |> Result.get_ok in
  ignore (Sqlite3.db_close db);
  results
(* (passage_id * fused_score) list *)
```

Replace the placeholders in `model_id` with the hashes of the exact model and tokenizer files used to create the index. `attach` loads the bundled Vec1 extension and creates the index tables in the existing database; it also imports vectors from a previous Oonnxx index. The tables retain their `oonnxx_` prefix so existing databases remain readable after the package rename. It records the model ID and vector dimension and refuses incompatible embeddings. `upsert_many` writes a batch in one transaction, updating both the FTS5 text index and the Vec1 vector index. Single-row `upsert` and `delete` keep both indexes current. The caller owns and closes the SQLite connection.

The package installs `vec1.so` with the library and loads it for each attached SQLite connection. If the extension has been relocated, set `SEMANTIC_SQLITE_VEC1_EXTENSION` to its file path. The old `OONNXX_VEC1_EXTENSION` variable remains a fallback for existing deployments. SQLite 3.38.0 or newer is required at build time and run time, with loadable extensions enabled; `attach` and `open_readonly` return an error when Vec1 cannot be loaded.

For an existing `Caqti_blocking.CONNECTION` using the SQLite3 driver, use `Semantic_sqlite_caqti.attach ~model_id conn` instead and add `semantic-sqlite.caqti` to the Dune dependencies. The Caqti facade exposes the same indexing, rebuild, and search functions. It uses the driver's underlying SQLite connection, so keep the blocking connection on the thread that created it.

`search_vector` searches Vec1 by cosine distance; `search_fts` accepts SQLite FTS5 query syntax; `search_hybrid` treats its text as literal search terms and combines keyword and vector rankings with reciprocal rank fusion. To give user input to `search_fts`, convert it with `literal_fts_query`, which quotes each word so FTS5 operators such as `AND`, `NOT`, and `*` are searched as text; it returns an empty string when the input has no searchable word, and `search_fts` returns no results for an empty query. The returned scores from these methods have different meanings and should not be compared across methods. Semantic SQLite's tables are independent of other application tables.

## Build and use the approximate index

Passages can be searched before ANN training. Once at least four times `nbucket` vectors are stored, call `rebuild_ann` while the writable `index` handle is open to train a persistent cosine IVF index from a sample of stored vectors:

```ocaml
Semantic_sqlite_sqlite3.rebuild_ann index ~nbucket:256 ~training_limit:16_384
|> Result.get_ok
```

`nbucket` must be between 2 and 65,536; `training_limit` must be at least four times `nbucket`. The function samples at most `training_limit` vectors, rebuilds the index in SQLite, and can take `~threads` to control training parallelism. Later inserts, updates, and deletes update the trained index incrementally. Rebuilding is an explicit operation, not part of each write.

After a rebuild, `search_vector` defaults to approximate search (`nprobe = 0.05`, a fraction of buckets). Increase `~nprobe` to search more buckets; values of at least 1 select an absolute bucket count. Use `search_vector ~exact:true` when you need an exhaustive scan and exact cosine scores. `search_hybrid` uses the default vector search, so after a rebuild its semantic candidates are approximate. ANN results can omit neighbors returned by exact search; tune `nprobe` against recall and latency on your data.

To query an existing database without changing it, open a read-only `Sqlite3.db` and call `Semantic_sqlite_sqlite3.open_readonly ~model_id db`. It loads Vec1 and checks the stored model ID but does not create or migrate tables; writes and `rebuild_ann` through that handle return errors. `Semantic_sqlite_caqti.open_readonly` provides the same behavior for a Caqti SQLite connection.

For large inputs, split files into passages and give each passage a stable ID. Inputs above 16 KiB are rejected, and tokenization keeps at most 510 body tokens per passage. `Semantic_sqlite_eio.embed_passages` embeds independent passages on Eio domains with a caller-selected concurrency limit and preserves input order; add `semantic-sqlite.eio` to the Dune dependencies.

## Current limits

The embedding path was tested on Linux x86-64 with ONNX Runtime 1.23.2 and the model files above. The tokenizer matches the Hugging Face implementation on the French and English cases in `test/test_semantic_sqlite.ml`. It implements Unicode NFKC and the whitespace behavior exercised by those cases, but does not decode the model's complete precompiled normalization table. Validate other scripts and unusual Unicode before relying on exact token parity.

Exact vector search scans every stored vector; approximate search needs a trained IVF index and trades recall for speed. The [million-vector benchmark](bench/RESULTS.md) reports indexing, training, recall against exact top-10, query latency, and database size for two synthetic corpora. It uses a scratch database with durability disabled, so its ingestion setup is unsuitable for application data. Measure recall, latency, and storage with your own corpus before choosing bucket count and `nprobe`.

The Eio helper provides CPU parallelism across passages; it does not batch inputs into one ONNX tensor. Measure embedding throughput with your workload before choosing passage size and concurrency.

The implementation uses OCaml and a small C binding to ONNX Runtime's C API. Its MIT-licensed C headers are included under `lib/vendor/` so package builds need no system ONNX Runtime headers.

## Renaming from Oonnxx

The source repository and package are now `semantic-sqlite`. In application Dune files, replace `oonnxx`, `oonnxx.sqlite3`, `oonnxx.caqti`, and `oonnxx.eio` with the corresponding `semantic-sqlite` library names. The OCaml modules are `Semantic_sqlite`, `Semantic_sqlite_sqlite3`, `Semantic_sqlite_caqti`, and `Semantic_sqlite_eio`. Existing SQLite indexes keep their `oonnxx_` table names and can be opened with the same `model_id`; no data copy is required.

## License

Semantic SQLite is licensed under the MIT License. See [LICENSE](LICENSE). The vendored ONNX Runtime C headers retain their own [MIT license notice](lib/vendor/ONNXRUNTIME-LICENSE). The bundled [SQLite Vec1 source](sqlite3/vendor/vec1.c) carries its author's public-domain dedication.

## Run the tests

```sh
ONNXRUNTIME_LIBRARY=/path/to/libonnxruntime.so \
SEMANTIC_SQLITE_TEST_MODEL=/path/to/model_O4.onnx \
SEMANTIC_SQLITE_TEST_TOKENIZER=/path/to/tokenizer.json \
dune runtest
```

Without the two test asset variables, the vector math and blob tests run but the model integration test is skipped.
