(** SQLite persistence and hybrid search through Caqti's blocking API. *)

type t

val attach : model_id:string -> (module Caqti_blocking.CONNECTION) -> (t, string) result
(** Create the Semantic_sqlite tables and FTS5 index on an existing SQLite connection.
    [model_id] must identify the exact model and tokenizer versions; attaching
    with a different ID refuses to mix embeddings. The caller retains ownership
    of the connection. Loads the bundled SQLite Vec1 extension. *)

val open_readonly : model_id:string -> (module Caqti_blocking.CONNECTION) -> (t, string) result
(** Open an existing index without modifying the database. *)

val upsert : t -> id:string -> text:string -> vector:float array -> (unit, string) result
val upsert_many : t -> (string * string * float array) list -> (unit, string) result
val delete : t -> id:string -> (unit, string) result

val search_fts : t -> query:string -> limit:int -> (string list, string) result
(** Search text with SQLite FTS5 query syntax. A query that is empty or made
    only of ASCII whitespace and NUL bytes returns [Ok []]. *)

val literal_fts_query : string -> string
(** Turn user text into an FTS5 query that matches its words literally, for
    [search_fts]. The text is split on ASCII whitespace (space, tab, line
    feed, vertical tab, form feed, carriage return) and on NUL bytes. Words
    without an ASCII letter, ASCII digit, or non-ASCII byte are dropped. Each
    remaining word becomes an FTS5 string (double quotes doubled), so FTS5
    operators ([AND], [OR], [NOT], [NEAR]), prefix [*], column filters, and
    parentheses are searched as text; the strings are joined with [OR].
    Returns [""] when no word remains, for which [search_fts] returns no
    results. [search_hybrid] applies this function to its text. *)

val search_vector :
  ?exact:bool -> ?nprobe:float -> t -> query:float array -> limit:int ->
  ((string * float) list, string) result
(** Search the persistent Vec1 index. After [rebuild_ann], the default is
    approximate IVF search. [nprobe] controls the fraction of buckets searched
    (default 0.05; values >= 1 select an absolute bucket count). [exact=true]
    scans all stored vectors and returns exact cosine scores. *)

val rebuild_ann : ?threads:int -> t -> nbucket:int -> training_limit:int -> (unit, string) result
(** Train a cosine IVF index using up to [training_limit] sampled vectors and
    rebuild it in SQLite. Inserts, updates, and deletes remain incremental. *)

val search_hybrid :
  t -> text:string -> vector:float array -> limit:int ->
  ((string * float) list, string) result
(** Combine the FTS5 and vector rankings with reciprocal rank fusion. The text
    is treated as literal search terms, not raw FTS5 query syntax. *)
