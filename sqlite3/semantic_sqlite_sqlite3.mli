(** Persistent semantic index on an existing SQLite connection. *)

type t

val attach : model_id:string -> Sqlite3.db -> (t, string) result
(** Create the independent Semantic_sqlite tables, load the bundled Vec1 extension,
    and import vectors from an existing 0.1 index. The caller owns the
    connection. [model_id] must identify the exact embedding model and
    tokenizer. *)

val open_readonly : model_id:string -> Sqlite3.db -> (t, string) result
(** Open an existing index without changing the database. *)

val upsert : t -> id:string -> text:string -> vector:float array -> (unit, string) result
(* Insert or update a batch in one SQLite transaction. *)
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
    approximate IVF search. [exact=true] scans all vectors. *)

val rebuild_ann : ?threads:int -> t -> nbucket:int -> training_limit:int -> (unit, string) result
(** Train a cosine IVF index from a sampled subset and rebuild it in SQLite.
    Subsequent writes update the index incrementally. *)

val search_hybrid :
  t -> text:string -> vector:float array -> limit:int ->
  ((string * float) list, string) result
