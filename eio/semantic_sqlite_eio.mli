(** Domain-parallel embedding helpers. *)

val embed_passages :
  domain_mgr:_ Eio.Domain_manager.t ->
  max_fibers:int -> Semantic_sqlite.t -> string list -> float array list
(** Embed independent passages on at most [max_fibers] Eio domains, preserving
    input order. The ONNX Runtime session is shared and its [Run] call releases
    the OCaml runtime lock. *)
