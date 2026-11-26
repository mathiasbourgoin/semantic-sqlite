(** Local ONNX embeddings for multilingual E5 models. *)

type t

val open_model : model:string -> tokenizer:string -> t
(** Load an ONNX model and a Hugging Face [tokenizer.json] for multilingual E5.
    Raises [Failure] on errors. *)

val open_model_with :
  ?threads:int -> model:string -> tokenizer:string -> unit -> t
(** Like [open_model], with ONNX Runtime options. [threads] sets the
    session's intra-op thread count, the threads one inference may use;
    [threads=1] runs each inference on the calling thread. Without
    [threads], ONNX Runtime chooses its default (a pool sized to the
    machine's cores, whose threads spin while waiting). Raises
    [Invalid_argument] unless [1 <= threads <= 1024]; ONNX Runtime was seen
    to hang or crash with much larger pools, and counts above the machine's
    cores bring no speed-up. *)

val tokenize : t -> string -> int array
(** Return model input IDs including the E5 [<s>] and [</s>] tokens.
    Inputs above 16 KiB are rejected; split files into passages first. *)

val embed_query : t -> string -> float array
val embed_passage : t -> string -> float array
(** Embed with the required E5 prefix, mean pooling, and L2 normalization. *)

val cosine : float array -> float array -> float
(** Cosine similarity. Raises [Invalid_argument] on mismatched or zero vectors. *)

val rank : query:float array -> (string * float array) list -> (string * float) list
(** Sort by descending cosine similarity. *)

val reciprocal_rank_fusion : ?k:int -> string list list -> (string * float) list
(** Fuse ranked identifier lists, using [k=60] by default. *)

val encode_f32le : float array -> string
val decode_f32le : string -> float array
(** Portable little-endian float32 blobs for a SQLite BLOB column. [decode_f32le]
    raises [Invalid_argument] when the blob length is not a multiple of four. *)
