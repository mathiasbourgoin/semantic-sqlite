type runtime
external runtime_open : string -> int -> runtime = "semantic_sqlite_ort_open"
external runtime_run : runtime -> int array -> float array array = "semantic_sqlite_ort_run"

type t = { runtime : runtime; tokenizer : Tokenizer.t }

let max_threads = 1024

let open_model_with ?threads ~model ~tokenizer () =
  let threads =
    match threads with
    | None -> 0
    | Some count when count >= 1 && count <= max_threads -> count
    | Some _ ->
        invalid_arg "Semantic_sqlite.open_model_with: threads must be between 1 and 1024" in
  let tokenizer = Tokenizer.load tokenizer in
  let runtime = runtime_open model threads in
  { runtime; tokenizer }

let open_model ~model ~tokenizer = open_model_with ~model ~tokenizer ()

let tokenize t text =
  if String.length text > 16_384 then
    invalid_arg "Semantic_sqlite.tokenize: input exceeds 16 KiB; split into passages";
  let body = Tokenizer.encode t.tokenizer text in
  let length = min (Array.length body) 510 in
  Array.init (length + 2) (function
    | 0 -> 0
    | i when i = length + 1 -> 2
    | i -> body.(i - 1))

let normalize vector =
  let sum = Array.fold_left (fun acc x -> acc +. (x *. x)) 0. vector in
  if sum <= 0. || not (Float.is_finite sum) then failwith "Semantic_sqlite: invalid embedding norm";
  let norm = sqrt sum in
  Array.map (fun x -> x /. norm) vector

let embed t text =
  let ids = tokenize t text in
  let rows = runtime_run t.runtime ids in
  if Array.length rows <> Array.length ids || Array.length rows = 0 then
    failwith "Semantic_sqlite: unexpected model output shape";
  let dims = Array.length rows.(0) in
  if dims = 0 then failwith "Semantic_sqlite: empty embedding";
  let pooled = Array.make dims 0. in
  Array.iter (fun row ->
    if Array.length row <> dims then failwith "Semantic_sqlite: ragged model output";
    for i = 0 to dims - 1 do pooled.(i) <- pooled.(i) +. row.(i) done) rows;
  let length = float_of_int (Array.length rows) in
  normalize (Array.map (fun x -> x /. length) pooled)

let embed_query t text = embed t ("query: " ^ text)
let embed_passage t text = embed t ("passage: " ^ text)

let cosine a b =
  if Array.length a = 0 || Array.length a <> Array.length b then
    invalid_arg "Semantic_sqlite.cosine: vector dimensions differ or are empty";
  let dot = ref 0. and an = ref 0. and bn = ref 0. in
  for i = 0 to Array.length a - 1 do
    if not (Float.is_finite a.(i) && Float.is_finite b.(i)) then
      invalid_arg "Semantic_sqlite.cosine: nonfinite vector";
    dot := !dot +. (a.(i) *. b.(i));
    an := !an +. (a.(i) *. a.(i));
    bn := !bn +. (b.(i) *. b.(i))
  done;
  if !an <= 0. || !bn <= 0. then invalid_arg "Semantic_sqlite.cosine: zero vector";
  !dot /. sqrt (!an *. !bn)

let rank ~query items =
  items
  |> List.map (fun (id, vector) -> (id, cosine query vector))
  |> List.sort (fun (_, a) (_, b) -> Float.compare b a)

let reciprocal_rank_fusion ?(k = 60) lists =
  if k <= 0 then invalid_arg "Semantic_sqlite.reciprocal_rank_fusion: k must be positive";
  let scores = Hashtbl.create 32 in
  List.iter (fun ids ->
    let seen = Hashtbl.create (List.length ids) in
    List.iteri (fun index id ->
      if not (Hashtbl.mem seen id) then (
        Hashtbl.add seen id ();
        let old = Option.value (Hashtbl.find_opt scores id) ~default:0. in
        Hashtbl.replace scores id (old +. (1. /. float_of_int (k + index + 1))))) ids) lists;
  Hashtbl.fold (fun id score acc -> (id, score) :: acc) scores []
  |> List.sort (fun (id_a, a) (id_b, b) ->
       let cmp = Float.compare b a in
       if cmp = 0 then String.compare id_a id_b else cmp)

let encode_f32le vector =
  let bytes = Bytes.create (4 * Array.length vector) in
  Array.iteri (fun index value ->
    let bits = Int32.bits_of_float value in
    for offset = 0 to 3 do
      Bytes.set bytes ((4 * index) + offset)
        (Char.chr (Int32.to_int (Int32.logand 0xffl
          (Int32.shift_right_logical bits (8 * offset)))))
    done) vector;
  Bytes.unsafe_to_string bytes

let decode_f32le blob =
  if String.length blob mod 4 <> 0 then
    invalid_arg "Semantic_sqlite.decode_f32le: invalid blob length";
  Array.init (String.length blob / 4) (fun index ->
    let bits = ref 0l in
    for offset = 0 to 3 do
      let byte = Char.code blob.[(4 * index) + offset] in
      bits := Int32.logor !bits (Int32.shift_left (Int32.of_int byte) (8 * offset))
    done;
    Int32.float_of_bits !bits)
