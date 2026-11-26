type t = {
  vocab : (string, int * float) Hashtbl.t;
  max_piece_bytes : int;
}

let load filename =
  let open Yojson.Basic.Util in
  let json = Yojson.Basic.from_file filename in
  let model = member "model" json in
  if to_string (member "type" model) <> "Unigram" then
    failwith "Semantic_sqlite: expected a Unigram tokenizer";
  let pretokenizer = member "pre_tokenizer" json in
  if to_string (member "type" pretokenizer) <> "Metaspace"
     || to_string (member "replacement" pretokenizer) <> "▁"
     || to_bool (member "add_prefix_space" pretokenizer) <> true then
    failwith "Semantic_sqlite: unsupported tokenizer pre-processing";
  if to_int (member "unk_id" model) <> 3 then
    failwith "Semantic_sqlite: unexpected E5 unknown token ID";
  let entries = to_list (member "vocab" model) in
  let piece = function
    | `List [ `String text; _ ] -> text
    | _ -> failwith "Semantic_sqlite: invalid tokenizer vocabulary" in
  (match entries with
   | a :: b :: c :: d :: _
     when List.map piece [ a; b; c; d ] = [ "<s>"; "<pad>"; "</s>"; "<unk>" ] -> ()
   | _ -> failwith "Semantic_sqlite: unexpected E5 special tokens");
  let vocab = Hashtbl.create 300_001 in
  let max_piece_bytes = ref 0 in
  entries
  |> List.iteri (fun id entry ->
       match to_list entry with
       | [ `String piece; `Float score ] when id >= 4 ->
           Hashtbl.replace vocab piece (id, score);
           max_piece_bytes := max !max_piece_bytes (String.length piece)
       | [ `String piece; `Int score ] when id >= 4 ->
           Hashtbl.replace vocab piece (id, float_of_int score);
           max_piece_bytes := max !max_piece_bytes (String.length piece)
       | [ `String _; _ ] -> ()
       | _ -> failwith "Semantic_sqlite: invalid tokenizer vocabulary");
  if Hashtbl.length vocab < 1000 then failwith "Semantic_sqlite: tokenizer vocabulary too small";
  { vocab; max_piece_bytes = !max_piece_bytes }

let normalize_nfkc text =
  let decoder = Uutf.decoder ~encoding:`UTF_8 (`String text) in
  let normalizer = Uunf.create `NFKC in
  let output = Buffer.create (String.length text) in
  let rec feed input =
    match Uunf.add normalizer input with
    | `Uchar scalar ->
        Uutf.Buffer.add_utf_8 output scalar;
        feed `Await
    | `Await -> ()
    | `End -> ()
  in
  let rec decode () =
    match Uutf.decode decoder with
    | `Uchar scalar ->
        (match Uchar.to_int scalar with
         | 0x0007 -> ()
         | 0x200b | 0x2028 -> feed (`Uchar (Uchar.of_int 0x20))
         | _ -> feed (`Uchar scalar));
        decode ()
    | `End -> feed `End
    | `Malformed _ -> invalid_arg "Semantic_sqlite: malformed UTF-8 input"
    | `Await -> assert false
  in
  decode ();
  Buffer.contents output

let prepare text =
  let text = normalize_nfkc text in
  let output = Buffer.create (String.length text + 3) in
  Buffer.add_string output "▁";
  let previous_space = ref false in
  String.iter (fun character ->
    if character = ' ' || character = '\t' || character = '\r' || character = '\n' then (
      if not !previous_space then Buffer.add_string output "▁";
      previous_space := true)
    else (
      Buffer.add_char output character;
      previous_space := false)) text;
  Buffer.contents output

let encode t text =
  let text = prepare text in
  let size = String.length text in
  let scores = Array.make (size + 1) neg_infinity in
  let previous = Array.make (size + 1) (-1, 3) in
  scores.(0) <- 0.;
  for start = 0 to size - 1 do
    if Float.is_finite scores.(start) then (
      let max_length = min t.max_piece_bytes (size - start) in
      let matched = ref false in
      for length = 1 to max_length do
        match Hashtbl.find_opt t.vocab (String.sub text start length) with
        | None -> ()
        | Some (id, score) ->
            matched := true;
            let stop = start + length in
            let candidate = scores.(start) +. score in
            if candidate > scores.(stop) then (
              scores.(stop) <- candidate;
              previous.(stop) <- (start, id))
      done;
      if not !matched then (
        let byte = Char.code text.[start] in
        let width =
          if byte land 0xf8 = 0xf0 then 4 else
          if byte land 0xf0 = 0xe0 then 3 else
          if byte land 0xe0 = 0xc0 then 2 else 1 in
        let stop = min size (start + width) in
        let candidate = scores.(start) -. 100. in
        if candidate > scores.(stop) then (
          scores.(stop) <- candidate;
          previous.(stop) <- (start, 3)))
    )
  done;
  if not (Float.is_finite scores.(size)) then failwith "Semantic_sqlite: tokenization failed";
  let rec collect position ids =
    if position = 0 then Array.of_list ids
    else
      let start, id = previous.(position) in
      if start < 0 then failwith "Semantic_sqlite: tokenization path broken";
      collect start (id :: ids)
  in
  collect size []
