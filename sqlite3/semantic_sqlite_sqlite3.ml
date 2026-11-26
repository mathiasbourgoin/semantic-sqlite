type t = { db : Sqlite3.db; readonly : bool }

let ( let* ) = Result.bind
let guard f =
  try f () with
  | (Out_of_memory | Stack_overflow) as e -> raise e
  | e -> Error (Printexc.to_string e)

let sql_error db rc =
  Printf.sprintf "SQLite %s: %s" (Sqlite3.Rc.to_string rc) (Sqlite3.errmsg db)

let with_stmt db sql args f =
  let stmt = Sqlite3.prepare db sql in
  Fun.protect ~finally:(fun () -> ignore (Sqlite3.finalize stmt)) (fun () ->
    match Sqlite3.bind_values stmt args with
    | Sqlite3.Rc.OK -> f stmt
    | rc -> Error (sql_error db rc))

let run db sql args =
  with_stmt db sql args (fun stmt ->
    match Sqlite3.step stmt with
    | Sqlite3.Rc.DONE -> Ok ()
    | rc -> Error (sql_error db rc))

let fold db sql args init f =
  with_stmt db sql args (fun stmt ->
    let rec loop acc =
      match Sqlite3.step stmt with
      | Sqlite3.Rc.ROW -> loop (f acc stmt)
      | Sqlite3.Rc.DONE -> Ok acc
      | rc -> Error (sql_error db rc) in
    loop init)

let first db sql args f =
  with_stmt db sql args (fun stmt ->
    match Sqlite3.step stmt with
    | Sqlite3.Rc.ROW -> Ok (Some (f stmt))
    | Sqlite3.Rc.DONE -> Ok None
    | rc -> Error (sql_error db rc))

let transaction db f =
  let* () = run db "SAVEPOINT oonnxx_tx" [] in
  try
    match f () with
    | Ok result ->
        (match run db "RELEASE SAVEPOINT oonnxx_tx" [] with
         | Ok () -> Ok result
         | Error error ->
             ignore (run db "ROLLBACK TO SAVEPOINT oonnxx_tx" []);
             ignore (run db "RELEASE SAVEPOINT oonnxx_tx" []);
             Error error)
    | Error error ->
        ignore (run db "ROLLBACK TO SAVEPOINT oonnxx_tx" []);
        ignore (run db "RELEASE SAVEPOINT oonnxx_tx" []);
        Error error
  with error ->
    ignore (run db "ROLLBACK TO SAVEPOINT oonnxx_tx" []);
    ignore (run db "RELEASE SAVEPOINT oonnxx_tx" []);
    raise error

let extension_path () =
  match Sys.getenv_opt "SEMANTIC_SQLITE_VEC1_EXTENSION" with
  | Some path when path <> "" -> path
  | _ ->
      (match Sys.getenv_opt "OONNXX_VEC1_EXTENSION" with
       | Some path when path <> "" -> path
       | _ -> Filename.concat (Findlib.package_directory "semantic-sqlite") "vec1.so")

let load_vec1 db =
  if Sqlite3.sqlite_version () < 3038000 then
    Error "Semantic_sqlite_sqlite3: SQLite 3.38.0 or newer is required for Vec1"
  else
    match (try first db "SELECT vec1_info()" []
                 (fun stmt -> Sqlite3.column_text stmt 0)
           with Sqlite3.Error _ -> Error "Vec1 is not loaded") with
    | Ok (Some info) when String.starts_with ~prefix:"version 0.8" info -> Ok ()
    | Ok (Some info) -> Error ("Semantic_sqlite_sqlite3: incompatible Vec1 extension: " ^ info)
    | _ ->
        let path = extension_path () in
        if not (Sys.file_exists path) then
          Error ("Semantic_sqlite_sqlite3: Vec1 extension not found: " ^ path)
        else if not (Sqlite3.enable_load_extension db true) then
          Error "Semantic_sqlite_sqlite3: cannot enable SQLite extension loading"
        else
          Fun.protect ~finally:(fun () -> ignore (Sqlite3.enable_load_extension db false))
            (fun () ->
              let quoted = String.concat "''" (String.split_on_char '\'' path) in
              match Sqlite3.exec db ("SELECT load_extension('" ^ quoted ^ "')") with
              | Sqlite3.Rc.OK -> Ok ()
              | rc -> Error ("Semantic_sqlite_sqlite3: cannot load Vec1: " ^ sql_error db rc))

let schema =
  [ "CREATE TABLE IF NOT EXISTS oonnxx_metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)";
    "CREATE TABLE IF NOT EXISTS oonnxx_documents (id TEXT PRIMARY KEY, text TEXT NOT NULL, embedding BLOB NOT NULL, dimension INTEGER NOT NULL)";
    "CREATE VIRTUAL TABLE IF NOT EXISTS oonnxx_vec1 USING vec1(embedding)";
    "CREATE VIRTUAL TABLE IF NOT EXISTS oonnxx_fts USING fts5(text, content='oonnxx_documents', content_rowid='rowid', tokenize='unicode61 remove_diacritics 1')";
    "DROP TRIGGER IF EXISTS oonnxx_ai";
    "DROP TRIGGER IF EXISTS oonnxx_ad";
    "DROP TRIGGER IF EXISTS oonnxx_au";
    "CREATE TRIGGER oonnxx_ai AFTER INSERT ON oonnxx_documents BEGIN INSERT INTO oonnxx_fts(rowid, text) VALUES (new.rowid, new.text); INSERT INTO oonnxx_vec1(rowid, embedding) VALUES (new.rowid, new.embedding); END";
    "CREATE TRIGGER oonnxx_ad AFTER DELETE ON oonnxx_documents BEGIN INSERT INTO oonnxx_fts(oonnxx_fts, rowid, text) VALUES ('delete', old.rowid, old.text); DELETE FROM oonnxx_vec1 WHERE rowid=old.rowid; END";
    "CREATE TRIGGER oonnxx_au AFTER UPDATE ON oonnxx_documents BEGIN INSERT INTO oonnxx_fts(oonnxx_fts, rowid, text) VALUES ('delete', old.rowid, old.text); INSERT INTO oonnxx_fts(rowid, text) VALUES (new.rowid, new.text); UPDATE oonnxx_vec1 SET embedding=new.embedding WHERE rowid=new.rowid; END" ]

let metadata db name =
  first db "SELECT value FROM oonnxx_metadata WHERE name=?"
    [ Sqlite3.Data.TEXT name ] (fun stmt -> Sqlite3.column_text stmt 0)

let set_metadata db name value =
  run db
    "INSERT INTO oonnxx_metadata(name,value) VALUES (?,?) ON CONFLICT(name) DO UPDATE SET value=excluded.value"
    [ Sqlite3.Data.TEXT name; Sqlite3.Data.TEXT value ]

let check_model_id db model_id =
  let* existing = metadata db "model_id" in
  match existing with
  | Some id when id = model_id -> Ok ()
  | Some _ -> Error "Semantic_sqlite_sqlite3: model_id differs; rebuild the vector index"
  | None -> Error "Semantic_sqlite_sqlite3: index has no model_id"

let attach ~model_id db =
  guard (fun () ->
    if model_id = "" then invalid_arg "Semantic_sqlite_sqlite3: empty model_id";
    let* () = load_vec1 db in
    let* () = transaction db (fun () ->
      let* () = List.fold_left (fun state sql -> let* () = state in run db sql []) (Ok ()) schema in
      let* existing = metadata db "model_id" in
      let* () =
        match existing with
        | None -> set_metadata db "model_id" model_id
        | Some _ -> check_model_id db model_id in
      run db
        "INSERT INTO oonnxx_vec1(rowid, embedding) SELECT d.rowid, d.embedding FROM oonnxx_documents d WHERE NOT EXISTS (SELECT 1 FROM oonnxx_vec1 v WHERE v.rowid=d.rowid)"
        []) in
    Ok { db; readonly = false })

let open_readonly ~model_id db =
  guard (fun () ->
    if model_id = "" then invalid_arg "Semantic_sqlite_sqlite3: empty model_id";
    let* () = load_vec1 db in
    let* () = check_model_id db model_id in
    Ok { db; readonly = true })

let validate_vector vector =
  if Array.length vector = 0 then invalid_arg "Semantic_sqlite_sqlite3: empty vector";
  Array.iter (fun value ->
    if not (Float.is_finite value) then invalid_arg "Semantic_sqlite_sqlite3: nonfinite vector") vector;
  ignore (Semantic_sqlite.cosine vector vector)

let require_writable t =
  if t.readonly then Error "Semantic_sqlite_sqlite3: index is read-only" else Ok ()

let upsert_many t rows =
  guard (fun () ->
    let* () = require_writable t in
    match rows with
    | [] -> Ok ()
    | (_, _, first_vector) :: _ ->
        let dimension = Array.length first_vector in
        List.iter (fun (id, _, vector) ->
          if id = "" then invalid_arg "Semantic_sqlite_sqlite3: empty id";
          validate_vector vector;
          if Array.length vector <> dimension then
            invalid_arg "Semantic_sqlite_sqlite3: mixed dimensions in batch") rows;
        transaction t.db (fun () ->
          let dim_text = string_of_int dimension in
          let* existing = metadata t.db "dimension" in
          let* () =
            match existing with
            | None -> set_metadata t.db "dimension" dim_text
            | Some value when value = dim_text -> Ok ()
            | Some _ -> Error "Semantic_sqlite_sqlite3: vector dimension differs from index" in
          List.fold_left (fun state (id, text, vector) ->
            let* () = state in
            run t.db
              "INSERT INTO oonnxx_documents(id,text,embedding,dimension) VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET text=excluded.text, embedding=excluded.embedding, dimension=excluded.dimension"
              [ Sqlite3.Data.TEXT id; Sqlite3.Data.TEXT text;
                Sqlite3.Data.BLOB (Semantic_sqlite.encode_f32le vector);
                Sqlite3.Data.INT (Int64.of_int dimension) ])
            (Ok ()) rows))

let upsert t ~id ~text ~vector = upsert_many t [ id, text, vector ]

let delete t ~id =
  guard (fun () ->
    let* () = require_writable t in
    run t.db "DELETE FROM oonnxx_documents WHERE id=?" [ Sqlite3.Data.TEXT id ])

(* ASCII whitespace and NUL. SQLite stops reading a query string at a NUL
   byte, so a NUL inside a quoted word would leave it unterminated. *)
let is_query_separator = function
  | ' ' | '\t' | '\n' | '\011' | '\012' | '\r' | '\000' -> true
  | _ -> false

let search_fts t ~query ~limit =
  guard (fun () ->
    if limit <= 0 then invalid_arg "Semantic_sqlite_sqlite3: limit must be positive";
    if String.for_all is_query_separator query then Ok [] else
    let* reversed =
      fold t.db
        "SELECT d.id FROM oonnxx_fts JOIN oonnxx_documents d ON d.rowid=oonnxx_fts.rowid WHERE oonnxx_fts MATCH ? ORDER BY bm25(oonnxx_fts), d.id LIMIT ?"
        [ Sqlite3.Data.TEXT query; Sqlite3.Data.INT (Int64.of_int limit) ]
        [] (fun acc stmt -> Sqlite3.column_text stmt 0 :: acc) in
    Ok (List.rev reversed))

let check_query_dimension t query =
  let* dimension = metadata t.db "dimension" in
  match dimension with
  | None -> Ok ()
  | Some value when value = string_of_int (Array.length query) -> Ok ()
  | Some _ -> Error "Semantic_sqlite_sqlite3: query dimension differs from index"

(* Result order: descending score, then ascending id. Ids are unique, so the
   order is total and the selected set does not depend on scan order. *)
let compare_ranked (id_a, a) (id_b, b) =
  let order = Float.compare b a in
  if order = 0 then String.compare id_a id_b else order

(* Bounded selection of the [limit] best entries in O(n log limit): a binary
   heap whose root is the worst entry kept. Storage grows with the number of
   entries kept, never with [limit] itself. *)
module Top_k = struct
  type t = { limit : int; mutable items : (string * float) array; mutable size : int }

  let create limit = { limit; items = [||]; size = 0 }

  (* [worse a b]: [a] ranks after [b], so it belongs nearer the root. *)
  let worse a b = compare_ranked a b > 0

  let swap heap i j =
    let item = heap.items.(i) in
    heap.items.(i) <- heap.items.(j);
    heap.items.(j) <- item

  let rec sift_up heap i =
    let parent = (i - 1) / 2 in
    if i > 0 && worse heap.items.(i) heap.items.(parent) then (
      swap heap i parent;
      sift_up heap parent)

  let rec sift_down heap i =
    let left = (2 * i) + 1 in
    let right = left + 1 in
    let target = ref i in
    if left < heap.size && worse heap.items.(left) heap.items.(!target) then target := left;
    if right < heap.size && worse heap.items.(right) heap.items.(!target) then target := right;
    if !target <> i then (
      swap heap i !target;
      sift_down heap !target)

  let push heap item =
    if heap.size = Array.length heap.items then (
      let capacity = min heap.limit (max 16 (2 * heap.size)) in
      let items = Array.make capacity item in
      Array.blit heap.items 0 items 0 heap.size;
      heap.items <- items);
    heap.items.(heap.size) <- item;
    heap.size <- heap.size + 1;
    sift_up heap (heap.size - 1)

  let add heap item =
    if heap.size < heap.limit then push heap item
    else if worse heap.items.(0) item then (
      heap.items.(0) <- item;
      sift_down heap 0)

  let to_sorted_list heap =
    let items = Array.sub heap.items 0 heap.size in
    Array.sort compare_ranked items;
    Array.to_list items
end

let squared_norm vector =
  let sum = ref 0. in
  for i = 0 to Array.length vector - 1 do
    sum := !sum +. (vector.(i) *. vector.(i))
  done;
  !sum

(* [Semantic_sqlite.cosine query (Semantic_sqlite.decode_f32le blob)], bit for bit and with the
   same errors, without decoding the blob into an array. The sums follow the
   same order; [query_norm] is the query's squared norm, computed once. A
   float32 square cannot overflow a double sum, so a non-finite row sum means
   a non-finite element. The blob holds [Array.length query] floats. *)
let cosine_f32le ~query ~query_norm blob =
  let dot = ref 0. and norm = ref 0. in
  for i = 0 to Array.length query - 1 do
    let x = Int32.float_of_bits (String.get_int32_le blob (4 * i)) in
    dot := !dot +. (query.(i) *. x);
    norm := !norm +. (x *. x)
  done;
  if not (Float.is_finite !norm) then invalid_arg "Semantic_sqlite.cosine: nonfinite vector";
  if query_norm <= 0. || !norm <= 0. then invalid_arg "Semantic_sqlite.cosine: zero vector";
  !dot /. sqrt (query_norm *. !norm)

let search_vector_exact t ~query ~limit =
  guard (fun () ->
    if limit <= 0 then invalid_arg "Semantic_sqlite_sqlite3: limit must be positive";
    validate_vector query;
    let* () = check_query_dimension t query in
    let query_norm = squared_norm query in
    let best = Top_k.create limit in
    let* () =
      fold t.db "SELECT id, embedding, dimension FROM oonnxx_documents" [] ()
        (fun () stmt ->
          let id = Sqlite3.column_text stmt 0 in
          let blob = Sqlite3.column_blob stmt 1 in
          if String.length blob mod 4 <> 0 then
            invalid_arg "Semantic_sqlite.decode_f32le: invalid blob length";
          let length = String.length blob / 4 in
          if length <> Sqlite3.column_int stmt 2 || length <> Array.length query then
            failwith ("Semantic_sqlite_sqlite3: inconsistent vector dimension for " ^ id);
          Top_k.add best (id, cosine_f32le ~query ~query_norm blob)) in
    Ok (Top_k.to_sorted_list best))

let search_vector ?(exact = false) ?(nprobe = 0.05) t ~query ~limit =
  if exact then search_vector_exact t ~query ~limit
  else
    guard (fun () ->
      let* mode = metadata t.db "ann_mode" in
      if mode = None then search_vector_exact t ~query ~limit
      else if mode <> Some "ivf" then Error "Semantic_sqlite_sqlite3: unknown ANN mode"
      else (
      if limit <= 0 then invalid_arg "Semantic_sqlite_sqlite3: limit must be positive";
      if not (Float.is_finite nprobe) || nprobe <= 0. then
        invalid_arg "Semantic_sqlite_sqlite3: nprobe must be positive and finite";
      validate_vector query;
      let* () = check_query_dimension t query in
      let options = Printf.sprintf "{\"K\":%d,\"nprobe\":%.17g}" limit nprobe in
      let* reversed =
        fold t.db
          "SELECT d.id, 1.0-v.distance FROM oonnxx_vec1(?,?) v JOIN oonnxx_documents d ON d.rowid=v.rowid ORDER BY v.distance, d.id LIMIT ?"
          [ Sqlite3.Data.BLOB (Semantic_sqlite.encode_f32le query);
            Sqlite3.Data.TEXT options; Sqlite3.Data.INT (Int64.of_int limit) ]
          [] (fun acc stmt ->
            (Sqlite3.column_text stmt 0, Sqlite3.column_double stmt 1) :: acc) in
      Ok (List.rev reversed)))

let rebuild_ann ?(threads = 1) t ~nbucket ~training_limit =
  guard (fun () ->
    let* () = require_writable t in
    if nbucket < 2 || nbucket > 65536 then
      invalid_arg "Semantic_sqlite_sqlite3: nbucket must be between 2 and 65536";
    if training_limit < nbucket * 4 then
      invalid_arg "Semantic_sqlite_sqlite3: training_limit must be at least 4 * nbucket";
    if threads < 1 then invalid_arg "Semantic_sqlite_sqlite3: threads must be positive";
    let* _ = first t.db "SELECT vec1_config('nthread', ?)"
        [ Sqlite3.Data.INT (Int64.of_int threads) ]
        (fun stmt -> Sqlite3.column_text stmt 0) in
    let options =
      Printf.sprintf "{\"nbucket\":%d,\"distance\":\"cos\",\"quantizer\":\"none\",\"codesize\":0}"
        nbucket in
    let* model =
      first t.db
        "WITH training AS (SELECT embedding FROM oonnxx_vec1 ORDER BY RANDOM() LIMIT ?) SELECT vec1_train(embedding, ?) FROM training"
        [ Sqlite3.Data.INT (Int64.of_int training_limit); Sqlite3.Data.TEXT options ]
        (fun stmt -> Sqlite3.column_blob stmt 0) in
    match model with
    | None -> Error "Semantic_sqlite_sqlite3: no vectors available to train ANN index"
    | Some model ->
        transaction t.db (fun () ->
          let* () = run t.db "INSERT INTO oonnxx_vec1(cmd,arg) VALUES ('rebuild',?)"
              [ Sqlite3.Data.BLOB model ] in
          set_metadata t.db "ann_mode" "ivf"))

let literal_fts_query text =
  let words =
    String.map (fun c -> if is_query_separator c then ' ' else c) text
    |> String.split_on_char ' '
    |> List.filter (fun word ->
         String.exists (fun c ->
           let code = Char.code c in
           (code >= 48 && code <= 57) || (code >= 65 && code <= 90)
           || (code >= 97 && code <= 122) || code >= 128) word) in
  words
  |> List.map (fun word ->
       let escaped = String.concat "\"\"" (String.split_on_char '"' word) in
       "\"" ^ escaped ^ "\"")
  |> String.concat " OR "

let search_hybrid t ~text ~vector ~limit =
  if limit <= 0 then Error "Semantic_sqlite_sqlite3: limit must be positive"
  else
    let candidates = if limit > 3333 then 10000 else limit * 3 in
    let fts_query = literal_fts_query text in
    let* fts =
      if fts_query = "" then Ok []
      else search_fts t ~query:fts_query ~limit:candidates in
    let* semantic = search_vector t ~query:vector ~limit:candidates in
    let ids = List.map fst semantic in
    Ok (Semantic_sqlite.reciprocal_rank_fusion [ fts; ids ]
        |> List.filteri (fun index _ -> index < limit))
