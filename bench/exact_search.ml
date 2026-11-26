(* Exact vector search latency on a synthetic corpus of normalized vectors.
   SEMANTIC_SQLITE_BENCH_DB must name a scratch database; it is created on first use
   and reused afterwards. *)

let ok = function Ok value -> value | Error error -> failwith error

let env_int name default =
  match Sys.getenv_opt name with
  | None -> default
  | Some value -> int_of_string value

let gaussian state =
  let u = 1. -. Random.State.float state 1. and v = Random.State.float state 1. in
  sqrt (-2. *. log u) *. cos (2. *. Float.pi *. v)

let normalized state dimension =
  let vector = Array.init dimension (fun _ -> gaussian state) in
  let norm = sqrt (Array.fold_left (fun sum x -> sum +. (x *. x)) 0. vector) in
  Array.map (fun x -> x /. norm) vector

let populate index state ~count ~dimension =
  let batch = 1_000 in
  let rec loop first =
    if first < count then (
      let rows =
        List.init (min batch (count - first)) (fun offset ->
          (Printf.sprintf "doc-%06d" (first + offset), "synthetic passage",
           normalized state dimension)) in
      Semantic_sqlite_sqlite3.upsert_many index rows |> ok;
      loop (first + batch)) in
  loop 0

let stored_count db =
  let stmt = Sqlite3.prepare db "SELECT count(*) FROM oonnxx_documents" in
  Fun.protect ~finally:(fun () -> ignore (Sqlite3.finalize stmt)) (fun () ->
    match Sqlite3.step stmt with
    | Sqlite3.Rc.ROW -> Sqlite3.column_int stmt 0
    | rc -> failwith (Sqlite3.Rc.to_string rc))

let () =
  let path =
    match Sys.getenv_opt "SEMANTIC_SQLITE_BENCH_DB" with
    | Some path -> path
    | None -> failwith "set SEMANTIC_SQLITE_BENCH_DB to a scratch SQLite database path" in
  let count = env_int "SEMANTIC_SQLITE_BENCH_N" 21_928 in
  let dimension = env_int "SEMANTIC_SQLITE_BENCH_DIM" 384 in
  let repeats = env_int "SEMANTIC_SQLITE_BENCH_REPEATS" 5 in
  let limits =
    match Sys.getenv_opt "SEMANTIC_SQLITE_BENCH_LIMITS" with
    | Some text -> List.map int_of_string (String.split_on_char ',' text)
    | None -> [ 10; count ] in
  let db = Sqlite3.db_open path in
  Fun.protect ~finally:(fun () -> ignore (Sqlite3.db_close db)) (fun () ->
    let index = Semantic_sqlite_sqlite3.attach ~model_id:"bench-exact" db |> ok in
    let state = Random.State.make [| 7 |] in
    let stored = stored_count db in
    if stored = 0 then populate index state ~count ~dimension
    else if stored <> count then
      failwith (Printf.sprintf "scratch database holds %d vectors, expected %d"
        stored count);
    let query = normalized (Random.State.make [| 11 |]) dimension in
    List.iter (fun limit ->
      let timings =
        List.init repeats (fun _ ->
          let started = Unix.gettimeofday () in
          let results =
            Semantic_sqlite_sqlite3.search_vector ~exact:true index ~query ~limit |> ok in
          let elapsed = Unix.gettimeofday () -. started in
          if List.length results <> min limit count then
            failwith "unexpected result count";
          elapsed *. 1000.) in
      let sorted = List.sort Float.compare timings |> Array.of_list in
      Printf.printf "n=%d dim=%d limit=%d runs=%d min=%.1fms median=%.1fms max=%.1fms\n%!"
        count dimension limit repeats sorted.(0)
        sorted.(Array.length sorted / 2) sorted.(Array.length sorted - 1))
      limits)
