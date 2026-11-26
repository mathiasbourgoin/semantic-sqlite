let ok = function Ok value -> value | Error error -> failwith error
let now = Unix.gettimeofday

let env_int name default =
  match Sys.getenv_opt name with
  | None -> default
  | Some value -> int_of_string value

let env_float name default =
  match Sys.getenv_opt name with
  | None -> default
  | Some value -> float_of_string value

let normalize vector =
  let length = sqrt (Array.fold_left (fun sum x -> sum +. x *. x) 0. vector) in
  Array.map (fun x -> x /. length) vector

let median values =
  let sorted = List.sort Float.compare values |> Array.of_list in
  let n = Array.length sorted in
  if n mod 2 = 0 then (sorted.(n / 2 - 1) +. sorted.(n / 2)) /. 2.
  else sorted.(n / 2)

let () =
  let db_path =
    match Sys.getenv_opt "SEMANTIC_SQLITE_BENCH_DB" with
    | Some path -> path
    | None -> failwith "set SEMANTIC_SQLITE_BENCH_DB to a scratch SQLite database path" in
  let count = env_int "SEMANTIC_SQLITE_BENCH_N" 1_000_000 in
  let dimension = env_int "SEMANTIC_SQLITE_BENCH_DIM" 384 in
  let query_count = env_int "SEMANTIC_SQLITE_BENCH_QUERIES" 10 in
  let nbucket = env_int "SEMANTIC_SQLITE_BENCH_BUCKETS" 256 in
  let training_limit = env_int "SEMANTIC_SQLITE_BENCH_TRAIN" 16_384 in
  let threads = env_int "SEMANTIC_SQLITE_BENCH_THREADS" 16 in
  let nprobe = env_float "SEMANTIC_SQLITE_BENCH_NPROBE" 0.05 in
  let reuse = Sys.getenv_opt "SEMANTIC_SQLITE_BENCH_REUSE" = Some "1" in
  let corpus = Option.value (Sys.getenv_opt "SEMANTIC_SQLITE_BENCH_CORPUS") ~default:"clustered" in
  if corpus <> "clustered" && corpus <> "uniform" then
    invalid_arg "SEMANTIC_SQLITE_BENCH_CORPUS must be clustered or uniform";
  if count < query_count || dimension < 2 || query_count < 1 then
    invalid_arg "invalid benchmark size";
  let db = Sqlite3.db_open db_path in
  Fun.protect ~finally:(fun () -> ignore (Sqlite3.db_close db)) (fun () ->
    List.iter (fun pragma ->
      match Sqlite3.exec db pragma with
      | Sqlite3.Rc.OK -> ()
      | rc -> failwith ("SQLite pragma: " ^ Sqlite3.Rc.to_string rc))
      [ "PRAGMA journal_mode=OFF"; "PRAGMA synchronous=OFF";
        "PRAGMA temp_store=MEMORY"; "PRAGMA cache_size=-524288" ];
    let index =
      if reuse then Semantic_sqlite_sqlite3.open_readonly ~model_id:"synthetic-clusters-v1" db |> ok
      else Semantic_sqlite_sqlite3.attach ~model_id:"synthetic-clusters-v1" db |> ok in
    let random = Random.State.make [| 0x0a11; 0x2026 |] in
    let cluster_count = min 256 (max 2 (count / 100)) in
    let centroids = Array.init cluster_count (fun _ ->
      Array.init dimension (fun _ -> Random.State.float random 2. -. 1.) |> normalize) in
    let selected = ref [] in
    let select_every = max 1 (count / query_count) in
    let batch = ref [] in
    let t0 = now () in
    for i = 1 to count do
      let vector =
        if corpus = "uniform" then
          Array.init dimension (fun _ -> Random.State.float random 2. -. 1.) |> normalize
        else
          let center = centroids.(i mod cluster_count) in
          Array.init dimension (fun j ->
            center.(j) +. (Random.State.float random 2. -. 1.) *. 0.10) |> normalize in
      if i mod select_every = 0 && List.length !selected < query_count then
        selected := vector :: !selected;
      if not reuse then (
        batch := (string_of_int i, "synthetic passage " ^ string_of_int i, vector) :: !batch;
        if i mod 500 = 0 || i = count then (
          Semantic_sqlite_sqlite3.upsert_many index (List.rev !batch) |> ok;
          batch := []));
      if not reuse && i mod 50_000 = 0 then (
        Printf.eprintf "indexed %d/%d in %.1fs\n%!" i count (now () -. t0))
    done;
    let ingestion_seconds = if reuse then 0. else now () -. t0 in
    let t_train = now () in
    if not reuse then Semantic_sqlite_sqlite3.rebuild_ann ~threads index ~nbucket ~training_limit |> ok;
    let training_seconds = if reuse then 0. else now () -. t_train in
    let queries = List.rev !selected |> List.map (fun vector ->
      Array.map (fun x -> x +. (Random.State.float random 2. -. 1.) *. 0.02) vector
      |> normalize) in
    let exact_times = ref [] and ann_times = ref [] and recalls = ref [] in
    List.iteri (fun number query ->
      let measure f = let start = now () in let result = f () in result, now () -. start in
      let exact_search () =
        Semantic_sqlite_sqlite3.search_vector ~exact:true index ~query ~limit:10 |> ok in
      let ann_search () = Semantic_sqlite_sqlite3.search_vector ~nprobe index ~query ~limit:10 |> ok in
      let exact, exact_time, ann, ann_time =
        if number mod 2 = 0 then
          let ann, ann_time = measure ann_search in
          let exact, exact_time = measure exact_search in
          exact, exact_time, ann, ann_time
        else
          let exact, exact_time = measure exact_search in
          let ann, ann_time = measure ann_search in
          exact, exact_time, ann, ann_time in
      exact_times := exact_time :: !exact_times;
      ann_times := ann_time :: !ann_times;
      let exact_ids = List.map fst exact in
      let overlap =
        List.fold_left (fun count (id, _) -> if List.mem id exact_ids then count + 1 else count) 0 ann in
      let recall = float overlap /. 10. in
      recalls := recall :: !recalls;
      Printf.eprintf "query %d/%d recall@10=%.2f exact=%.3fs ann=%.3fs\n%!"
        (number + 1) query_count recall (List.hd !exact_times) (List.hd !ann_times))
      queries;
    let mean_recall = List.fold_left ( +. ) 0. !recalls /. float (List.length !recalls) in
    let file_size = (Unix.stat db_path).Unix.st_size in
    Printf.printf
      "{\"corpus\":\"%s\",\"count\":%d,\"dimension\":%d,\"queries\":%d,\"buckets\":%d,\"training_limit\":%d,\"nprobe\":%.2f,\"ingestion_seconds\":%.3f,\"training_seconds\":%.3f,\"recall_at_10\":%.4f,\"exact_median_ms\":%.3f,\"ann_median_ms\":%.3f,\"db_bytes\":%d}\n%!"
      corpus count dimension query_count nbucket training_limit nprobe ingestion_seconds
      training_seconds mean_recall (median !exact_times *. 1000.)
      (median !ann_times *. 1000.) file_size)
