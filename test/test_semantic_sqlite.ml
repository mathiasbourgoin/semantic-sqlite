let check condition message = if not condition then failwith message

let passages =
  List.init 24 (fun i ->
    String.concat " "
      (List.init 60 (fun j ->
         Printf.sprintf "passage %d mot %d recherche sémantique locale" i j)))

(* Process CPU time (all threads) divided by wall time over a batch. *)
let cores_used engine =
  let before = Unix.times () and started = Unix.gettimeofday () in
  let vectors = List.map (Semantic_sqlite.embed_passage engine) passages in
  let after = Unix.times () and elapsed = Unix.gettimeofday () -. started in
  let cpu =
    after.Unix.tms_utime -. before.Unix.tms_utime
    +. after.Unix.tms_stime -. before.Unix.tms_stime in
  (cpu /. elapsed, vectors)

let check_thread_option ~model ~tokenizer =
  List.iter (fun threads ->
    try ignore (Semantic_sqlite.open_model_with ~threads ~model ~tokenizer ());
      failwith (Printf.sprintf "threads:%d accepted" threads)
    with Invalid_argument _ -> ())
    [ 0; -1; 1025; 100_000; 0x8000_0000 ];
  let single = Semantic_sqlite.open_model_with ~threads:1 ~model ~tokenizer () in
  ignore (cores_used single);
  let single_cores, single_vectors = cores_used single in
  Printf.printf "cores used: threads:1 %.2f\n%!" single_cores;
  check (single_cores <= 1.5) "threads:1 uses at most about one core";
  let wide = Semantic_sqlite.open_model_with ~threads:200 ~model ~tokenizer () in
  let text = List.hd passages in
  check (Semantic_sqlite.cosine (Semantic_sqlite.embed_passage wide text) (List.hd single_vectors)
         > 0.99999) "threads:200 embedding matches threads:1";
  if Domain.recommended_domain_count () < 4 then
    print_endline "skip: default-pool core comparison needs at least 4 cores"
  else (
    let default = Semantic_sqlite.open_model ~model ~tokenizer in
    ignore (cores_used default);
    let default_cores, default_vectors = cores_used default in
    Printf.printf "cores used: default %.2f\n%!" default_cores;
    check (default_cores > single_cores +. 0.5)
      "default thread pool uses more cores than threads:1";
    List.iter2 (fun a b ->
      check (Semantic_sqlite.cosine a b > 0.99999) "embedding depends on thread count")
      single_vectors default_vectors)

let () =
  let a = [| 1.; 0. |] and b = [| 0.; 1. |] in
  check (Semantic_sqlite.cosine a a > 0.9999) "identity cosine";
  check (abs_float (Semantic_sqlite.cosine a b) < 0.0001) "orthogonal cosine";
  let ranked = Semantic_sqlite.rank ~query:a [ "b", b; "a", a ] in
  check (fst (List.hd ranked) = "a") "vector ranking";
  let fused = Semantic_sqlite.reciprocal_rank_fusion [ [ "a"; "b" ]; [ "b"; "c" ] ] in
  check (fst (List.hd fused) = "b") "rank fusion";
  let deduplicated = Semantic_sqlite.reciprocal_rank_fusion [ [ "a"; "a" ]; [ "b" ] ] in
  check (fst (List.hd deduplicated) = "a") "duplicate rank ignored";
  (try ignore (Semantic_sqlite.cosine [| nan |] [| 1. |]); failwith "NaN accepted"
   with Invalid_argument _ -> ());
  let blob = Semantic_sqlite.encode_f32le [| 1.; -0.5; 0. |] in
  check (Semantic_sqlite.decode_f32le blob = [| 1.; -0.5; 0. |]) "f32 blob round-trip";
  match Sys.getenv_opt "SEMANTIC_SQLITE_TEST_MODEL", Sys.getenv_opt "SEMANTIC_SQLITE_TEST_TOKENIZER" with
  | Some model, Some tokenizer ->
      let engine = Semantic_sqlite.open_model ~model ~tokenizer in
      let ids = Semantic_sqlite.tokenize engine "query: chat noir" in
      check (ids.(0) = 0 && ids.(Array.length ids - 1) = 2) "E5 special tokens";
      check (ids = [| 0; 41; 1294; 12; 3245; 86494; 2 |])
        "E5 French tokenization matches SentencePiece";
      List.iter (fun (text, expected) ->
        let actual = Semantic_sqlite.tokenize engine text in
        if actual <> expected then
          failwith (Printf.sprintf "token mismatch for %S: got %s" text
            (String.concat "," (List.map string_of_int (Array.to_list actual)))))
        [ "query: black cat", [| 0; 41; 1294; 12; 22556; 7515; 2 |];
          "passage: un chat noir", [| 0; 46692; 12; 51; 3245; 86494; 2 |];
          "query: Où est l’entrée ?", [| 0; 41; 1294; 12; 180; 8612; 437; 96; 26; 66989; 705; 2 |];
          "query: café\tnoir", [| 0; 41; 1294; 12; 26216; 86494; 2 |];
          "query:  déjà   vu  ", [| 0; 41; 1294; 12; 15154; 13946; 6; 2 |];
          "query: naïve façade", [| 0; 41; 1294; 12; 24; 9392; 272; 65335; 112; 2 |];
          "query: 🚀 search", [| 0; 41; 1294; 12; 6; 247206; 33938; 2 |] ];
      List.iter (fun text ->
        check
          (Semantic_sqlite.tokenize engine text =
           Semantic_sqlite.tokenize engine "query: foo bar")
          "E5 whitespace normalization")
        [ "query: foo\u{200b}bar"; "query: foo\u{2028}bar" ];
      check
        (Semantic_sqlite.tokenize engine "query: foo\007bar" =
         Semantic_sqlite.tokenize engine "query: foobar")
        "E5 control normalization";
      let query = Semantic_sqlite.embed_query engine "black cat" in
      let related = Semantic_sqlite.embed_passage engine "un chat noir" in
      let unrelated = Semantic_sqlite.embed_passage engine "database schema migration" in
      check (Array.length query = 384) "E5 embedding dimensions";
      check (Semantic_sqlite.cosine query related > Semantic_sqlite.cosine query unrelated)
        "English-to-French semantic ranking";
      check_thread_option ~model ~tokenizer
  | _ -> ()
