let () =
  match Sys.getenv_opt "SEMANTIC_SQLITE_TEST_MODEL", Sys.getenv_opt "SEMANTIC_SQLITE_TEST_TOKENIZER" with
  | Some model, Some tokenizer ->
      let engine = Semantic_sqlite.open_model ~model ~tokenizer in
      let texts = [ "un chat noir"; "database schema"; "a black cat" ] in
      let serial = List.map (Semantic_sqlite.embed_passage engine) texts in
      let parallel =
        Eio_main.run (fun env ->
          Semantic_sqlite_eio.embed_passages
            ~domain_mgr:(Eio.Stdenv.domain_mgr env) ~max_fibers:2 engine texts) in
      List.iter2 (fun a b ->
        if Semantic_sqlite.cosine a b < 0.99999 then failwith "parallel embedding differs")
        serial parallel;
      let single = Semantic_sqlite.open_model_with ~threads:1 ~model ~tokenizer () in
      let parallel_single =
        Eio_main.run (fun env ->
          Semantic_sqlite_eio.embed_passages
            ~domain_mgr:(Eio.Stdenv.domain_mgr env) ~max_fibers:2 single texts) in
      List.iter2 (fun a b ->
        if Semantic_sqlite.cosine a b < 0.99999 then
          failwith "parallel embedding with threads:1 differs")
        serial parallel_single
  | _ -> ()
