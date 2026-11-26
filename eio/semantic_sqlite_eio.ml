let embed_passages ~domain_mgr ~max_fibers engine texts =
  if max_fibers <= 0 then invalid_arg "Semantic_sqlite_eio.embed_passages: max_fibers must be positive";
  Eio.Fiber.List.map ~max_fibers (fun text ->
    Eio.Domain_manager.run_raw domain_mgr (fun () -> Semantic_sqlite.embed_passage engine text)) texts
