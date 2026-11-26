type t = Semantic_sqlite_sqlite3.t

let raw_db (module C : Caqti_blocking.CONNECTION) =
  match C.driver_connection with
  | Some (Caqti_driver_sqlite3.Driver_connection db) -> Ok db
  | _ -> Error "Semantic_sqlite_caqti: expected the Caqti SQLite3 driver"

let attach ~model_id conn =
  match raw_db conn with
  | Error _ as error -> error
  | Ok db -> Semantic_sqlite_sqlite3.attach ~model_id db

let open_readonly ~model_id conn =
  match raw_db conn with
  | Error _ as error -> error
  | Ok db -> Semantic_sqlite_sqlite3.open_readonly ~model_id db

let upsert = Semantic_sqlite_sqlite3.upsert
let upsert_many = Semantic_sqlite_sqlite3.upsert_many
let delete = Semantic_sqlite_sqlite3.delete
let search_fts = Semantic_sqlite_sqlite3.search_fts
let literal_fts_query = Semantic_sqlite_sqlite3.literal_fts_query
let search_vector = Semantic_sqlite_sqlite3.search_vector
let rebuild_ann = Semantic_sqlite_sqlite3.rebuild_ann
let search_hybrid = Semantic_sqlite_sqlite3.search_hybrid
