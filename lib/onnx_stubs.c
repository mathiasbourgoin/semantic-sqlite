#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <onnxruntime_c_api.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  const OrtApi *api;
  void *library;
  OrtEnv *env;
  OrtSession *session;
} semantic_sqlite_session;

static void session_finalize(value v) {
  semantic_sqlite_session *s = (semantic_sqlite_session *)Data_custom_val(v);
  if (s->session) s->api->ReleaseSession(s->session);
  if (s->env) s->api->ReleaseEnv(s->env);
  if (s->library) dlclose(s->library);
}

static struct custom_operations session_ops = {
  "semantic-sqlite.ort", session_finalize, custom_compare_default, custom_hash_default,
  custom_serialize_default, custom_deserialize_default, custom_compare_ext_default,
  custom_fixed_length_default
};

static void raise_status(const OrtApi *api, OrtStatus *status) {
  if (status) {
    const char *message = api->GetErrorMessage(status);
    char copy[4096];
    snprintf(copy, sizeof(copy), "%s", message ? message : "ONNX Runtime error");
    api->ReleaseStatus(status);
    caml_failwith(copy);
  }
}

/* [threads] = 0 keeps ONNX Runtime's default thread pools. */
CAMLprim value semantic_sqlite_ort_open(value path, value threads) {
  CAMLparam2(path, threads);
  CAMLlocal1(result);
  const char *library_path = getenv("ONNXRUNTIME_LIBRARY");
  if (!library_path || !*library_path) library_path = "libonnxruntime.so";
  void *library = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
  if (!library) caml_failwith("Semantic_sqlite: cannot load ONNX Runtime; set ONNXRUNTIME_LIBRARY");
  const OrtApiBase *(*get_api_base)(void) = dlsym(library, "OrtGetApiBase");
  if (!get_api_base) {
    dlclose(library);
    caml_failwith("Semantic_sqlite: ONNX Runtime C API symbol missing");
  }
  const OrtApi *api = get_api_base()->GetApi(ORT_API_VERSION);
  if (!api) {
    dlclose(library);
    caml_failwith("Semantic_sqlite: incompatible ONNX Runtime C API");
  }
  result = caml_alloc_custom(&session_ops, sizeof(semantic_sqlite_session), 0, 1);
  semantic_sqlite_session *s = (semantic_sqlite_session *)Data_custom_val(result);
  s->api = api;
  s->library = library;
  s->env = NULL;
  s->session = NULL;
  OrtSessionOptions *options = NULL;
  raise_status(api, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "semantic-sqlite", &s->env));
  raise_status(api, api->CreateSessionOptions(&options));
  int thread_count = Int_val(threads);
  OrtStatus *status = NULL;
  if (thread_count > 0) status = api->SetIntraOpNumThreads(options, thread_count);
  if (!status)
    status = api->CreateSession(s->env, String_val(path), options, &s->session);
  api->ReleaseSessionOptions(options);
  raise_status(api, status);
  CAMLreturn(result);
}

CAMLprim value semantic_sqlite_ort_run(value handle, value ids) {
  CAMLparam2(handle, ids);
  CAMLlocal3(result, row, scalar);
  semantic_sqlite_session *s = (semantic_sqlite_session *)Data_custom_val(handle);
  const OrtApi *api = s->api;
  mlsize_t count = Wosize_val(ids);
  if (count == 0 || count > 512) caml_invalid_argument("Semantic_sqlite: token count must be 1..512");
  int64_t *input_ids = malloc(count * sizeof(int64_t));
  int64_t *mask = malloc(count * sizeof(int64_t));
  int64_t *types = calloc(count, sizeof(int64_t));
  if (!input_ids || !mask || !types) {
    free(input_ids); free(mask); free(types);
    caml_raise_out_of_memory();
  }
  for (mlsize_t i = 0; i < count; ++i) {
    input_ids[i] = Long_val(Field(ids, i));
    mask[i] = 1;
  }
  int64_t shape[2] = {1, (int64_t)count};
  OrtMemoryInfo *memory = NULL;
  OrtValue *input[3] = {NULL, NULL, NULL};
  OrtValue *output = NULL;
  const char *input_names[3] = {"input_ids", "attention_mask", "token_type_ids"};
  const char *output_names[1] = {"last_hidden_state"};
  OrtStatus *status = api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory);
  if (!status) status = api->CreateTensorWithDataAsOrtValue(memory, input_ids,
    count * sizeof(int64_t), shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input[0]);
  if (!status) status = api->CreateTensorWithDataAsOrtValue(memory, mask,
    count * sizeof(int64_t), shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input[1]);
  if (!status) status = api->CreateTensorWithDataAsOrtValue(memory, types,
    count * sizeof(int64_t), shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input[2]);
  if (!status) {
    caml_enter_blocking_section();
    status = api->Run(s->session, NULL, input_names,
      (const OrtValue *const *)input, 3, output_names, 1, &output);
    caml_leave_blocking_section();
  }
  if (status) goto cleanup_error;

  OrtTensorTypeAndShapeInfo *info = NULL;
  status = api->GetTensorTypeAndShape(output, &info);
  if (status) goto cleanup_error;
  size_t dims_count = 0;
  status = api->GetDimensionsCount(info, &dims_count);
  if (status) { api->ReleaseTensorTypeAndShapeInfo(info); goto cleanup_error; }
  if (dims_count != 3) {
    api->ReleaseTensorTypeAndShapeInfo(info);
    status = api->CreateStatus(ORT_INVALID_ARGUMENT, "Semantic_sqlite: expected rank-3 model output");
    goto cleanup_error;
  }
  ONNXTensorElementDataType element_type;
  status = api->GetTensorElementType(info, &element_type);
  if (status) { api->ReleaseTensorTypeAndShapeInfo(info); goto cleanup_error; }
  if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    api->ReleaseTensorTypeAndShapeInfo(info);
    status = api->CreateStatus(ORT_INVALID_ARGUMENT, "Semantic_sqlite: expected float32 model output");
    goto cleanup_error;
  }
  int64_t dims[3];
  status = api->GetDimensions(info, dims, 3);
  api->ReleaseTensorTypeAndShapeInfo(info);
  if (status) goto cleanup_error;
  if (dims[0] != 1 || dims[1] != (int64_t)count || dims[2] <= 0 || dims[2] > 16384) {
    status = api->CreateStatus(ORT_INVALID_ARGUMENT, "Semantic_sqlite: unexpected model output dimensions");
    goto cleanup_error;
  }
  float *data = NULL;
  status = api->GetTensorMutableData(output, (void **)&data);
  if (status) goto cleanup_error;
  result = caml_alloc(count, 0);
  for (mlsize_t i = 0; i < count; ++i) {
    row = caml_alloc_float_array((mlsize_t)dims[2]);
    for (mlsize_t j = 0; j < (mlsize_t)dims[2]; ++j)
      Store_double_field(row, j, (double)data[i * dims[2] + j]);
    Store_field(result, i, row);
  }
  api->ReleaseValue(output);
  api->ReleaseValue(input[0]);
  api->ReleaseValue(input[1]);
  api->ReleaseValue(input[2]);
  api->ReleaseMemoryInfo(memory);
  free(input_ids); free(mask); free(types);
  CAMLreturn(result);

cleanup_error:
  if (output) api->ReleaseValue(output);
  if (input[0]) api->ReleaseValue(input[0]);
  if (input[1]) api->ReleaseValue(input[1]);
  if (input[2]) api->ReleaseValue(input[2]);
  if (memory) api->ReleaseMemoryInfo(memory);
  free(input_ids); free(mask); free(types);
  raise_status(api, status);
  caml_failwith("Semantic_sqlite: unknown ONNX Runtime error");
}
