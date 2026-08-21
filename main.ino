#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <math.h>
#include "esp_heap_caps.h"

const int SD_CS = 10;
#define SEQ_LEN 32
#define D_MODEL 768
#define N_HEADS 12
#define HEAD_DIM (D_MODEL / N_HEADS)
#define N_LAYERS 16
#define D_FF 2048
#define PAD_ID 0
#define BOS_ID 1
#define EOS_ID 2
#define UNK_ID 3
#define MAX_ROWS1 2304 //1536
#define MAX_COLS1 768
#define MAX_ROWS2 768
#define MAX_COLS2 2048//1024

int vocab_size = 0;
char** vocab_words = NULL;
char*  vocab_pool = NULL;
float* enc_state = NULL;
float* dec_state = NULL;
float* enc_mem = NULL;
float* raw_dec_tokens = NULL;
int8_t* emb_weights = NULL;
float*  emb_scales = NULL;
int8_t* proj_weights = NULL;
float*  proj_scales = NULL;
float*  proj_bias = NULL;
float*  logits_buf = NULL;
float* ws_raw_enc_tokens = NULL;
float* ws_norm_state = NULL;
float* ws_sub_out = NULL;
float* ws_Q = NULL;
float* ws_K = NULL;
float* ws_V = NULL;
float* ws_attn_out = NULL;
float* ws_scores = NULL;
float* ws_ffn_h = NULL;
struct ReusableLayer {
  float ln_w[D_MODEL];
  float ln_b[D_MODEL];
  int8_t* w1;
  float* scale_w1;
  float* b1;
  int8_t* w2;
  float* scale_w2;
  float* b2;
} g_layer;

bool initLayerBuffer() {
  g_layer.scale_w1 = (float*)heap_caps_malloc(MAX_ROWS1 * sizeof(float), MALLOC_CAP_SPIRAM);
  g_layer.w1 = (int8_t*)heap_caps_malloc(MAX_ROWS1 * MAX_COLS1 * sizeof(int8_t), MALLOC_CAP_SPIRAM);
  g_layer.b1 = (float*)heap_caps_malloc(MAX_ROWS1 * sizeof(float), MALLOC_CAP_SPIRAM);

  g_layer.scale_w2 = (float*)heap_caps_malloc(MAX_ROWS2 * sizeof(float), MALLOC_CAP_SPIRAM);
  g_layer.w2 = (int8_t*)heap_caps_malloc(MAX_ROWS2 * MAX_COLS2 * sizeof(int8_t), MALLOC_CAP_SPIRAM);
  g_layer.b2 = (float*)heap_caps_malloc(MAX_ROWS2 * sizeof(float), MALLOC_CAP_SPIRAM);

  return (g_layer.scale_w1 && g_layer.w1 && g_layer.b1 &&
          g_layer.scale_w2 && g_layer.w2 && g_layer.b2);
}

void toLowerInPlace(char* s) {
  for (; *s; s++) {
    if (*s >= 'A' && *s <= 'Z') *s = *s - 'A' + 'a';
  }
}

int getWordId(const char* word) {
  for (int i = 0; i < vocab_size; i++) {
    if (strcmp(vocab_words[i], word) == 0) return i;
  }
  return UNK_ID;
}

bool loadVocabulary() {
  File file = SD.open("/vocab.bin", FILE_READ);
  if (!file) return false;
  size_t len = file.size();
  uint8_t* vocab_buf = (uint8_t*)heap_caps_malloc(len, MALLOC_CAP_SPIRAM);
  if (!vocab_buf) { file.close(); return false; }
  if (file.read(vocab_buf, len) != len) { free(vocab_buf); file.close(); return false; }
  file.close();
  size_t offset = 0;
  memcpy(&vocab_size, vocab_buf + offset, sizeof(uint32_t));
  offset += sizeof(uint32_t);
  vocab_words = (char**)heap_caps_malloc(vocab_size * sizeof(char*), MALLOC_CAP_SPIRAM);
  vocab_pool  = (char*)heap_caps_malloc(len, MALLOC_CAP_SPIRAM);
  if (!vocab_words || !vocab_pool) { free(vocab_buf); return false; }
  size_t pool_offset = 0;
  for (int i = 0; i < vocab_size; i++) {
    uint8_t str_len = vocab_buf[offset++];
    vocab_words[i] = &vocab_pool[pool_offset];
    memcpy(vocab_words[i], vocab_buf + offset, str_len);
    vocab_words[i][str_len] = '\0';
    offset += str_len;
    pool_offset += (str_len + 1);
  }
  free(vocab_buf);
  return true;
}

void loadEmbeddingsOnly() {
  Serial.println("Loading embeddings...");
  size_t emb_size  = (size_t)vocab_size * D_MODEL;
  size_t proj_size = (size_t)vocab_size * D_MODEL;
  emb_weights  = (int8_t*)heap_caps_malloc(emb_size * sizeof(int8_t), MALLOC_CAP_SPIRAM);
  emb_scales   = (float*)heap_caps_malloc(vocab_size * sizeof(float), MALLOC_CAP_SPIRAM);
  proj_weights = (int8_t*)heap_caps_malloc(proj_size * sizeof(int8_t), MALLOC_CAP_SPIRAM);
  proj_scales  = (float*)heap_caps_malloc(vocab_size * sizeof(float), MALLOC_CAP_SPIRAM);
  proj_bias    = (float*)heap_caps_malloc(vocab_size * sizeof(float), MALLOC_CAP_SPIRAM);
  if (!emb_weights || !emb_scales || !proj_weights || !proj_scales || !proj_bias) return;
  File file = SD.open("/embeddings.bin", FILE_READ);
  if (!file) return;
  file.read((uint8_t*)emb_scales, vocab_size * sizeof(float));
  file.read((uint8_t*)emb_weights, emb_size);
  file.read((uint8_t*)proj_scales, vocab_size * sizeof(float));
  file.read((uint8_t*)proj_weights, proj_size);
  file.read((uint8_t*)proj_bias, vocab_size * sizeof(float));
  file.close();
  Serial.println("Embeddings loaded.");
}

bool fetchAndParseLayer(const char* filename, bool is_ffn) {
  char path[64];
  snprintf(path, sizeof(path), "/%s", filename);
  File file = SD.open(path, FILE_READ);
  if (!file) return false;
  int rows1 = is_ffn ? D_FF : (3 * D_MODEL);
  int cols1 = D_MODEL;
  int rows2 = D_MODEL;
  int cols2 = is_ffn ? D_FF : D_MODEL;
  size_t expected_bytes = (2 * D_MODEL * sizeof(float)) + (rows1 * sizeof(float)) + (rows1 * cols1 * sizeof(int8_t)) + (rows1 * sizeof(float)) + (rows2 * sizeof(float)) + (rows2 * cols2 * sizeof(int8_t)) + (rows2 * sizeof(float));

  if (file.size() < expected_bytes) {
    file.close();
    return false;
  }

  file.read((uint8_t*)g_layer.ln_w, D_MODEL * sizeof(float));
  file.read((uint8_t*)g_layer.ln_b, D_MODEL * sizeof(float));

  file.read((uint8_t*)g_layer.scale_w1, rows1 * sizeof(float));
  file.read((uint8_t*)g_layer.w1, rows1 * cols1 * sizeof(int8_t));
  file.read((uint8_t*)g_layer.b1, rows1 * sizeof(float));

  file.read((uint8_t*)g_layer.scale_w2, rows2 * sizeof(float));
  file.read((uint8_t*)g_layer.w2, rows2 * cols2 * sizeof(int8_t));
  file.read((uint8_t*)g_layer.b2, rows2 * sizeof(float));

  file.close();
  return true;
}

void positionalEncoding(float* outState, const float* inEmbedding, int seq_len) {
  for (int pos = 0; pos < seq_len; pos++) {
    for (int i = 0; i < D_MODEL; i++) {
      int pair_i = i - (i % 2);
      float div_term = expf((float)pair_i * (-logf(10000.0f) / (float)D_MODEL));
      float pe = (i % 2 == 0) ? sinf((float)pos * div_term) : cosf((float)pos * div_term);
      outState[pos * D_MODEL + i] = inEmbedding[pos * D_MODEL + i] + pe;
    }
  }
}

inline float gelu(float x) {
  return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

void layerNorm(float* out, const float* in, const float* gamma, const float* beta, int seq_len, int d_model) {
  for (int s = 0; s < seq_len; s++) {
    const float* x = in + s * d_model;
    float* y = out + s * d_model;
    float mean = 0.0f;
    for (int i = 0; i < d_model; i++) mean += x[i];
    mean /= d_model;
    float var = 0.0f;
    for (int i = 0; i < d_model; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= d_model;
    float invStd = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < d_model; i++) {
      y[i] = ((x[i] - mean) * invStd) * gamma[i] + beta[i];
    }
  }
}

void softmax(float* x, int size) {
  float max_val = x[0];
  for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  for (int i = 0; i < size; i++) x[i] /= sum;
}

void multiHeadAttention(float* state, const float* in_q, const float* in_kv, bool is_causal, int k_len, int active_seq_len) {
  const int8_t* w_q = g_layer.w1;
  const int8_t* w_k = g_layer.w1 + (D_MODEL * D_MODEL);
  const int8_t* w_v = g_layer.w1 + (2 * D_MODEL * D_MODEL);
  const float* b_q  = g_layer.b1;
  const float* b_k  = g_layer.b1 + D_MODEL;
  const float* b_v  = g_layer.b1 + (2 * D_MODEL);

  for (int s = 0; s < active_seq_len; s++) {
    for (int d = 0; d < D_MODEL; d++) {
      float sum = 0.0f;
      int row_offset = d * D_MODEL;
      for (int i = 0; i < D_MODEL; i++) {
        sum += in_q[s * D_MODEL + i] * (float)w_q[row_offset + i];
      }
      ws_Q[s * D_MODEL + d] = b_q[d] + (sum * g_layer.scale_w1[d]);
    }
  }
  
  for (int s = 0; s < k_len; s++) {
    for (int d = 0; d < D_MODEL; d++) {
      float sumk = 0.0f, sumv = 0.0f;
      int row_offset = d * D_MODEL;
      for (int i = 0; i < D_MODEL; i++) {
        sumk += in_kv[s * D_MODEL + i] * (float)w_k[row_offset + i];
        sumv += in_kv[s * D_MODEL + i] * (float)w_v[row_offset + i];
      }
      ws_K[s * D_MODEL + d] = b_k[d] + (sumk * g_layer.scale_w1[D_MODEL + d]);
      ws_V[s * D_MODEL + d] = b_v[d] + (sumv * g_layer.scale_w1[2 * D_MODEL + d]);
    }
  }

  float scale = 1.0f / sqrtf((float)HEAD_DIM);
  for (int h = 0; h < N_HEADS; h++) {
    for (int q_idx = 0; q_idx < active_seq_len; q_idx++) {
      for (int k_idx = 0; k_idx < k_len; k_idx++) {
        if (is_causal && k_idx > q_idx) {
          ws_scores[k_idx] = -1e9f;
          continue;
        }
        float score = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
          score += ws_Q[q_idx * D_MODEL + h * HEAD_DIM + d] *
                   ws_K[k_idx * D_MODEL + h * HEAD_DIM + d];
        }
        ws_scores[k_idx] = score * scale;
      }
      softmax(ws_scores, k_len);
      for (int d = 0; d < HEAD_DIM; d++) {
        float val = 0.0f;
        for (int k_idx = 0; k_idx < k_len; k_idx++) {
          val += ws_scores[k_idx] * ws_V[k_idx * D_MODEL + h * HEAD_DIM + d];
        }
        ws_attn_out[q_idx * D_MODEL + h * HEAD_DIM + d] = val;
      }
    }
  }

  const int8_t* proj_w = g_layer.w2;
  const float* proj_b  = g_layer.b2;
  for (int s = 0; s < active_seq_len; s++) {
    for (int d = 0; d < D_MODEL; d++) {
      float sum = 0.0f;
      int row_offset = d * D_MODEL;
      for (int i = 0; i < D_MODEL; i++) {
        sum += ws_attn_out[s * D_MODEL + i] * (float)proj_w[row_offset + i];
      }
      state[s * D_MODEL + d] += proj_b[d] + (sum * g_layer.scale_w2[d]);
    }
  }
}

void feedForwardFixed(float* state, const float* input, int active_seq_len) {
  const int8_t* ff1_w = g_layer.w1;
  const float*  ff1_b = g_layer.b1;
  const int8_t* ff2_w = g_layer.w2;
  const float*  ff2_b = g_layer.b2;

  for (int s = 0; s < active_seq_len; s++) {
    for (int f = 0; f < D_FF; f++) {
      float sum = 0.0f;
      int row_offset = f * D_MODEL;
      for (int d = 0; d < D_MODEL; d++) {
        sum += input[s * D_MODEL + d] * (float)ff1_w[row_offset + d];
      }
      ws_ffn_h[f] = gelu(ff1_b[f] + (sum * g_layer.scale_w1[f]));
    }
    for (int d = 0; d < D_MODEL; d++) {
      float sum = 0.0f;
      int row_offset = d * D_FF;
      for (int f = 0; f < D_FF; f++) {
        sum += ws_ffn_h[f] * (float)ff2_w[row_offset + f];
      }
      state[s * D_MODEL + d] += ff2_b[d] + (sum * g_layer.scale_w2[d]);
    }
  }
}

void generate(const char* prompt) {
  Serial.printf("\nUser : %s\n", prompt);
  if (!emb_weights || !proj_weights || !proj_bias) return;

  int token_ids[SEQ_LEN];
  for (int i = 0; i < SEQ_LEN; i++) token_ids[i] = PAD_ID;
  int count = 0;
  char prompt_copy[256];
  strncpy(prompt_copy, prompt, sizeof(prompt_copy) - 1);
  prompt_copy[sizeof(prompt_copy) - 1] = '\0';
  char* token = strtok(prompt_copy, " ");
  while (token != NULL && count < SEQ_LEN) {
    toLowerInPlace(token);
    token_ids[count++] = getWordId(token);
    token = strtok(NULL, " ");
  }

  memset(enc_state, 0, SEQ_LEN * D_MODEL * sizeof(float));
  memset(ws_raw_enc_tokens, 0, SEQ_LEN * D_MODEL * sizeof(float));

  float sqrt_d = sqrtf((float)D_MODEL);
  for (int s = 0; s < SEQ_LEN; s++) {
    int id = token_ids[s];
    if (id < 0 || id >= vocab_size) id = UNK_ID;
    for (int d = 0; d < D_MODEL; d++) {
      ws_raw_enc_tokens[s * D_MODEL + d] = ((float)emb_weights[id * D_MODEL + d] * emb_scales[id]) * sqrt_d;
    }
  }
  positionalEncoding(enc_state, ws_raw_enc_tokens, SEQ_LEN);

  for (int l = 0; l < N_LAYERS; l++) {
    char fname[64];
    
    snprintf(fname, sizeof(fname), "enc_%d_attn.bin", l);
    if (fetchAndParseLayer(fname, false)) {
      layerNorm(ws_norm_state, enc_state, g_layer.ln_w, g_layer.ln_b, SEQ_LEN, D_MODEL);
      memcpy(ws_sub_out, enc_state, SEQ_LEN * D_MODEL * sizeof(float));
      multiHeadAttention(ws_sub_out, ws_norm_state, ws_norm_state, false, SEQ_LEN, SEQ_LEN);
      memcpy(enc_state, ws_sub_out, SEQ_LEN * D_MODEL * sizeof(float));
    }

    snprintf(fname, sizeof(fname), "enc_%d_ffn.bin", l);
    if (fetchAndParseLayer(fname, true)) {
      layerNorm(ws_norm_state, enc_state, g_layer.ln_w, g_layer.ln_b, SEQ_LEN, D_MODEL);
      memcpy(ws_sub_out, enc_state, SEQ_LEN * D_MODEL * sizeof(float));
      feedForwardFixed(ws_sub_out, ws_norm_state, SEQ_LEN);
      memcpy(enc_state, ws_sub_out, SEQ_LEN * D_MODEL * sizeof(float));
    }
  }

  memcpy(enc_mem, enc_state, SEQ_LEN * D_MODEL * sizeof(float));

  memset(raw_dec_tokens, 0, SEQ_LEN * D_MODEL * sizeof(float));
  for (int d = 0; d < D_MODEL; d++) {
    raw_dec_tokens[d] = ((float)emb_weights[BOS_ID * D_MODEL + d] * emb_scales[BOS_ID]) * sqrt_d;
  }

  Serial.print("ESP: ");
  for (int token_step = 0; token_step < SEQ_LEN - 1; token_step++) {
    int active_len = token_step + 1;
    memset(dec_state, 0, active_len * D_MODEL * sizeof(float));
    positionalEncoding(dec_state, raw_dec_tokens, active_len);

    for (int l = 0; l < N_LAYERS; l++) {
      char fname[64];

      snprintf(fname, sizeof(fname), "dec_%d_self_attn.bin", l);
      if (fetchAndParseLayer(fname, false)) {
        layerNorm(ws_norm_state, dec_state, g_layer.ln_w, g_layer.ln_b, active_len, D_MODEL);
        memcpy(ws_sub_out, dec_state, active_len * D_MODEL * sizeof(float));
        multiHeadAttention(ws_sub_out, ws_norm_state, ws_norm_state, true, active_len, active_len);
        memcpy(dec_state, ws_sub_out, active_len * D_MODEL * sizeof(float));
      }

      snprintf(fname, sizeof(fname), "dec_%d_cross_attn.bin", l);
      if (fetchAndParseLayer(fname, false)) {
        layerNorm(ws_norm_state, dec_state, g_layer.ln_w, g_layer.ln_b, active_len, D_MODEL);
        memcpy(ws_sub_out, dec_state, active_len * D_MODEL * sizeof(float));
        multiHeadAttention(ws_sub_out, ws_norm_state, enc_mem, false, SEQ_LEN, active_len);
        memcpy(dec_state, ws_sub_out, active_len * D_MODEL * sizeof(float));
      }

      snprintf(fname, sizeof(fname), "dec_%d_ffn.bin", l);
      if (fetchAndParseLayer(fname, true)) {
        layerNorm(ws_norm_state, dec_state, g_layer.ln_w, g_layer.ln_b, active_len, D_MODEL);
        memcpy(ws_sub_out, dec_state, active_len * D_MODEL * sizeof(float));
        feedForwardFixed(ws_sub_out, ws_norm_state, active_len);
        memcpy(dec_state, ws_sub_out, active_len * D_MODEL * sizeof(float));
      }
    }

    float* last_state = dec_state + (token_step * D_MODEL);
    for (int v = 0; v < vocab_size; v++) {
      float sum = 0.0f;
      int row_offset = v * D_MODEL;
      for (int d = 0; d < D_MODEL; d++) {
        sum += last_state[d] * (float)proj_weights[row_offset + d];
      }
      logits_buf[v] = proj_bias[v] + (sum * proj_scales[v]);
    }

    int best_token = 0;
    float max_val = -1e30f;
    for (int v = 0; v < vocab_size; v++) {
      if (logits_buf[v] > max_val) {
        max_val = logits_buf[v];
        best_token = v;
      }
    }
    if (best_token == EOS_ID || best_token == PAD_ID) break;

    Serial.printf("%s ", vocab_words[best_token]);

    if (token_step + 1 < SEQ_LEN) {
      for (int d = 0; d < D_MODEL; d++) {
        raw_dec_tokens[(token_step + 1) * D_MODEL + d] =
          ((float)emb_weights[best_token * D_MODEL + d] * emb_scales[best_token]) * sqrt_d;
      }
    }
  }
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Starting...");
  
  if (!psramFound()) {
    Serial.println("PSRAM NOT FOUND");
    while (1) delay(1000);
  }

  if (!SD.begin(SD_CS)) {
    Serial.println("SD Card Failed");
    while (1) delay(1000);
  }

  if (!initLayerBuffer()) {
    Serial.println("Failed to allocate static layer buffer in PSRAM");
    while (1) delay(1000);
  }

  if (!loadVocabulary()) {
    Serial.println("Vocabulary Load Failed");
    while (1) delay(1000);
  }

  enc_state = (float*)heap_caps_malloc(SEQ_LEN * D_MODEL * sizeof(float), MALLOC_CAP_SPIRAM);
  dec_state = (float*)heap_caps_malloc(SEQ_LEN * D_MODEL * sizeof(float), MALLOC_CAP_SPIRAM);
  enc_mem = (float*)heap_caps_malloc(SEQ_LEN * D_MODEL * sizeof(float), MALLOC_CAP_SPIRAM);
  raw_dec_tokens = (float*)heap_caps_malloc(SEQ_LEN * D_MODEL * sizeof(float), MALLOC_CAP_SPIRAM);
  logits_buf = (float*)heap_caps_malloc(vocab_size * sizeof(float), MALLOC_CAP_SPIRAM);
  ws_raw_enc_tokens = (float*)heap_caps_malloc(SEQ_LEN * D_MODEL * sizeof(float), MALLOC_CAP_SPIRAM);
  ws_norm_state = (float*)heap_caps_malloc(SEQ_LEN * D_MODEL * sizeof(float), MALLOC_CAP_SPIRAM);
  ws_sub_out = (float*)heap_caps_malloc(SEQ_LEN * D_MODEL * sizeof(float), MALLOC_CAP_SPIRAM);
  ws_Q = (float*)heap_caps_malloc(SEQ_LEN * D_MODEL * sizeof(float), MALLOC_CAP_SPIRAM);
  ws_K = (float*)heap_caps_malloc(SEQ_LEN * D_MODEL * sizeof(float), MALLOC_CAP_SPIRAM);
  ws_V = (float*)heap_caps_malloc(SEQ_LEN * D_MODEL * sizeof(float), MALLOC_CAP_SPIRAM);
  ws_attn_out = (float*)heap_caps_malloc(SEQ_LEN * D_MODEL * sizeof(float), MALLOC_CAP_SPIRAM);
  ws_scores = (float*)heap_caps_malloc(SEQ_LEN * sizeof(float), MALLOC_CAP_SPIRAM);
  ws_ffn_h = (float*)heap_caps_malloc(D_FF * sizeof(float), MALLOC_CAP_SPIRAM);

  loadEmbeddingsOnly();
  Serial.println("Ready!");
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) {
      generate(input.c_str());
    }
  }
}
