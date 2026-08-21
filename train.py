import os
import math
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
SEQ_LEN       = 32
D_MODEL       = 768
N_HEADS       = 12
N_LAYERS      = 16
D_FF          = 2048
BATCH_SIZE    = 4
EPOCHS        = 150
LEARNING_RATE = 3e-4
OUTPUT_DIR    = "weights"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAIRS = [
    ("hello", "hi there !"),
    ("hi", "hello ! how can i help ?"),
    ("hey", "hey ! what is on your mind ?"),
    ("how are you", "i am good , thanks !"),
    ("how is it going", "everything is running smoothly !"),
    ("good morning", "morning ! hope you have a great day ."),
    ("good afternoon", "good afternoon ! how is your day going ?"),
    ("good evening", "good evening ! hope you had a productive day ."),
    ("good night", "good night ! sleep well ."),
    ("bye", "goodbye ! see you later ."),
    ("see you", "see you soon ! take care ."),
    ("thanks", "you are very welcome !"),
    ("thank you", "happy to help !"),
    ("what is your name", "i am a tiny toy transformer ."),
    ("who made you", "i was created using pytorch and deployed on embedded hardware ."),
    ("favorite color", "i like binary green ."),
    ("tell me a joke", "why do programmers prefer dark mode ? because light attracts bugs !"),
    ("another joke", "there are 10 types of people : those who understand binary and those who do not ."),
    ("weather", "i live inside a computer , so it is always 20 degrees celsius ."),
    ("ping", "pong"),
    ("status", "all systems operational ."),
    ("what time is it", "time to write some code !"),
    ("are you smart", "i try my best with my tiny parameter budget !"),
    ("what is esp32", "a low cost wi fi and bluetooth microcontroller ."),
    ("what is arduino", "an open source electronics platform ."),
    ("what is raspberry pi", "a small single board computer ."),
    ("what is psram", "pseudo static random access memory used for extra storage ."),
    ("what is gpio", "general purpose input output pins for connecting sensors ."),
    ("what is flash memory", "non volatile memory used to store program code ."),
    ("what is uart", "universal asynchronous receiver transmitter for serial communication ."),
    ("what is i2c", "a two wire serial protocol for connecting low speed peripherals ."),
    ("what is spi", "a fast four wire synchronous serial communication protocol ."),
    ("what is pwm", "pulse width modulation used to control power to electronic devices ."),
    ("what is adc", "analog to digital converter that reads voltage signals ."),
    ("what is dac", "digital to analog converter that produces dynamic analog voltages ."),
    ("what is an led", "a light emitting diode that produces illumination ."),
    ("what is a sensor", "a device that detects physical inputs like temperature or motion ."),
    ("what is a relay", "an electrically operated switch used to control high voltage ."),
    ("what is dynamic power", "the energy consumed by switching transistors in an integrated circuit ."),
    ("what is ram", "random access memory used for temporary storage ."),
    ("what is cpu", "the central processing unit of a computer ."),
    ("what is gpu", "a graphics processing unit optimized for parallel compute ."),
    ("what is npu", "a neural processing unit designed for AI matrix operations ."),
    ("what is an operating system", "software that manages hardware resources and execution ."),
    ("what is Linux", "an open source Unix like operating system kernel ."),
    ("what is a compiler", "a tool that translates human code into machine instructions ."),
    ("what is a pointer", "a variable that stores the memory address of another value ."),
    ("what is a stack", "a first in last out data structure ."),
    ("what is a queue", "a first in first out data structure ."),
    ("what is binary", "a base two number system consisting of zeroes and ones ."),
    ("what is hex", "a base sixteen numbering system used in low level programming ."),
    ("what is an array", "a contiguous block of memory holding identical data types ."),
    ("what is cache", "a small high speed memory buffer near the processing core ."),
    ("what is latency", "the delay time between sending a request and receiving a response ."),
    ("what is python", "a popular high level programming language ."),
    ("what is pytorch", "an open source machine learning library ."),
    ("what is tensorflow", "an end to end open source platform for machine learning ."),
    ("what is a transformer", "a neural network architecture based on self attention mechanisms ."),
    ("what is self attention", "a mechanism mapping context relationships across sequence tokens ."),
    ("what is positional encoding", "vectors added to embeddings to preserve token sequence order ."),
    ("what is quantization", "converting high precision floating point numbers to smaller integer formats ."),
    ("what is int8", "an eight bit integer representation used to save memory ."),
    ("what is float32", "a thirty two bit single precision floating point format ."),
    ("what is loss", "a mathematical score measuring model prediction error ."),
    ("what is an epoch", "one complete pass through the entire training dataset ."),
    ("what is batch size", "the number of training samples processed before updating weights ."),
    ("what is learning rate", "a hyperparameter controlling gradient step size during optimization ."),
    ("what is soft max", "a mathematical function converting raw logits into probabilities ."),
    ("what is 1 plus 1", "one plus one equals two ."),
    ("what is 2 plus 2", "two plus two equals four ."),
    ("what is 3 plus 3", "three plus three equals six ."),
    ("what is 5 plus 5", "five plus five equals ten ."),
    ("what is 10 times 10", "ten times ten equals one hundred ."),
    ("what is 0 times 5", "zero times any number equals zero ."),
    ("what is the capital of france", "paris is the capital ."),
    ("what is the capital of japan", "tokyo is the capital ."),
    ("what is the capital of germany", "berlin is the capital ."),
    ("what is the capital of italy", "rome is the capital ."),
    ("what is the capital of india", "new delhi is the capital ."),
    ("what is the largest ocean", "the pacific ocean is the largest ."),
    ("what planet is closest to the sun", "mercury is the closest planet ."),
    ("what is water made of", "water consists of two hydrogen atoms and one oxygen atom ."),
    ("turn on light", "switching power channel one to active high ."),
    ("turn off light", "switching power channel one to active low ."),
    ("check wifi", "connected to local network with strong signal ."),
    ("read temperature", "ambient temperature sensor reads 24 degrees celsius ."),
    ("system reboot", "restarting system services now ."),
    ("memory usage", "psram consumption is within nominal operating parameters ."),
    ("who are you", "an embedded transformer inference engine running on an esp32 ."),
    ("can you help me", "yes ! ask me a question about hardware , software , or math ."),
    ("how fast are you", "i run matrix multiplication layers directly in low level C code ."),
    ("good job", "thank you ! keeping neural network execution fast and lightweight ."),
]

print("Loaded dataset...")
def tokenize(text):
    return text.lower().strip().split()

def build_vocab(pairs):
    words = set()
    for src, tgt in pairs:
        words.update(tokenize(src))
        words.update(tokenize(tgt))
    vocab = SPECIALS + sorted(words)
    word2id = {w:i for i,w in enumerate(vocab)}
    return vocab,word2id

VOCAB, WORD2ID = build_vocab(PAIRS)
VOCAB_SIZE = len(VOCAB)

def encode(text):
    return [WORD2ID.get(w, UNK_ID) for w in tokenize(text)]

class WordDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_str, tgt_str = self.pairs[idx]
        src = encode(src_str)
        tgt = encode(tgt_str)
        src     = (src + [PAD_ID] * SEQ_LEN)[:SEQ_LEN]
        tgt_in  = ([BOS_ID] + tgt + [PAD_ID] * SEQ_LEN)[:SEQ_LEN]
        tgt_out = (tgt + [EOS_ID] + [PAD_ID] * SEQ_LEN)[:SEQ_LEN]
        return (torch.tensor(src, dtype=torch.long),torch.tensor(tgt_in, dtype=torch.long),torch.tensor(tgt_out, dtype=torch.long),)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]

def gelu_tanh(x):
    return F.gelu(x, approximate='tanh')

class EncoderBlock(nn.Module):
    def __init__(self, d_model, h, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        norm_x2 = self.ln2(x)
        x = x + self.ff2(gelu_tanh(self.ff1(norm_x2)))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, h, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, h, batch_first=True)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

    def forward(self, x, enc_out):
        seq_len = x.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        norm_x = self.ln1(x)
        s_out, _ = self.self_attn(norm_x, norm_x, norm_x, attn_mask=causal_mask)
        x = x + s_out
        norm_x2 = self.ln2(x)
        c_out, _ = self.cross_attn(norm_x2, enc_out, enc_out)
        x = x + c_out
        norm_x3 = self.ln3(x)
        x = x + self.ff2(gelu_tanh(self.ff1(norm_x3)))
        return x

class TinySeq2Seq(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, D_MODEL, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(D_MODEL, SEQ_LEN)
        self.enc_layers = nn.ModuleList([EncoderBlock(D_MODEL, N_HEADS, D_FF) for _ in range(N_LAYERS)])
        self.dec_layers = nn.ModuleList([DecoderBlock(D_MODEL, N_HEADS, D_FF) for _ in range(N_LAYERS)])
        self.proj = nn.Linear(D_MODEL, vocab_size)

    def encode(self, src):
        x = self.pos(self.emb(src) * math.sqrt(D_MODEL))
        for layer in self.enc_layers:
            x = layer(x)
        return x

    def decode(self, tgt, enc_out):
        x = self.pos(self.emb(tgt) * math.sqrt(D_MODEL))
        for layer in self.dec_layers:
            x = layer(x, enc_out)
        return self.proj(x)

    def forward(self, src, tgt_in):
        enc_out = self.encode(src)
        return self.decode(tgt_in, enc_out)
def dump_tensor(f, tensor):
    data = tensor.detach().cpu().numpy().astype(np.float32)

    if data.ndim == 1:
        f.write(data.tobytes())
        return
    n_rows, n_cols = data.shape
    scales = np.zeros(n_rows, dtype=np.float32)
    quantized = np.zeros_like(data, dtype=np.int8)

    for r in range(n_rows):
        row = data[r]
        std_dev = np.std(row)
        mean_val = np.mean(row)
        clip_val = max(np.max(np.abs(row)), 1e-5)
        if std_dev > 0:
            clip_val = min(clip_val, np.abs(mean_val) + 3.5 * std_dev)
        scale = clip_val / 127.0
        scales[r] = scale
        quantized[r] = np.clip(np.round(row / scale), -128, 127).astype(np.int8)

    f.write(scales.tobytes())
    f.write(quantized.tobytes())

def export_vocab(vocab, out_path):
    with open(out_path, "wb") as f:
        f.write(struct.pack('<I', len(vocab)))
        for w in vocab:
            b = w.encode("utf-8")
            if len(b) > 255:
                b = b[:255]
            f.write(struct.pack('B', len(b)))
            f.write(b)

def export_binaries(model, vocab):
    model.eval()
    print("Exporting...")
    with open(os.path.join(OUTPUT_DIR, "embeddings.bin"), "wb") as f:
        dump_tensor(f, model.emb.weight)
        dump_tensor(f, model.proj.weight)
        dump_tensor(f, model.proj.bias)

    for idx, layer in enumerate(model.enc_layers):
        with open(os.path.join(OUTPUT_DIR, f"enc_{idx}_attn.bin"), "wb") as f:
            dump_tensor(f,layer.ln1.weight)
            dump_tensor(f,layer.ln1.bias)
            dump_tensor(f,layer.attn.in_proj_weight)
            dump_tensor(f,layer.attn.in_proj_bias)
            dump_tensor(f,layer.attn.out_proj.weight)
            dump_tensor(f,layer.attn.out_proj.bias)
        with open(os.path.join(OUTPUT_DIR, f"enc_{idx}_ffn.bin"), "wb") as f:
            dump_tensor(f,layer.ln2.weight)
            dump_tensor(f,layer.ln2.bias)
            dump_tensor(f,layer.ff1.weight)
            dump_tensor(f,layer.ff1.bias)
            dump_tensor(f,layer.ff2.weight)
            dump_tensor(f,layer.ff2.bias)

    for idx, layer in enumerate(model.dec_layers):
        with open(os.path.join(OUTPUT_DIR, f"dec_{idx}_self_attn.bin"), "wb") as f:
            dump_tensor(f,layer.ln1.weight)
            dump_tensor(f,layer.ln1.bias)
            dump_tensor(f,layer.self_attn.in_proj_weight)
            dump_tensor(f,layer.self_attn.in_proj_bias)
            dump_tensor(f,layer.self_attn.out_proj.weight)
            dump_tensor(f,layer.self_attn.out_proj.bias)
        with open(os.path.join(OUTPUT_DIR, f"dec_{idx}_cross_attn.bin"), "wb") as f:
            dump_tensor(f,layer.ln2.weight)
            dump_tensor(f,layer.ln2.bias)
            dump_tensor(f,layer.cross_attn.in_proj_weight)
            dump_tensor(f,layer.cross_attn.in_proj_bias)
            dump_tensor(f,layer.cross_attn.out_proj.weight)
            dump_tensor(f,layer.cross_attn.out_proj.bias)
        with open(os.path.join(OUTPUT_DIR, f"dec_{idx}_ffn.bin"), "wb") as f:
            dump_tensor(f,layer.ln3.weight)
            dump_tensor(f,layer.ln3.bias)
            dump_tensor(f,layer.ff1.weight)
            dump_tensor(f,layer.ff1.bias)
            dump_tensor(f,layer.ff2.weight)
            dump_tensor(f,layer.ff2.bias)

    export_vocab(vocab, os.path.join(OUTPUT_DIR, "vocab.bin"))
    print("Exported...")

def train():
    dataset = WordDataset(PAIRS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = TinySeq2Seq(VOCAB_SIZE).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params / 1e6:.2f} Million")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    model.train()
    print("Starting training loop...")
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for src, tgt_in, tgt_out in loader:
            src, tgt_in, tgt_out = src.to(DEVICE), tgt_in.to(DEVICE), tgt_out.to(DEVICE)
            optimizer.zero_grad()
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            total_loss += loss.item()

        if epoch % 50 == 0 or epoch == 1:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch:4d}/{EPOCHS}  loss={avg_loss:.4f}")
    return model

if __name__ == "__main__":
    model = train()
    export_binaries(model, VOCAB)