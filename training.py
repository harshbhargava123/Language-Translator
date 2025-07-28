import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import logging
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import os
import torch.amp
import pickle
import nltk

# Define paths for RunPod
DATASET_PATH = '/workspace/Europarl_Data'
OUTPUT_PATH = '/workspace/Outputs'

# Ensure output directory exists
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Configure logging
logging.basicConfig(filename=os.path.join(OUTPUT_PATH, 'train.log'), level=logging.INFO, format='%(asctime)s - %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
logging.info(f"Using device: {device}")

# Install required libraries
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
except ImportError:
    os.system('pip install nltk')
    nltk.download('punkt')
    nltk.download('punkt_tab')

try:
    import matplotlib
except ImportError:
    os.system('pip install matplotlib')
    import matplotlib

# Custom GRU Cell
class CustomGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), dim=1)
        input_gate = torch.sigmoid(self.input_gate(combined))
        forget_gate = torch.sigmoid(self.forget_gate(combined))
        cell_gate = torch.tanh(self.cell_gate(combined))
        output_gate = torch.sigmoid(self.output_gate(combined))
        new_cell_state = forget_gate * hidden + input_gate * cell_gate
        new_hidden = output_gate * torch.tanh(new_cell_state)
        return new_hidden, new_cell_state

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.gru(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden

# Attention
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        self.v.data.normal_(mean=0, std=1./self.v.size(0)**0.5)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)
        return torch.softmax(attention, dim=1)

# Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, dropout, attention):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = CustomGRUCell(emb_dim + hidden_dim * 2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim * 3 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        embedded = embedded.squeeze(1)
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.squeeze(1)
        gru_input = torch.cat((embedded, weighted), dim=1)
        hidden, _ = self.gru(gru_input, hidden)
        embedded = embedded.squeeze(1)
        weighted = weighted.squeeze(1)
        prediction = self.fc_out(torch.cat((embedded, weighted, hidden), dim=1))
        return prediction, hidden, a.squeeze(1)

# Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs

# Data Loading
def load_data(dataset_path, max_sentences=5000):
    english_file = os.path.join(dataset_path, 'europarl-v7.fr-en.en')
    french_file = os.path.join(dataset_path, 'europarl-v7.fr-en.fr')
    english_sentences = []
    french_sentences = []
    with open(english_file, 'r', encoding='utf-8') as f:
        for line in f:
            if len(english_sentences) >= max_sentences:
                break
            line = line.strip()
            if line:
                english_sentences.append(line)
    with open(french_file, 'r', encoding='utf-8') as f:
        for line in f:
            if len(french_sentences) >= max_sentences:
                break
            line = line.strip()
            if line:
                french_sentences.append(line)
    data = list(zip(english_sentences, french_sentences))
    return data

# Vocabulary Building
def build_vocab(sentences, max_size=30000):
    word_counts = {}
    for sentence in sentences:
        words = nltk.word_tokenize(sentence.lower())
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
    vocab = ['<pad>', '<sos>', '<eos>', '<unk>'] + sorted(word_counts, key=word_counts.get, reverse=True)
    vocab = vocab[:max_size]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    return vocab, word_to_idx

# Sentence to Tensor
def sentence_to_tensor(sentence, vocab, word_to_idx, max_len):
    tokens = nltk.word_tokenize(sentence.lower())
    tokens = ['<sos>'] + tokens[:max_len-2] + ['<eos>']
    indices = [word_to_idx.get(token, word_to_idx['<unk>']) for token in tokens]
    while len(indices) < max_len:
        indices.append(word_to_idx['<pad>'])
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)

# Prepare Batch
def prepare_batch(data, batch_size, max_len):
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        src_batch = []
        trg_batch = []
        for eng, fr in batch:
            src_tensor = sentence_to_tensor(eng, english_vocab, english_word_to_idx, max_len)
            trg_tensor = sentence_to_tensor(fr, french_vocab, french_word_to_idx, max_len)
            src_batch.append(src_tensor)
            trg_batch.append(trg_tensor)
        src_batch = torch.cat(src_batch, dim=0).to(device)
        trg_batch = torch.cat(trg_batch, dim=0).to(device)
        yield src_batch, trg_batch

# BLEU Score Calculation
def calculate_bleu(model, data, src_vocab, trg_vocab, max_len, batch_size, beam_width=1):
    model.eval()
    bleu_scores = []
    smoothing = SmoothingFunction()
    sample_data = random.sample(data, min(500, len(data)))
    for src_sentence, trg_sentence in sample_data:
        src_tensor = sentence_to_tensor(src_sentence, src_vocab, english_word_to_idx, max_len).to(device)
        trg_tokens = nltk.word_tokenize(trg_sentence.lower())
        if beam_width > 1:
            pred_tokens = translate_sentence(model, src_sentence, src_vocab, trg_vocab, max_len=max_len, beam_width=beam_width)
        else:
            pred_tokens = translate_sentence(model, src_sentence, src_vocab, trg_vocab, max_len=max_len)
        pred_tokens = pred_tokens[1:-1]
        reference = [trg_tokens]
        candidate = pred_tokens
        score = sentence_bleu(reference, candidate, smoothing_function=smoothing.method1)
        bleu_scores.append(score)
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

# Translate Sentence
def translate_sentence(model, sentence, src_vocab, trg_vocab, max_len=50, beam_width=1, return_attention=False):
    model.eval()
    src_tensor = sentence_to_tensor(sentence, src_vocab, english_word_to_idx, max_len).to(device)
    src_tokens = nltk.word_tokenize(sentence.lower())[:max_len-2]
    src_tokens = ['<sos>'] + src_tokens + ['<eos>']
    if beam_width == 1:
        with torch.no_grad():
            encoder_outputs, hidden = model.encoder(src_tensor)
            input = torch.tensor([french_word_to_idx['<sos>']], dtype=torch.long).to(device)
            output_tokens = []
            attention_weights = []
            for _ in range(max_len):
                output, hidden, attn = model.decoder(input, hidden, encoder_outputs)
                input = output.argmax(1)
                output_tokens.append(input.item())
                attention_weights.append(attn.cpu().detach().numpy())
                if input.item() == french_word_to_idx['<eos>']:
                    break
            translated = [trg_vocab[idx] for idx in output_tokens]
            if return_attention:
                return translated, attention_weights, src_tokens
            return translated
    else:
        sequences = [[[french_word_to_idx['<sos>']], 0.0]]
        completed = []
        with torch.no_grad():
            encoder_outputs, hidden = model.encoder(src_tensor)
            for _ in range(max_len):
                new_sequences = []
                for seq, score in sequences:
                    if seq[-1] == french_word_to_idx['<eos>']:
                        completed.append([seq, score])
                        continue
                    input = torch.tensor([seq[-1]], dtype=torch.long).to(device)
                    output, next_hidden, _ = model.decoder(input, hidden, encoder_outputs)
                    probs, indices = output.topk(beam_width)
                    probs = probs.log().squeeze()
                    indices = indices.squeeze()
                    for i in range(beam_width):
                        new_seq = seq + [indices[i].item()]
                        new_score = score + probs[i].item()
                        new_sequences.append([new_seq, new_score])
                new_sequences.sort(key=lambda x: x[1], reverse=True)
                sequences = new_sequences[:beam_width]
                if not sequences:
                    break
            if not completed:
                completed = sequences
            best_seq = min(completed, key=lambda x: x[1])[0]
            translated = [trg_vocab[idx] for idx in best_seq]
            return translated

# Check Test Sentences
def check_test_sentences(model, test_data, src_vocab, trg_vocab, num_samples=5):
    model.eval()
    sample_data = random.sample(test_data, min(num_samples, len(test_data)))
    for i, (src_sentence, trg_sentence) in enumerate(sample_data):
        pred_sentence = translate_sentence(model, src_sentence, src_vocab, trg_vocab, beam_width=4)
        pred_sentence = pred_sentence[1:-1]
        print(f"Test Sample {i+1}:")
        print(f"English: {src_sentence}")
        logging.info(f"Test Sample {i+1}:")
        logging.info(f"English: {src_sentence}")
        reference = [nltk.word_tokenize(trg_sentence.lower())]
        candidate = pred_sentence
        smoothing = SmoothingFunction()
        bleu = sentence_bleu(reference, candidate, smoothing_function=smoothing.method1)
        print(f"BLEU Score: {bleu:.4f}\n")
        logging.info(f"BLEU Score: {bleu:.4f}\n")

# Adjust Learning Rate
def adjust_lr(optimizer, epoch, warmup_epochs, base_lr):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

# Load and preprocess data
data = load_data(DATASET_PATH, max_sentences=5000)
random.seed(42)
random.shuffle(data)
train_size = int(0.8 * len(data))  # 4,000 pairs
val_size = int(0.1 * len(data))    # 500 pairs
train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# Save split datasets
with open(os.path.join(OUTPUT_PATH, 'train_data.pkl'), 'wb') as f:
    pickle.dump(train_data, f)
with open(os.path.join(OUTPUT_PATH, 'val_data.pkl'), 'wb') as f:
    pickle.dump(val_data, f)
with open(os.path.join(OUTPUT_PATH, 'test_data.pkl'), 'wb') as f:
    pickle.dump(test_data, f)
print(f"Saved split datasets to {OUTPUT_PATH}/train_data.pkl, val_data.pkl, test_data.pkl")
logging.info(f"Saved split datasets to {OUTPUT_PATH}/train_data.pkl, val_data.pkl, test_data.pkl")

english_sentences = [eng for eng, _ in data]
french_sentences = [fr for _, fr in data]

english_vocab, english_word_to_idx = build_vocab(english_sentences)
french_vocab, french_word_to_idx = build_vocab(french_sentences)

# Save vocabularies
with open(os.path.join(OUTPUT_PATH, 'english_vocab.pkl'), 'wb') as f:
    pickle.dump(english_vocab, f)
with open(os.path.join(OUTPUT_PATH, 'english_word_to_idx.pkl'), 'wb') as f:
    pickle.dump(english_word_to_idx, f)
with open(os.path.join(OUTPUT_PATH, 'french_vocab.pkl'), 'wb') as f:
    pickle.dump(french_vocab, f)
with open(os.path.join(OUTPUT_PATH, 'french_word_to_idx.pkl'), 'wb') as f:
    pickle.dump(french_word_to_idx, f)
print(f"Saved vocabularies to {OUTPUT_PATH}/english_vocab.pkl, english_word_to_idx.pkl, french_vocab.pkl, french_word_to_idx.pkl")
logging.info(f"Saved vocabularies to {OUTPUT_PATH}/english_vocab.pkl, english_word_to_idx.pkl, french_vocab.pkl, french_word_to_idx.pkl")

# Model parameters
input_dim = len(english_vocab)
output_dim = len(french_vocab)
emb_dim = 256
hidden_dim = 512
dropout = 0.5
max_len = 50
batch_size = 16
base_lr = 0.0001
warmup_epochs = 100

# Initialize model
attn = Attention(hidden_dim)
enc = Encoder(input_dim, emb_dim, hidden_dim, dropout)
dec = Decoder(output_dim, emb_dim, hidden_dim, dropout, attn)
model = Seq2Seq(enc, dec, device).to(device)

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=base_lr)
criterion = nn.CrossEntropyLoss(ignore_index=french_word_to_idx['<pad>'])
scaler = torch.amp.GradScaler('cuda')
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# Training setup
train_data.sort(key=lambda x: len(x[0].split()))
epochs = 1000
best_bleu = 0
best_train_loss = float('inf')
total_training_time = 0

train_losses = []
val_losses = []
test_losses = []
bleu_scores = []
train_bleu_scores = []
bleu_epochs = []
teacher_forcing_ratios = []

# Start training from epoch 1
start_epoch = 0

# Training loop
model.train()
for epoch in range(start_epoch, epochs):
    epoch_start_time = time.time()
    optimizer = adjust_lr(optimizer, epoch, warmup_epochs, base_lr)
    total_loss = 0
    total_samples = 0

    teacher_forcing_ratio = max(0.8 - (epoch / 1500) * 0.72, 0.1)
    teacher_forcing_ratios.append(teacher_forcing_ratio)

    for src_batch, trg_batch in prepare_batch(train_data, batch_size, max_len):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            output = model(src_batch, trg_batch, teacher_forcing_ratio=teacher_forcing_ratio)
            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)
            trg = trg_batch[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * src_batch.size(0)
        total_samples += src_batch.size(0)

    avg_loss = total_loss / total_samples
    train_losses.append(avg_loss)

    # Validation and Test every 25 epochs
    if (epoch + 1) % 25 == 0:
        model.eval()
        val_loss = 0
        val_samples = 0
        test_loss = 0
        test_samples = 0
        with torch.no_grad():
            # Validation loss
            for src_batch, trg_batch in prepare_batch(val_data, batch_size, max_len):
                output = model(src_batch, trg_batch, teacher_forcing_ratio=0)
                output_dim = output.shape[-1]
                output = output[:, 1:].contiguous().view(-1, output_dim)
                trg = trg_batch[:, 1:].contiguous().view(-1)
                loss = criterion(output, trg)
                val_loss += loss.item() * src_batch.size(0)
                val_samples += src_batch.size(0)
            # Test loss
            for src_batch, trg_batch in prepare_batch(test_data, batch_size, max_len):
                output = model(src_batch, trg_batch, teacher_forcing_ratio=0)
                output_dim = output.shape[-1]
                output = output[:, 1:].contiguous().view(-1, output_dim)
                trg = trg_batch[:, 1:].contiguous().view(-1)
                loss = criterion(output, trg)
                test_loss += loss.item() * src_batch.size(0)
                test_samples += src_batch.size(0)
        val_loss = val_loss / val_samples
        test_loss = test_loss / test_samples
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        bleu_epochs.append(epoch + 1)

        train_bleu_score = calculate_bleu(model, train_data, english_vocab, french_vocab, max_len, batch_size, beam_width=4)
        bleu_score = calculate_bleu(model, val_data, english_vocab, french_vocab, max_len, batch_size, beam_width=4)
        bleu_scores.append(bleu_score)
        train_bleu_scores.append(train_bleu_score)

        scheduler.step(bleu_score)

        print(f'Epoch: {epoch+1}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}, Train BLEU: {train_bleu_score:.4f}, Val BLEU: {bleu_score:.4f}, Time: {time.time() - epoch_start_time:.2f}s')
        logging.info(f'Epoch: {epoch+1}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}, Train BLEU: {train_bleu_score:.4f}, Val BLEU: {bleu_score:.4f}, Time: {time.time() - epoch_start_time:.2f}s')
        print(f"Teacher Forcing Ratio: {teacher_forcing_ratio:.4f}")
        logging.info(f"Teacher Forcing Ratio: {teacher_forcing_ratio:.4f}")

        if bleu_score > best_bleu:
            best_bleu = bleu_score
            try:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1
                }, os.path.join(OUTPUT_PATH, 'best_model.pt'))
                print(f"Saved best model at epoch {epoch+1} with BLEU: {bleu_score:.4f}")
                logging.info(f"Saved best model at epoch {epoch+1} with BLEU: {bleu_score:.4f}")
            except RuntimeError as e:
                print(f"Failed to save best model at epoch {epoch+1}: {e}")
                logging.error(f"Failed to save best model at epoch {epoch+1}: {e}")

        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            try:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1
                }, os.path.join(OUTPUT_PATH, 'best_train_loss_model.pt'))
                print(f"Saved best train loss model at epoch {epoch+1} with Train Loss: {avg_loss:.4f}")
                logging.info(f"Saved best train loss model at epoch {epoch+1} with Train Loss: {avg_loss:.4f}")
            except RuntimeError as e:
                print(f"Failed to save best train loss model at epoch {epoch+1}: {e}")
                logging.error(f"Failed to save best train loss model at epoch {epoch+1}: {e}")

        check_test_sentences(model, test_data, english_vocab, french_vocab)
        model.train()

    else:
        print(f'Epoch: {epoch+1}, Train Loss: {avg_loss:.4f}, Time: {time.time() - epoch_start_time:.2f}s')
        logging.info(f'Epoch: {epoch+1}, Train Loss: {avg_loss:.4f}, Time: {time.time() - epoch_start_time:.2f}s')

    # Save checkpoint every 200 epochs
    if (epoch + 1) % 200 == 0:
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1
            }, os.path.join(OUTPUT_PATH, f'checkpoint_epoch_{epoch+1}.pt'))
            print(f"Saved checkpoint at epoch {epoch+1} to {os.path.join(OUTPUT_PATH, f'checkpoint_epoch_{epoch+1}.pt')}")
            logging.info(f"Saved checkpoint at epoch {epoch+1} to {os.path.join(OUTPUT_PATH, f'checkpoint_epoch_{epoch+1}.pt')}")
        except RuntimeError as e:
            print(f"Failed to save checkpoint at epoch {epoch+1}: {e}")
            logging.error(f"Failed to save checkpoint at epoch {epoch+1}: {e}")

        # Save loss and BLEU arrays every 200 epochs
        try:
            torch.save({
                'train_losses': train_losses,
                'val_losses': val_losses,
                'test_losses': test_losses,
                'bleu_epochs': bleu_epochs,
                'train_bleu_scores': train_bleu_scores,
                'val_bleu_scores': bleu_scores
            }, os.path.join(OUTPUT_PATH, 'loss_arrays.pt'))
            print(f"Saved loss and BLEU arrays at epoch {epoch+1} to {os.path.join(OUTPUT_PATH, 'loss_arrays.pt')}")
            logging.info(f"Saved loss and BLEU arrays at epoch {epoch+1} to {os.path.join(OUTPUT_PATH, 'loss_arrays.pt')}")
        except RuntimeError as e:
            print(f"Failed to save loss and BLEU arrays at epoch {epoch+1}: {e}")
            logging.error(f"Failed to save loss and BLEU arrays at epoch {epoch+1}: {e}")

    total_training_time += time.time() - epoch_start_time
    print(f"Progress: Completed {epoch+1}/{epochs} epochs, Total Time: {total_training_time / 3600:.2f} hours")
    logging.info(f"Progress: Completed {epoch+1}/{epochs} epochs, Total Time: {total_training_time / 3600:.2f} hours")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(bleu_epochs, val_losses, label='Validation Loss')
plt.plot(bleu_epochs, test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training, Validation, and Test Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_PATH, 'loss.png'))
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(bleu_epochs, bleu_scores, label='Validation BLEU')
plt.plot(bleu_epochs, train_bleu_scores, label='Train BLEU')
plt.xlabel('Epoch')
plt.ylabel('BLEU Score')
plt.title('BLEU Score Over Time')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_PATH, 'bleu.png'))
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(teacher_forcing_ratios) + 1), teacher_forcing_ratios, label='Teacher Forcing Ratio')
plt.xlabel('Epoch')
plt.ylabel('Teacher Forcing Ratio')
plt.title('Teacher Forcing Ratio Over Time')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_PATH, 'teacher_forcing.png'))
plt.close()

print(f"Training completed in {total_training_time / 3600:.2f} hours")
logging.info(f"Training completed in {total_training_time / 3600:.2f} hours")
print(f"Plots saved to {OUTPUT_PATH}/loss.png, bleu.png, teacher_forcing.png")
logging.info(f"Plots saved to {OUTPUT_PATH}/loss.png, bleu.png, teacher_forcing.png")