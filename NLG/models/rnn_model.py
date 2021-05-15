import torch
from torch import nn
from torch.nn import functional as F
import random


class RNNEncoder(nn.Module):
    def __init__(self, src_size, embed_size, hidden_size, is_bidirect=True, dropout_rate=0.1):
        super(RNNEncoder, self).__init__()
        self.input_size = src_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.is_bidirect = is_bidirect

        self.embedder = nn.Embedding(src_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=is_bidirect)
        self.reduce_h = nn.Linear(hidden_size * 2, hidden_size)
        self.reduce_c = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def init_weight(self):
        """
        Initializes weight matrices using Xavier initialization
        :return:
        """
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        if self.is_bidirect:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        if self.is_bidirect:
            nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)

    def forward(self, src_seqs, lens):
        # B x S -> B x S x E
        embedded = self.dropout(self.embedder(src_seqs))

        # encoder_outputs: B x S x E -> B x S x H, encoder_hc: (h,c), h: n_layers*n_direction x B x H
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lens.to('cpu'), batch_first=True)
        packed_outputs, encoder_hc = self.rnn(packed_embedded)
        encoder_outputs, lens = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        if self.is_bidirect:
            h, c = encoder_hc
            h_cat, c_cat = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
            reduced_h = torch.tanh(self.reduce_h(h_cat))
            reduced_c = torch.tanh(self.reduce_c(c_cat))
            encoder_hc = (reduced_h.unsqueeze(0), reduced_c.unsqueeze(0))

        return encoder_outputs, encoder_hc


class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'concat':
            self.fc1 = nn.Linear(hidden_size * 3, hidden_size)
            self.fc2 = nn.Linear(hidden_size, 1, bias=False)
        else:
            self.fc = nn.Linear(hidden_size, hidden_size * 2)

    def forward(self, decoder_h, encoder_outputs, attn_mask):
        # energy: B x S
        energy = self.score(decoder_h, encoder_outputs)
        energy = energy.masked_fill(attn_mask == 0, -1e10)
        attn = F.softmax(energy, dim=1)
        return attn

    def score(self, decoder_h, encoder_outputs):
        if self.method == 'concat':
            seq_len = encoder_outputs.size(1)
            # (1 x B x H -> B x 1 x H) -> B x S x H
            h_repeated = decoder_h.permute(1, 0, 2).repeat(1, seq_len, 1)
            # energy: B x S x 3H -> B x S x H
            energy = torch.tanh(self.fc1(torch.cat((h_repeated, encoder_outputs), dim=2)))
            # attn_energy: B x S x H -> B x S x 1 -> B x S
            energy = self.fc2(energy).squeeze(2)
            return energy
        else:
            # aligned_h: 1 x B x H -> 1 x B x 2H
            aligned_h = self.fc(decoder_h)
            # energy: B x S, encoder_outputs: B x S x 2H, decoder_h: 1 x B x 2H -> B x 2H x 1
            energy = torch.matmul(encoder_outputs, aligned_h.permute(1, 2, 0)).squeeze(2)
            return energy


class RNNDecoder(nn.Module):
    def __init__(self, trg_size, embed_size, hidden_size, attn_method='luong', is_bidirect=False, dropout_rate=0.1):
        super(RNNDecoder, self).__init__()
        self.input_size = trg_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.attn_method = attn_method
        self.is_bidirect = is_bidirect

        self.embedder = nn.Embedding(trg_size, embed_size)
        self.rnn = nn.LSTM(embed_size + hidden_size * 2, hidden_size, batch_first=True, bidirectional=is_bidirect)
        self.attn = Attention('concat', hidden_size)
        self.out = nn.Linear(hidden_size * 3, trg_size)
        self.dropout = nn.Dropout(dropout_rate)

    def init_weight(self):
        """
        Initializes weight matrices using Xavier initialization
        :return:
        """
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        if self.is_bidirect:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        if self.is_bidirect:
            nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)

    def forward(self, current_input, last_context, decoder_hc, encoder_outputs, mask):
        # embedded: B x 1 -> B x 1 x E
        word_embedded = self.dropout(self.embedder(current_input))
        if self.attn_method == 'luong':
            # rnn_input: (B x 1 x E) + (B x 1 x 2H) -> (B x 1 x (E + 2H))
            rnn_input = torch.cat((word_embedded, last_context), dim=2)
            # rnn_output: B x 1 x H, (decoder_h, decoder_c): 1 x B x H
            rnn_output, decoder_hc = self.rnn(rnn_input, decoder_hc)
            # attn_weights: B x S
            attn_weights = self.attn(decoder_hc[0], encoder_outputs, mask)
            # context: (B x S -> B x 1 x S) @ (B x S x 2H) -> B x 1 x 2H
            context = torch.matmul(attn_weights.unsqueeze(1), encoder_outputs)

            # output: B x 3H -> B x T
            output = self.out(torch.cat((rnn_output.squeeze(1), context.squeeze(1)), dim=1))
        else:
            # attn_weights: B x S
            attn_weights = self.attn(decoder_hc[0], encoder_outputs, mask)
            # context: (B x S -> B x 1 x S) @ (B x S x 2H) -> B x 1 x 2H
            context = torch.matmul(attn_weights.unsqueeze(1), encoder_outputs)

            # rnn_input: (B x 1 x E) + (B x 1 x 2H) -> (B x 1 x (E + 2H))
            rnn_input = torch.cat((word_embedded, context), dim=2)
            # rnn_output: B x 1 x H, (decoder_h, decoder_c): 1 x B x H
            rnn_output, decoder_hc = self.rnn(rnn_input, decoder_hc)

            # output: B x 3H -> B x T
            output = self.out(torch.cat((rnn_output.squeeze(1), context.squeeze(1), word_embedded.squeeze(1)), dim=1))

        return output, context, decoder_hc, attn_weights


class EncoderDecoder(nn.Module):
    def __init__(self, src_size, embed_size, hidden_size, trg_size, src_pad_idx, max_trg_len):
        super(EncoderDecoder, self).__init__()
        self.src_size = src_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.trg_size = trg_size
        self.src_pad_idx = src_pad_idx
        self.max_trg_len = max_trg_len

        self.encoder = RNNEncoder(src_size, embed_size, hidden_size)
        self.decoder = RNNDecoder(trg_size, embed_size, hidden_size)

    def init_weight(self):
        self.encoder.init_weight()
        self.decoder.init_weight()

    def forward(self, src_seqs, src_lens, trg_seqs, teacher_forcing_ratio=0.5):
        # attn_mask: B x S
        attn_mask = (src_seqs != self.src_pad_idx)
        # encoder_outputs: B x S x 2H, encoder_hc: 1 x B x H
        encoder_outputs, encoder_hc = self.encoder(src_seqs, src_lens)
        batch_size = src_seqs.size(0)
        # B x 1
        # decoder_input = torch.tensor([[self.SOS_idx]] * batch_size, dtype=torch.long).to(src_seqs.device)
        decoder_input = trg_seqs[:, 0:1]
        decoder_context = torch.zeros(batch_size, 1, 2 * self.hidden_size).to(src_seqs.device)
        decoder_hc = encoder_hc

        # trg_logits: B x S x T
        trg_logits = torch.zeros(batch_size, self.max_trg_len, self.trg_size).to(src_seqs.device)
        for di in range(1, self.max_trg_len):
            # decoder_output: B x T
            decoder_output, decoder_context, decoder_hc, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hc, encoder_outputs, attn_mask)
            trg_logits[:, di, :] = decoder_output
            # top_idxs: (B,)
            top_idxs = decoder_output.argmax(dim=1)
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            # B x 1
            decoder_input = trg_seqs[:, di:di + 1] if use_teacher_forcing else top_idxs.unsqueeze(1)

        # trg_logits: B x S x T
        return trg_logits
