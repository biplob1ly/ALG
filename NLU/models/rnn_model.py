import torch
from torch import nn
from torch.nn import functional as F
import random


class RNNEncoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, bidirect=True, dropout_rate=0.1):
        super(RNNEncoder, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.bidirect = bidirect

        self.embedder = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=bidirect)
        self.dropout = nn.Dropout(dropout_rate)

    def init_weight(self):
        """
        Initializes weight matrices using Xavier initialization
        :return:
        """
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        if self.bidirect:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        if self.bidirect:
            nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)

    def forward(self, src_seqs, lens):
        # B x S -> B x S x E
        embedded = self.embedder(src_seqs)

        # encoder_outputs: B x S x E -> B x S x H, encoder_hc: (h,c), h: n_layers*n_direction x B x H
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lens, batch_first=True)
        packed_output, encoder_hc = self.rnn(packed_embedded)
        encoder_outputs, lens = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        return encoder_outputs, encoder_hc


class SlotAttention(nn.Module):
    def __init__(self, method, hidden_size):
        super(SlotAttention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        self.attn = nn.Linear(hidden_size * 3, 1, bias=False)

    def forward(self, decoder_h, encoder_outputs):
        seqLen = encoder_outputs.size(1)
        # decoder_h: (1 x B x H -> B x 1 x H) -> B x S x H
        decoder_h = decoder_h.permute(1, 0, 2).repeat(1, seqLen, 1)
        # (B x S x H) + (B x S x 2H) -> B x S x 3H
        attn_input = torch.cat((decoder_h, encoder_outputs), dim=2)

        #TODO Use valid lens to mask before softmaxing
        # B x S x 3H -> B x S
        attn_energies = F.softmax(self.attn(attn_input).squeeze(2), dim=1)
        return attn_energies


class RNNDecoder(nn.Module):
    def __init__(self, slot_size, embed_size, hidden_size, attn_method='general', bidirect=False, dropout_rate=0.1):
        super(RNNDecoder, self).__init__()
        self.slot_size = slot_size  # T
        self.embed_size = embed_size    # E
        self.hidden_size = hidden_size  # H
        self.bidirect = bidirect

        self.embedder = nn.Embedding(slot_size, embed_size)
        self.rnn = nn.LSTM(embed_size + 4 * hidden_size, hidden_size, batch_first=True, bidirectional=bidirect)
        self.slot_attn = SlotAttention(attn_method, hidden_size)
        self.slot_tagger = nn.Linear(hidden_size, slot_size)
        self.dropout = nn.Dropout(dropout_rate)

    def init_weight(self):
        """
        Initializes weight matrices using Xavier initialization
        :return:
        """
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        if self.bidirect:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        if self.bidirect:
            nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)

    # target_input: B x 1,
    # aligned_input: B x 1 X 2H,
    # (decoder_h, decoder_c): 1 x B x H,
    # encoder_outputs: B x S x 2H
    def forward(self, target_input, aligned_input, decoder_h, decoder_c, encoder_outputs):
        # B x 1 -> B x 1 x E
        embedded = self.embedder(target_input)
        #TODO Add dropout here

        # attn_weights: B x S
        attn_weights = self.slot_attn(decoder_h, encoder_outputs)
        # context: (B x S -> B x 1 x S) @ (B x S x 2H) -> B x 1 x 2H
        context = torch.matmul(attn_weights.unsqueeze(1), encoder_outputs)

        # rnn_input: (B x 1 x E) + (B x 1 x 2H) + (B x 1 x 2H) -> B x 1 x (E + 4H)
        rnn_input = torch.cat((embedded, aligned_input, context), dim=2)
        # rnn_output: B x 1 x H, (decoder_h, decoder_c): 1 x B x H
        rnn_output, (decoder_h, decoder_c) = self.rnn(rnn_input, (decoder_h, decoder_c))
        # slot_outputs: (B x 1 x H -> B x H) -> B x T
        slot_outputs = self.slot_tagger(rnn_output.squeeze(1))

        return slot_outputs, decoder_h, decoder_c


class IntentAttention(nn.Module):
    def __init__(self, hidden_size, attn_dims=1, attn_expansion=1, dropout_rate=0.1):
        super(IntentAttention, self).__init__()
        self.l1 = nn.Linear(hidden_size, hidden_size*attn_expansion)
        self.tnh = nn.Tanh()
        # self.dropout = nn.Dropout(dropout_rate)
        self.l2 = nn.Linear(hidden_size*attn_expansion, attn_dims)

    def forward(self, hidden, attn_mask=None):
        # output_1: B x S x 2H -> B x S x attn_expansion*2H
        output_1 = self.tnh(self.l1(hidden))
        # output_1 = self.dropout(output_1)

        # output_2: B x S x attn_expansion*2H -> B x S x attn_dims(O)
        output_2 = self.l2(output_1)

        # Masked fill to avoid softmaxing over padded words
        if attn_mask is not None:
            output_2 = output_2.masked_fill(attn_mask == 0, -1e9)

        # attn_weights: B x S x attn_dims(O) -> B x O x S
        attn_weights = F.softmax(output_2, dim=1).transpose(1, 2)

        # weighted_output: (B x O x S) @ (B x S x 2H) -> B x O x 2H
        weighted_output = attn_weights @ hidden
        weighted_output = weighted_output.sum(dim=1)   # B x O x 2H -> B x 2H
        return weighted_output, attn_weights


class IntentClassifier(nn.Module):
    def __init__(self, hidden_size, intent_size, dropout_rate=0.1):
        super(IntentClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.intent_size = intent_size
        self.intent_attn = IntentAttention(2 * hidden_size)
        self.intent_classifier = nn.Linear(3 * hidden_size, intent_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, encoder_h1, encoder_outputs):
        # intent_context: B x 2H
        intent_context, intent_attn_weights = self.intent_attn(encoder_outputs)
        # (B x H) + (B x 2H) -> B x 3H
        intent_classifier_input = torch.cat((encoder_h1, intent_context), dim=1)
        # intent_classifier_input = self.dropout(intent_classifier_input)
        # B x 3H -> B x I
        intent_outputs = self.intent_classifier(intent_classifier_input)
        return intent_outputs


class JointEncoderDecoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, intent_size, slot_size, SOS_idx, dropout_rate=0.1):
        super(JointEncoderDecoder, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size    # E
        self.hidden_size = hidden_size  # H
        self.intent_size = intent_size  # I
        self.slot_size = slot_size  # T
        self.SOS_idx = SOS_idx

        self.encoder = RNNEncoder(input_size, embed_size, hidden_size)
        self.intent_classifier = IntentClassifier(hidden_size, intent_size)

        self.decoder = RNNDecoder(slot_size, embed_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def init_weight(self):
        self.encoder.init_weight()
        self.decoder.init_weight()

    def forward(self, src_seqs, trg_seqs, lens, teacher_forcing_ratio=0.5):
        # encoder_outputs: B x S x 2H, encoder_hc: (h,c), h: n_layers*n_direction x B x H
        encoder_outputs, encoder_hc = self.encoder(src_seqs, lens)
        # Taking the last backward hidden state, encoder_h1: B x H
        encoder_h1 = encoder_hc[0][-1]

        # intent_logits: B x I
        intent_logits = self.intent_classifier(encoder_h1, encoder_outputs)

        batch_size, trg_len = src_seqs.size()
        # slot_logits: B x S x T
        slot_logits = torch.zeros((batch_size, trg_len, self.slot_size), device=src_seqs.device)
        use_teacher_forcing = random.random() < teacher_forcing_ratio

        # B x 1
        decoder_input = torch.tensor([[self.SOS_idx]]*batch_size, dtype=torch.long).to(src_seqs.device)
        # B x H -> 1 x B x H
        decoder_h, decoder_c = encoder_hc[0][-1].unsqueeze(0), encoder_hc[1][-1].unsqueeze(0)

        max_trg_len = encoder_outputs.size(1)
        if use_teacher_forcing:
            for di in range(max_trg_len):
                # aligned_input: B x 1 x 2H
                aligned_input = encoder_outputs[:, di:di+1, :]
                # decoder_output: B x T, (decoder_h, decoder_c): 1 x B x H
                decoder_output, decoder_h, decoder_c = self.decoder(decoder_input, aligned_input, decoder_h, decoder_c, encoder_outputs)
                slot_logits[:, di, :] = decoder_output
                # B x 1
                decoder_input = trg_seqs[:, di:di+1]
        else:
            for di in range(max_trg_len):
                aligned_input = encoder_outputs[:, di:di+1, :]
                # decoder_output: B x T, (decoder_h, decoder_c): 1 x B x H
                decoder_output, decoder_h, decoder_c = self.decoder(decoder_input, aligned_input, decoder_h, decoder_c, encoder_outputs)
                slot_logits[:, di, :] = decoder_output
                # top_idxs: (B,)
                top_idxs = torch.argmax(decoder_output, dim=1)
                # B x 1
                decoder_input = top_idxs.unsqueeze(1).to(src_seqs.device)

        # intent_logits: B x I, slot_logits: B x S x T
        return intent_logits, slot_logits
