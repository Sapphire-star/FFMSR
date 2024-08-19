import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec
import math
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class FeedForward(nn.Module):
    def __init__(self, hidden_size, inner_size, dropout):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, inner_size)
        self.w_2 = nn.Linear(inner_size, hidden_size)
        self.hidden_size = hidden_size
        self.inner_size = inner_size

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        return self.w_2(self.gelu(self.w_1(x)))

class FilterLayer(nn.Module):
    def __init__(self, hidden_size, max_length, dropout_prob):
        super(FilterLayer, self).__init__()
        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.complex_weight = nn.Parameter(torch.randn(1, max_length//2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(dropout_prob)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        # self.feedforward = FeedForward(hidden_size, 2 * hidden_size, dropout_prob)
        # self.ffn_dropout = nn.Dropout(dropout_prob)
        # self.ffn_layernorm = LayerNorm(hidden_size, eps=1e-12)
        # self.dense = nn.Linear(hidden_size, hidden_size)


    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        #sequence_emb_fft = torch.rfft(input_tensor, 2, onesided=False)  # [:, :, :, 0]
        #sequence_emb_fft = torch.fft(sequence_emb_fft.transpose(1, 2), 2)[:, :, :, 0].transpose(1, 2)
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        # sequence_emb_fft = self.dense(sequence_emb_fft)
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)

class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, n_items, drop_ratio=0.0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim // 2, 16),
            nn.LeakyReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1),
        )
        # self.last_layer_boost = nn.Parameter(torch.ones(n_items, 1))

    def forward(self, x, mask=None):
        out = self.linear(x)
        if mask is not None:
            out = out.masked_fill(mask, -100000)
            weight = F.softmax(out, dim=1)
            return weight
        else:
            weight = F.softmax(out, dim=1)
            # weight = torch.sigmoid(out)
        return weight


class MLTRec(SASRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.train_stage = config['train_stage']
        self.temperature = config['temperature']
        self.lam = config['lambda']
        self.ft_layer = config['ft_layer']
        self.device = config['device']

        self.layer_embedding = copy.deepcopy(dataset.layer_embedding)
        self.global_prompt = nn.Parameter(torch.zeros(self.n_items, config['plm_size']), requires_grad=False)
        self.id_gating = nn.Linear(self.hidden_size, 1)
        self.id_gating.weight.data.normal_(mean=0, std=0.02)
        self.prompt_gating = nn.Linear(self.hidden_size, 1)
        self.prompt_gating.weight.data.normal_(mean=0, std=0.02)
        self.complex_weight = nn.Parameter(torch.randn(1, self.max_seq_length // 2 + 1, self.hidden_size, 2, dtype=torch.float32) * 0.02)
        self.moe_adaptor = MoEAdaptorLayer(
            config['n_exps'],
            config['adaptor_layers'],
            config['adaptor_dropout_prob']
        )
        self.global_adapter = MoEAdaptorLayer(
            config['n_exps'],
            config['adaptor_layers'],
            config['adaptor_dropout_prob']
        )
        self.attention = AttentionLayer(2 * config['hidden_size'], self.n_items)
        self.input_dropout = nn.Dropout(0.1)
        # self.attn_layer = nn.MultiheadAttention(self.hidden_size, num_heads=4)


    def get_prompt(self, batch, seq_out):
        prompt = self.prompts.unsqueeze(1)
        prompt, _ = self.attn_layer(seq_out, prompt, prompt)
        return prompt

    def forward(self, item_seq, item_emb, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.input_dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]


    def diffloss(self, input1, input2):
        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

    def get_fed_emb(self):
        id_emb = self.item_embedding.weight.clone().detach().unsqueeze(1).expand(-1, self.ft_layer, -1)
        text_emb = self.moe_adaptor(self.layer_embedding)
        id_text_emb = torch.cat((id_emb, text_emb), dim=-1)
        attn = self.attention(id_text_emb)
        fed_emb = torch.matmul(attn.permute(0, 2, 1), self.layer_embedding).squeeze(1)
        return fed_emb

    def get_item_emb(self):
        id_emb = self.item_embedding.weight.clone().detach().unsqueeze(1).expand(-1, self.ft_layer, -1)
        text_emb = self.moe_adaptor(self.layer_embedding)
        id_text_emb = torch.cat((id_emb, text_emb), dim=-1)
        attn = self.attention(id_text_emb)
        text_emb = torch.matmul(attn.permute(0, 2, 1), text_emb).squeeze(1)
        return text_emb

    def filter_prompt(self, item_emb_list):
        x = torch.fft.rfft(item_emb_list, dim=1, norm='ortho')
        complex_weight = torch.view_as_complex(self.complex_weight)
        x = x * complex_weight
        res = torch.fft.irfft(x, n=item_emb_list.shape[1], dim=1, norm='ortho')
        return res


    def calculate_loss(self, interaction):
        # Loss for fine-tuning
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        text_emb = self.get_item_emb()

        fed_prompt = self.global_adapter(self.global_prompt)
        id_gate = torch.sigmoid(self.id_gating(self.item_embedding.weight))
        test_item_emb = id_gate * self.item_embedding.weight + text_emb
        prompt_list = self.filter_prompt(self.global_adapter(self.global_prompt[item_seq]))
        item_emb_list = test_item_emb[item_seq] + prompt_list
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        seq_output = F.normalize(seq_output, dim=1)
        prompt_gate = torch.sigmoid(self.prompt_gating(fed_prompt))
        test_item_emb = F.normalize(test_item_emb + prompt_gate * fed_prompt, dim=1)

        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items)
        diff_loss1 = self.diffloss(text_emb, self.item_embedding.weight)
        diff_loss2 = self.diffloss(fed_prompt, self.item_embedding.weight)
        losses = loss + diff_loss1 + diff_loss2
        return losses

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        text_emb = self.get_item_emb()
        fed_prompt = self.global_adapter(self.global_prompt)
        id_gate = torch.sigmoid(self.id_gating(self.item_embedding.weight))
        test_item_emb = id_gate * self.item_embedding.weight + text_emb
        prompt_list = self.filter_prompt(self.global_adapter(self.global_prompt[item_seq]))
        item_emb_list = test_item_emb[item_seq] + prompt_list
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        seq_output = F.normalize(seq_output, dim=1)
        prompt_gate = torch.sigmoid(self.prompt_gating(fed_prompt))
        test_item_emb = F.normalize(test_item_emb + prompt_gate * fed_prompt, dim=1)
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B n_items]
        return scores
