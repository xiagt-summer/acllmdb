"""
RNN Model by Hand
"""
import torch
import torch.nn as nn
from torchinfo import summary

class my_rnn_cell(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.W_ih = nn.Linear(embed_dim, hidden_dim, bias=True)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=True)
    def forward(self, x, h_last):
        return torch.tanh(self.W_ih(x) + self.W_hh(h_last))

# Multi-layer and bidirectional RNN
class my_rnn(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, num_layers=1, bidirectional=False):
        super().__init__()
        self.hidden_dim     = hidden_dim
        self.num_layers     = num_layers
        self.bidirectional  = bidirectional
        self.num_directions = 2 if bidirectional else 1

        layers = []
        for layer in range(num_layers):
            in_dim = embed_dim if layer == 0 else hidden_dim * self.num_directions
            cells = nn.ModuleDict({
                'fore': my_rnn_cell(in_dim, hidden_dim),
                'back': my_rnn_cell(in_dim, hidden_dim)
            }) if bidirectional else nn.ModuleDict({
                'fore': my_rnn_cell(in_dim, hidden_dim)
            })
            layers.append(cells)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Let's say B = batch_size,
                  N = seq_len,
                  E = embed_dim,
                  H = hidden_dim
        x: (B, N, E)
        """
        B, N, E = x.size()

        # Initialize zero hidden states with the same device as x.
        h = [
            [x.new_zeros(B, self.hidden_dim) for _ in range(self.num_directions)] 
            for _ in range(self.num_layers)
        ]

        # Process layer by layer
        layer_input = x
        for layer_idx, cells in enumerate(self.layers):
            
            # forward
            fore_seq = []
            # h_f: the hidden state in forward rnn, 
            h_f = h[layer_idx][0] # (B, H)
            for t in range(N):
                h_f = cells['fore'](layer_input[:, t, :], h_f)
                fore_seq.append(h_f) 
            fore_seq = torch.stack(fore_seq, dim=1) # [N * (B, H)] --> (B, N, H)

            # backward
            if self.bidirectional:
                back_seq = []
                # h_b: the hidden state in backward rnn 
                h_b = h[layer_idx][1] # (B, H)
                for t in reversed(range(N)):
                    h_b = cells['back'](layer_input[:, t, :], h_b) 
                    back_seq.insert(0, h_b)                            
                back_seq = torch.stack(back_seq, dim=1) # [N * (B, H)] --> (B, N, H)

                # Mix forward and backward information together!
                layer_input = torch.cat([fore_seq, back_seq], dim=-1)  # (B, N, 2H)
                h[layer_idx][0] = fore_seq[:, -1, :] # (B, H)
                h[layer_idx][1] = back_seq[:,  0, :] # (B, H)
            else: 
                layer_input = fore_seq # (B, N, H)
                h[layer_idx][0] = fore_seq[:, -1, :] # (B, H)
        
        final_h = torch.cat(h[-1], dim=-1)                             
        return final_h # (B, num_dirs * H) --> (B, O)
        
class my_rnn_classifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=1, bidirectional=False):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = my_rnn(embed_dim, hidden_dim, output_dim, num_layers, bidirectional)

        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

    def forward(self, x):
        embeded = self.embedding(x)         # (B, N, E)
        hidden = self.rnn(embeded)          # (B, num_directions * H)
        return self.fc(hidden)              # (B, O)


"""
RNN Model with Pytorch Internal Functions
Classic N-to-1 RNN Model
"""
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=1, bidirectional=False):
        super().__init__()
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

    def forward(self, x):
        embeded = self.embedding(x)         # (batch_size, seq_len, embeded_dim)
        output, hidden = self.rnn(embeded)  #  hidden: (num_layers * num_directions, batch_size, hidden_dim)
        # (num_layers, num_directions, batch_size, hidden_dim)
        hidden = hidden.reshape(self.num_layers, self.num_directions, embeded.size(0), self.hidden_dim)
        last_h = hidden[-1]
        if self.num_directions == 2:
            h = torch.cat([last_h[0], last_h[1]], dim=-1)
        else:
            h = last_h[0]
        return self.fc(h)          # (batch_size, output_dim)


def compare_my_rnn_with_pytorch_rnn():
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.double)

    # Temporary hyperparameters
    seq_len         = 512
    embed_dim       = 512
    hidden_dim      = 256
    num_layers      = 2
    bidirectional   = True
    output_dim      = 2
    batch_size      = 64

    vocab_size = 10000
    model1 = my_rnn_classifier(vocab_size, embed_dim, hidden_dim, output_dim, num_layers=num_layers, bidirectional=bidirectional)
    model2 = RNNClassifier(vocab_size, embed_dim, hidden_dim, output_dim, num_layers=num_layers, bidirectional=bidirectional)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1.to(device)
    model2.to(device)

    print(summary(model1, input_size=(batch_size, seq_len), dtypes=[torch.long], device=device))
    print(summary(model2, input_size=(batch_size, seq_len), dtypes=[torch.long], device=device))
    # my_rnn_classifer has the sane params with Pytorch internal RNN, but larger memory size...

    # 查看 model 的参数状态
    sd1 = model1.state_dict()
    sd2 = model2.state_dict()

    # Names of model training params
    # print("Names of model1 params:")
    # for k, _ in sd1.items():
    #     print(k)

    # print("Names of model2 params:")
    # for k, _ in sd2.items():
    #     print(k)

    # Match 两套网络的参数命名. Model1 加载 Model2 的参数
    map = {
      "embedding.weight"               : sd2["embedding.weight"],
      "fc.weight"                      : sd2["fc.weight"],
      "fc.bias"                        : sd2["fc.bias"],
    }
    for i in range(num_layers):
        if bidirectional:
            for direction, suffix in [("fore", ""), ("back", "_reverse")]:
                base = f"rnn.layers.{i}.{direction}"
                map[f"{base}.W_ih.weight"] = sd2[f"rnn.weight_ih_l{i}{suffix}"]
                map[f"{base}.W_ih.bias"]   = sd2[f"rnn.bias_ih_l{i}{suffix}"]
                map[f"{base}.W_hh.weight"] = sd2[f"rnn.weight_hh_l{i}{suffix}"]
                map[f"{base}.W_hh.bias"]   = sd2[f"rnn.bias_hh_l{i}{suffix}"]
        else:
            map[f"rnn.layers.{i}.fore.W_ih.weight"] = sd2[f"rnn.weight_ih_l{i}"]
            map[f"rnn.layers.{i}.fore.W_ih.bias"]   = sd2[f"rnn.bias_ih_l{i}"]
            map[f"rnn.layers.{i}.fore.W_hh.weight"] = sd2[f"rnn.weight_hh_l{i}"]
            map[f"rnn.layers.{i}.fore.W_hh.bias"]   = sd2[f"rnn.bias_hh_l{i}"]
    model1.load_state_dict(map, strict=False)

    # Random input
    x = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long).to(device)
    o1 = model1(x)
    o2 = model2(x)

    print("forward output max abs diff:", (o1 - o2).abs().max().item())
    print("Are out the same? ", torch.allclose(o1, o2, atol=1e-6, rtol=1e-5))

# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# my_rnn_classifier                        [64, 2]                   --
# ├─Embedding: 1-1                         [64, 512, 512]            51,549,696
# ├─my_rnn: 1-2                            [64, 512]                 --
# │    └─ModuleList: 2-1                   --                        --
# │    │    └─ModuleDict: 3-1              --                        525,312
# ├─Linear: 1-3                            [64, 2]                   1,026
# ==========================================================================================
# Total params: 52,076,034
# Trainable params: 52,076,034
# Non-trainable params: 0
# Total mult-adds (Units.GIGABYTES): 20.51
# ==========================================================================================
# Input size (MB): 0.26
# Forward/backward pass size (MB): 805.31
# Params size (MB): 416.61
# Estimated Total Size (MB): 1222.18
# ==========================================================================================
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# RNNClassifier                            [64, 2]                   --
# ├─Embedding: 1-1                         [64, 512, 512]            51,549,696
# ├─RNN: 1-2                               [64, 512, 512]            525,312
# ├─Linear: 1-3                            [64, 2]                   1,026
# ==========================================================================================
# Total params: 52,076,034
# Trainable params: 52,076,034
# Non-trainable params: 0
# Total mult-adds (Units.GIGABYTES): 20.51
# ==========================================================================================
# Input size (MB): 0.26
# Forward/backward pass size (MB): 536.87
# Params size (MB): 416.61
# Estimated Total Size (MB): 953.74
# ==========================================================================================
# forward output max abs diff: 5.551115123125783e-16
# Are out the same?  True