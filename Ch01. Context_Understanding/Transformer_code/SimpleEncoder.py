#2.MHA / 3.Encoder

from turtle import forward
import torch

# torch.nn.Transformer
# Transformer > TransformerEncoder > transformerEncoderLayer > MultiheadAttention > Multi_head_attention_forward > _scaled_dot_product_attention

class MultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
       # Q = K = V # self attention

       num_batch, num_head, num_token_length, att_dim = K.shape

       Q = Q / (att_dim**0.5)
       attention_score = Q @ K.permute(0,1,3,2) #(8,2) -> (2,8)로 변환위해
       # K.T 는 2차원 계산만 가능
       # num_batch, num_head, num_token_length, num_token_length

       attention_score = torch.softmax(attention_score, dim=3)

       Z = attention_score @ V # num_batch, num_head, num_token_length, att_dim

       return Z, attention_score
    
class EncoderLayer(torch.nn.Module):

    def __init__(self, hidden_dim, num_head, dropout_p=0.5): #hidden_dim != att_dim
        super().__init__()

        self.num_head = num_head
        self.hidden_dim = hidden_dim
    
        self.MHA = MultiHeadAttention()
        
        self.W_Q = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.W_O = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.LayerNorm1 = torch.nn.LayerNorm(hidden_dim)
        self.LayerNorm2 = torch.nn.LayerNorm(hidden_dim)

        self.Dropout = torch.nn.Dropout(p=dropout_p)

        self.Linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.Linear2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.Activation = torch.nn.ReLU()

        # hidden_dim = 64
        # num_head = 4
        # att_dim = hidden_dim // num_head = 16
    def to_multihead(self, vector):
        #split
        num_batch, num_token_length, hidden_dim = vector.shape
        att_dim = hidden_dim // self.num_head
        vector= vector.view(num_batch, num_token_length, self.num_head, att_dim) 
        vector = vector.permute(0,2,1,3)
        return vector
    
    def forward(self, input_Q, input_K, input_V):
        # input_Q : [num_batch, num_token_length, hidden_dim]

        Q = self.W_Q(input_Q) # [num_batch, num_token_length, hidden_dim]
        K = self.W_K(input_K)  
        V = self.W_V(input_V)

        num_batch, num_token_length, hidden_dim = Q.shape

        Q = self.to_multihead(Q) # num_batch, num_head, num_token_length, att_dim
        K = self.to_multihead(K)
        V = self.to_multihead(V)

        Z, attention_score = self.MHA(Q,K,V) # num_batch, num_head, num_token_length, att_dim

        Z = Z.permute(0,2,1,3) # num_batch, num_token_length, num_head, att_dim
        Z = Z.reshape(num_batch, num_token_length, self.hidden_dim)
        # num_batch, num_token_length, hidden_dim)
        Z = self.W_O(Z)

        Z = self.LayerNorm1(self.Activation(Z) + input_Q)
        Z1= self.Dropout(Z)

        Z = self.Activation(self.Linear1(Z1))
        Z = self.Dropout(Z)
        Z = self.Activation(self.Linear2(Z))
        Z = self.Dropout(Z)

        Z = Z + Z1

        Z = self.LayerNorm2(Z)

        return Z


device = torch.device('cpu')
num_batch = 16
num_head = 2
hidden_dim = 64
num_token_length = 8

X = torch.Tensor(torch.randn(num_batch, num_token_length, hidden_dim))
print("X.shape:",X.shape)

self_attention_encoder = EncoderLayer(hidden_dim=hidden_dim, num_head=num_head)

Z = self_attention_encoder(input_Q=X, input_K=X, input_V=X)
print("Z.shape:",Z.shape)

from torch.nn.modules import TransformerEncoderLayer
official_encoder = TransformerEncoderLayer(d_model = hidden_dim, nhead = num_head, dim_feedforward = hidden_dim)
official_Z = official_encoder(X)
print("official_Z.shape:",official_Z.shape)
            