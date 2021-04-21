from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import init
from torch import Tensor
import torch.nn.functional as F
import math


class PhyaLSTM(nn.Module):
    """Implementation of the physics-aware-LSTM (PhyaLSTM)
        Parameters
        ----------
        input_size : int
            Number of dynamic features, which are those, passed to the LSTM at each time step---x_t
        Sca_size : int
            Number of static catchment attributes---Sca(t)
        hidden_size : int
            Number of hidden/memory cells.
        """
    def __init__(self, input_size: int, Sca_size: int, hidden_size: int, with_AM: bool=False):
        super(PhyaLSTM, self).__init__()
        self.input_size = input_size
        self.Sca_size = Sca_size
        self.hidden_size = hidden_size
        self.ME = True  # Memory Enhancement
        self.with_am = with_AM

        # input gate
        self.w_ix = Parameter(Tensor(hidden_size, input_size))
        self.w_ih = Parameter(Tensor(hidden_size, hidden_size))
        self.w_ia = Parameter(Tensor(hidden_size, hidden_size))
        self.b_i = Parameter(Tensor(hidden_size, 1))

        # forget gate
        self.w_fx = Parameter(Tensor(hidden_size, input_size))
        self.w_fo = Parameter(Tensor(hidden_size, hidden_size))
        self.w_fa = Parameter(Tensor(hidden_size, hidden_size))
        self.b_f = Parameter(Tensor(hidden_size, 1))

        # output gate
        self.w_ox = Parameter(Tensor(hidden_size, input_size))
        self.w_oh = Parameter(Tensor(hidden_size, hidden_size))
        self.w_oa = Parameter(Tensor(hidden_size, hidden_size))
        self.b_o = Parameter(Tensor(hidden_size, 1))

        # cell
        self.w_gx = Parameter(Tensor(hidden_size, input_size))
        self.w_gh = Parameter(Tensor(hidden_size, hidden_size))
        self.b_g = Parameter(Tensor(hidden_size, 1))

        # sca(t)
        self.w_a = Parameter(Tensor(hidden_size, Sca_size))
        self.b_a = Parameter(Tensor(hidden_size, 1))

        # assimilation obs(AM)
        if self.with_am:
            self.w_sim = Parameter(Tensor(hidden_size, hidden_size))
            self.w_obs = Parameter(Tensor(hidden_size, hidden_size))
            self.b_k = Parameter(Tensor(hidden_size, 1))

        self.reset_weigths()

    def reset_weigths(self):
        """reset weights
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def calc_corr(self, a, b):
        """Calculate the correlation coefficient of two vectors: a, b
        """
        a_avg = sum(a) / len(a)
        b_avg = sum(b) / len(b)
        # Calculate the molecule, covariance --- according to the covariance formula,
        # it should be divided by n, because it reduces n up and down in the correlation coefficient, so it can not be divided by n.
        cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
        # Calculate the denominator, variance product --- variance should be divided by N, so it can not be divided by n.
        sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))
        corr_factor = cov_ab / sq
        return corr_factor


    def forward(self,
                x_input: Tensor,
                x_sca: Tensor,
                soil_state: Tensor,
                h_obs: Tensor=None,
                h_input:Tensor=None,
                c_input:Tensor=None) -> Tuple[Tensor, Tensor]:
        """forward,
        Args:
            inputs: [batch_size, seq_size, input_size]
            input_Sca: [batch_size, seq_size, Sca_size]
            soil_state: [batch_size, seq_size]
            h_obs: [batch_size, seq_size, hidden_size]
        """
        batch_size, seq_size, input_dim = x_input.size()

        h_output = torch.zeros(batch_size, seq_size, self.hidden_size)
        c_output = torch.zeros(batch_size, seq_size, self.hidden_size)

        if h_input==None:
            h_t = torch.zeros(batch_size, self.hidden_size).t()
        if c_input==None:
            c_t = torch.zeros(batch_size, self.hidden_size).t()

        r = torch.zeros(batch_size, self.hidden_size)  # correlation coefficient

        for t in range(seq_size):
            x_t = x_input[:, t, :].t()    # [input_size, batch_size]
            sca_t = x_sca[:, t, :].t()    # [Sca_size, batch_size]

            a_t = torch.sigmoid(self.w_a @ sca_t + self.b_a)
            # input gate
            i = torch.sigmoid(self.w_ix @ x_t + self.w_ih @ h_t + self.w_ia @ a_t + self.b_i)
            # cell
            g = torch.tanh(self.w_gx @ x_t + self.w_gh @ h_t + self.b_g)
            # forget gate
            f = torch.sigmoid(self.w_fx @ x_t + self.w_fo @ h_t + self.w_fa @ a_t + self.b_f)
            # output gate
            o = torch.sigmoid(self.w_ox @ x_t + self.w_oh @ h_t + self.w_oa @ a_t + self.b_o)

            c_next = f * c_t + i * g    # [hidden_dim, batch_size]
            h_next = o * torch.tanh(c_next)   # [hidden_dim, batch_size]

            if self.ME:
                for i in range(batch_size):
                    r[i, :] = self.calc_corr(c_output[i, :t], soil_state[i, :t])
                c_next = c_next.mul(abs(r)+1)

            if self.with_am:
                K = torch.tanh(self.w_sim @ h_next + self.w_obs @ h_obs + self.b_k)
                h_next = h_next + K.mul(h_obs - h_next)

            h_output[:, t] = h_next.t()  # transpose:[batch_size, hidden_dim]
            c_output[:, t] = c_next.t()

            h_t = h_next
            c_t = c_next

        return (h_output, c_output)



class Obs_encoder(nn.Module):
    def __init__(self, x_obs_size, sca_size, hidden_size):
        super(Obs_encoder, self).__init__()
        self.x_obs_size = x_obs_size
        self.hidden_size = hidden_size
        self.sca_size = sca_size

        self.obs_encoder = PhyaLSTM(x_obs_size, hidden_size, sca_size, with_AM=False)

    def forward(self, x_obs, x_sca, soil_state:Tensor):
        h_obs, _ = self.obs_encoder(x_obs, x_sca, soil_state, None)
        return h_obs




class Prediction_module(nn.Module):
    def __init__(self, input_size, sca_size, x_obs_size, hidden_size, window_size):
        super(Prediction_module, self).__init__()
        self.x_size = input_size
        self.sca_size = sca_size
        self.x_obs_size = x_obs_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        self.obs_encoder = PhyaLSTM(x_obs_size, sca_size, hidden_size, with_AM=False)  # x_obs_size == input_size
        self.prediction = PhyaLSTM(input_size, sca_size, hidden_size, with_AM=True)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=1)
        self.w_omega = Parameter(Tensor(input_size, input_size))


    def forward(self, x_input:Tensor,
                x_sca:Tensor,
                soil_state:Tensor,
                x_obs:Tensor):

        h_obs, _ = self.obs_encoder(x_obs, x_sca, soil_state)
        h_, c_ = self.prediction(x_input[:,:-1], x_sca[:,:-1], soil_state[:,:-1], h_obs=h_obs[:,:-1])
        h_t = h_[:,-1:,:]

        # Attention Net
        x_i = x_input[:, -self.window_size:, :]   # [batch_size, window_size, input_size]
        x_t = x_input[:, -1:, :].transpose(1,2)                # [batch_size, input_size, 1]
        u = torch.tanh(torch.matmul(x_i, self.w_omega))  # [batch_size, window_size, input_size]
        att = torch.matmul(u, x_t)      # [batch_size, window_size, 1]
        att_score = F.softmax(att, dim=1)

        scored_c = c_[:, -self.window_size:, :] * att_score  # [batch_size, window_size, hidden_size]
        c_t = torch.sum(scored_c, dim=1)    # [batch_size, 1, hidden_size]

        h_t, _ = self.prediction(x_input[:,-1:], x_sca[:,-1:], soil_state[:,-1:], h_obs[:,-1:], h_input=h_t, c_input=c_t)

        pred = self.fc1(self.relu(h_t))   # [batch_size, 1, 1]
        pred = pred.squeeze(2)  # [batch_size, 1]

        return pred



