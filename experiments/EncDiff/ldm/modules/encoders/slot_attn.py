import os
import sys
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

class CartesianPositionalEmbedding(nn.Module):

    def __init__(self, channels, image_size):
        super().__init__()

        # self.projection = conv2d(4, channels, 1)
        self.projection = nn.Conv2d(4, channels, 1)
        # self.pe = nn.Parameter(self.build_grid(image_size).unsqueeze(0), requires_grad=False)

        self.register_buffer('pe', self.build_grid(image_size).unsqueeze(0))

    def build_grid(self, side_length):
        coords = torch.linspace(0., 1., side_length + 1)
        coords = 0.5 * (coords[:-1] + coords[1:])
        grid_y, grid_x = torch.meshgrid(coords, coords)
        return torch.stack((grid_x, grid_y, 1 - grid_x, 1 - grid_y), dim=0)

    def forward(self, inputs):
        # `inputs` has shape: [batch_size, out_channels, height, width].
        # `grid` has shape: [batch_size, in_channels, height, width].
        return inputs + self.projection(self.pe)

def is_square(n: float) -> bool:
    if n < 0:
        return False
    sqrt_n = math.sqrt(n)
    return sqrt_n ** 2 == n

class MultiHeadSTEVESA(nn.Module):

    # enable diffusers style config and model save/load
    def __init__(self, num_iterations, num_slots, num_heads,
                 input_size, out_size, slot_size, mlp_hidden_size, 
                 input_resolution, epsilon=1e-8, 
                 learnable_slot_init=False, 
                 bi_level=False):
        super().__init__()

        self.pos = CartesianPositionalEmbedding(input_size, input_resolution)
        self.in_layer_norm = nn.LayerNorm(input_size)
        self.in_mlp = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size)
            )
        if bi_level:
            # We tested bi_level slot attention (Jia et al. in https://arxiv.org/abs/2210.08990) at the early stage of the project,
            # and we didn't find it helpful
            assert learnable_slot_init, 'Bi-level training requires learnable_slot_init=True'

        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.learnable_slot_init = learnable_slot_init
        self.bi_level = bi_level

        assert slot_size % num_heads == 0, 'slot_size must be divisible by num_heads'

        if learnable_slot_init:
            self.slot_mu = nn.Parameter(torch.Tensor(1, num_slots, slot_size))
            nn.init.xavier_uniform_(self.slot_mu)
        else:
            # parameters for Gaussian initialization (shared by all slots).
            self.slot_mu = nn.Parameter(torch.Tensor(1, 1, slot_size))
            self.slot_log_sigma = nn.Parameter(torch.Tensor(1, 1, slot_size))
            nn.init.xavier_uniform_(self.slot_mu)
            nn.init.xavier_uniform_(self.slot_log_sigma)

        # norms
        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)

        # linear maps for the attention module.
        self.project_q = nn.Linear(slot_size, slot_size, bias=False)
        self.project_k = nn.Linear(input_size, slot_size, bias=False)
        self.project_v = nn.Linear(input_size, slot_size, bias=False)

        # slot update functions.
        self.gru = nn.GRUCell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(slot_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, slot_size))
        
        self.out_layer_norm = nn.LayerNorm(slot_size)
        self.out_linear = nn.Linear(slot_size, out_size)
        
    def forward(self, inputs):
        slots_collect = self.forward_slots(inputs)
        slots_collect = self.out_layer_norm(slots_collect)
        slots_collect = self.out_linear(slots_collect)
        return slots_collect

    def forward_slots(self, inputs):
        """
        inputs: batch_size x input_size x h x w
        return: batch_size x num_slots x slot_size
        """
        B, input_size, h, w = inputs.size()
        inputs = self.pos(inputs)
        inputs = rearrange(inputs, 'b n_inp h w -> b (h w) n_inp')
        inputs = self.in_mlp(self.in_layer_norm(inputs))

        # num_inputs = h * w

        if self.learnable_slot_init:
            slots = repeat(self.slot_mu, '1 num_s d -> b num_s d', b=B)
        else:
            # initialize slots
            slots = inputs.new_empty(B, self.num_slots, self.slot_size).normal_()
            slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots

        # setup key and value
        inputs = self.norm_inputs(inputs)
        k = rearrange(self.project_k(inputs), 'b n_inp (h d) -> b h n_inp d',
                      h=self.num_heads)  # Shape: [batch_size, num_heads, num_inputs, slot_size].
        v = rearrange(self.project_v(inputs), 'b n_inp (h d) -> b h n_inp d',
                      h=self.num_heads)  # Shape: [batch_size, num_heads, num_inputs, slot_size].
        k = (self.slot_size ** (-0.5)) * k

        # loop over frames
        attns_collect = []
        slots_collect = []
        # corrector iterations
        for i in range(self.num_iterations):
            if self.bi_level and i == self.num_iterations - 1:
                slots = slots.detach() + self.slot_mu - self.slot_mu.detach()
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = rearrange(self.project_q(slots), 'b n_s (h d) -> b h n_s d',
                            h=self.num_heads)  # Shape: [batch_size, num_heads, num_slots, slot_size].
            attn_logits = torch.einsum('...id,...sd->...is', k,
                                        q)  # Shape: [batch_size, num_heads, num_inputs, num_slots]
            attn = F.softmax(rearrange(attn_logits, 'b h n_inp n_s -> b n_inp (h n_s)'), -1)
            attn_vis = rearrange(attn, 'b n_inp (h n_s) -> b h n_inp n_s', h=self.num_heads)
            # `attn_vis` has shape: [batch_size, num_inputs, num_slots].

            # Weighted mean.
            attn = attn_vis + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)  # norm over inputs
            updates = torch.einsum('...is,...id->...sd', attn,
                                    v)  # Shape: [batch_size, num_heads, num_slots, num_inp].
            updates = rearrange(updates, 'b h n_s d -> b n_s (h d)')
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            slots = self.gru(updates.view(-1, self.slot_size),
                                slots_prev.reshape(-1, self.slot_size))
            slots = slots.view(-1, self.num_slots, self.slot_size)

            slots = slots + self.mlp(self.norm_mlp(slots))

        # collect
        attns_collect = attn_vis
        slots_collect = slots

        return slots_collect

if __name__ == "__main__":
    # test
    slot_attn = MultiHeadSTEVESA(
        num_iterations=3, 
        num_slots=24, 
        num_heads=1,
        input_size=192, # unet_encoder.config.out_channels
        out_size=192, # unet.config.cross_attention_dim
        slot_size=192, 
        mlp_hidden_size=192,
        input_resolution=64, # unet_encoder.config.latent_size
        learnable_slot_init=False
    )
    inputs = torch.randn(2, 192, 64, 64)
    slots_collect = slot_attn(inputs)
    print(slots_collect.shape)
    pass