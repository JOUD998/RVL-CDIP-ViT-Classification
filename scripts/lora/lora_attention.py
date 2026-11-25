
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAMultiheadAttention(nn.Module):
    def __init__(self, original_attention, rank=4, alpha=8):
        super().__init__()

        self.attention = original_attention
        self.rank = rank
        self.scaling = alpha / rank

        embed_dim = original_attention.embed_dim

        # Trainierbare Low-Rank-Matrizen für Query- und Value-Projektionen
        # Die B-Matrizen werden mit null initialisiert, damit LoRA anfangs keine Änderung bewirkt
        self.lora_q_A = nn.Parameter(
            torch.empty(rank, embed_dim)
        )
        self.lora_q_B = nn.Parameter(
            torch.zeros(embed_dim, rank)
        )

        self.lora_v_A = nn.Parameter(
            torch.empty(rank, embed_dim)
        )
        self.lora_v_B = nn.Parameter(
            torch.zeros(embed_dim, rank)
        )

        # Kaiming-Initialisierung der A-Matrizen
        nn.init.kaiming_uniform_(self.lora_q_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_v_A, a=math.sqrt(5))

    def forward(self, query, key, value, **kwargs):
        # Ursprüngliche Projektionsgewichte für Query, Key und Value
        W = self.attention.in_proj_weight

        # Low-Rank-Aktualisierungen berechnen: Delta W = (alpha / rank) * B * A
        delta_q = (
            self.lora_q_B @ self.lora_q_A
        ) * self.scaling

        delta_v = (
            self.lora_v_B @ self.lora_v_A
        ) * self.scaling

        # Query und Value anpassen, während die Key-Projektion unverändert bleibt
        updated_W = torch.cat([
            W[:self.attention.embed_dim] + delta_q,
            W[self.attention.embed_dim:2*self.attention.embed_dim],
            W[2*self.attention.embed_dim:] + delta_v
        ], dim=0)

        # Multi-Head-Attention mit den angepassten Projektionsgewichten ausführen
        # Ein- und Ausgabe werden zwischen Batch-First- und Sequenz-First-Format umgewandelt
        output, weights = F.multi_head_attention_forward(
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1),
            embed_dim_to_check=self.attention.embed_dim,
            num_heads=self.attention.num_heads,
            in_proj_weight=updated_W,
            in_proj_bias=self.attention.in_proj_bias,
            bias_k=self.attention.bias_k,
            bias_v=self.attention.bias_v,
            add_zero_attn=self.attention.add_zero_attn,
            dropout_p=self.attention.dropout,
            out_proj_weight=self.attention.out_proj.weight,
            out_proj_bias=self.attention.out_proj.bias,
            training=self.training,
            key_padding_mask=kwargs.get("key_padding_mask"),
            need_weights=kwargs.get("need_weights", True),
            attn_mask=kwargs.get("attn_mask"),
            use_separate_proj_weight=False,
            average_attn_weights=kwargs.get(
                "average_attn_weights", True
            ),
            is_causal=kwargs.get("is_causal", False),
        )

        return output.transpose(0, 1), weights
