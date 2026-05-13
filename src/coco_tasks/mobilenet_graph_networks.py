"""
# Modified by Maneth Banula Perera (2026)
# Description: MobileNetV3-based graph network replacing ResNet101 backbone
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights



class ExtractorMobileNet(nn.Module):

    out_channels = 128

    def __init__(self, out_dim: int = 128):
        super().__init__()

        assert out_dim % 2 == 0, "out_dim must be even — split equally between avg and max streams"
        stream_dim = out_dim // 2   # each stream projects to half the output dim

        # load MobileNetV3-Large with ImageNet pretrained weights
        backbone = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2
        )

        # conv feature layers — named 'extractor' so graph_experiments.py
        # freeze call works: self.network.extractor.extractor.train(mode=False)
        self.extractor = backbone.features

        # pooling ops — both operate on same [B x 960 x 7 x 7] feature map
        self.avgpool = nn.AdaptiveAvgPool2d(1)   # global avg → [B x 960 x 1 x 1]
        self.maxpool = nn.AdaptiveMaxPool2d(1)   # global max → [B x 960 x 1 x 1]

        # separate learned projections — each stream independently compressed
        # avg stream: captures distributed activation (texture, colour, mean shape)
        self.avg_proj = nn.Sequential(
            nn.Linear(960, stream_dim, bias=False),
            nn.LayerNorm(stream_dim),
            nn.ReLU(inplace=True),
        )
        # max stream: captures peak activation (edges, corners, dominant features)
        self.max_proj = nn.Sequential(
            nn.Linear(960, stream_dim, bias=False),
            nn.LayerNorm(stream_dim),
            nn.ReLU(inplace=True),
        )

        # initialise both projections with xavier uniform for stable early training
        nn.init.xavier_uniform_(self.avg_proj[0].weight)
        nn.init.xavier_uniform_(self.max_proj[0].weight)

        # freeze all conv layer parameters — only projection layers train
        for param in self.extractor.parameters():
            param.requires_grad = False

        self.out_channels = out_dim   # 128 by default

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B x 3 x 224 x 224]
        Returns:
            phi_o: [B x 128]
                   first 64 dims = avg-pool projected features (texture/colour)
                   last  64 dims = max-pool projected features (edges/peaks)
        """
        feat = self.extractor(x)             # [B x 960 x 7 x 7]

        avg = self.avgpool(feat).flatten(1)  # [B x 960]
        mx  = self.maxpool(feat).flatten(1)  # [B x 960]

        avg_out = self.avg_proj(avg)         # [B x 64]
        max_out = self.max_proj(mx)          # [B x 64]

        return torch.cat([avg_out, max_out], dim=1)  # [B x 128]

    def __repr__(self) -> str:
        return "ExtractorMobileNet(avg_stream=64, max_stream=64, out=128)"



class AllLinearAggregator(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.drop = nn.Dropout(0.25)
        self.layer = nn.Linear(
            in_features=self.in_features, out_features=self.out_features
        )

    def forward(self, h_t: Tensor, phi_o: Tensor, d: Tensor) -> Tensor:

        B = h_t.size(0)
        transformed_h_t = self.layer(h_t)                          # [B x out_features]
        x_t_v = torch.sum(transformed_h_t, dim=0, keepdim=True)    # [1 x out_features]
        x_t = x_t_v.repeat(B, 1)                                   # [B x out_features]
        x_t -= transformed_h_t                                      # remove self
        x_t = self.drop(x_t)
        return x_t

    def __repr__(self) -> str:
        return "name:{}-inf:{}-outf:{}".format(
            self.__class__.__name__, self.in_features, self.out_features
        )


class AllLinearAggregatorWeightedWithDetScore(nn.Module):


    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.drop = nn.Dropout(0.25)
        self.layer = nn.Linear(
            in_features=self.in_features, out_features=self.out_features
        )

    def forward(self, h_t: Tensor, phi_o: Tensor, d: Tensor) -> Tensor:

        B = h_t.size(0)
        transformed_h_t = self.layer(h_t) * d                      # scale by det score
        x_t_v = torch.sum(transformed_h_t, dim=0, keepdim=True)    # [1 x out_features]
        x_t = x_t_v.repeat(B, 1)                                   # [B x out_features]
        x_t -= transformed_h_t                                      # remove self
        x_t = self.drop(x_t)
        return x_t

    def __repr__(self) -> str:
        return "name:{}-inf:{}-outf:{}".format(
            self.__class__.__name__, self.in_features, self.out_features
        )

class GATv2Aggregator(nn.Module):


    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # value projection — same role as W in AllLinearAggregator
        self.W_val = nn.Linear(in_features, out_features, bias=False)

        # attention: projects concatenated pair [h_v || h_v'] to scalar
        self.W_att = nn.Linear(in_features * 2, 1, bias=False)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.drop = nn.Dropout(0.25)

    def forward(self, h_t: Tensor, phi_o: Tensor, d: Tensor) -> Tensor:

        B = h_t.size(0)

        # value projection for all nodes
        val = self.W_val(h_t)                    # [B x out_features]

        # build all pairs for attention scoring
        # h_i repeated for each j: [B x B x in_features]
        h_i = h_t.unsqueeze(1).expand(B, B, -1)  # [B x B x in_features]
        h_j = h_t.unsqueeze(0).expand(B, B, -1)  # [B x B x in_features]

        # concatenate pairs and score
        pair = torch.cat([h_i, h_j], dim=-1)      # [B x B x in_features*2]
        e = self.leaky_relu(self.W_att(pair))      # [B x B x 1]
        e = e.squeeze(-1)                          # [B x B]

        # mask self-connections — set diagonal to -inf before softmax
        mask = torch.eye(B, device=h_t.device).bool()
        e = e.masked_fill(mask, float('-inf'))

        alpha = torch.softmax(e, dim=1)            # [B x B]
 

        alpha = alpha * d.T                        # [B x B]
 

        row_sum = alpha.sum(dim=1, keepdim=True).clamp(min=1e-8)
        alpha = alpha / row_sum                    # [B x B]
 
        alpha = self.drop(alpha)
        x_t = alpha @ val                          # [B x out_features]
 
        return x_t

    def __repr__(self) -> str:
        return "name:{}-inf:{}-outf:{}".format(
            self.__class__.__name__, self.in_features, self.out_features
        )



class InitializerMul(nn.Module):


    def __init__(self, h_dim: int, phi_dim: int = 128, c_dim: int = 90):
        super().__init__()
        self.h_dim = h_dim
        self.phi_dim = phi_dim
        self.c_dim = c_dim
        self.non_lin = F.relu
        self.phi_layer = nn.Linear(
            in_features=self.phi_dim, out_features=self.h_dim, bias=False
        )
        self.c_layer = nn.Linear(
            in_features=self.c_dim, out_features=self.h_dim, bias=False
        )

    def forward(self, phi_o: Tensor, c_hat: Tensor) -> Tensor:

        return self.non_lin(self.phi_layer(phi_o)) * self.non_lin(self.c_layer(c_hat))

    def __repr__(self) -> str:
        return "name:{}-nonlin:{}".format(
            self.__class__.__name__, self.non_lin.__name__
        )


class InitializerNoClass(nn.Module):

    def __init__(self, h_dim: int, phi_dim: int = 128):
        super().__init__()
        self.h_dim = h_dim
        self.phi_dim = phi_dim
        self.non_lin = F.relu
        self.phi_layer = nn.Linear(
            in_features=self.phi_dim, out_features=self.h_dim, bias=False
        )

    def forward(self, phi_o: Tensor, c_hat: Tensor) -> Tensor:
        return self.non_lin(self.phi_layer(phi_o))

    def __repr__(self) -> str:
        return "name:{}-nonlin:{}".format(
            self.__class__.__name__, self.non_lin.__name__
        )


class InitializerNoIMG(nn.Module):

    def __init__(self, h_dim: int, phi_dim: int = 128, c_dim: int = 90):
        super().__init__()
        self.h_dim = h_dim
        self.c_dim = c_dim
        self.non_lin = F.relu
        self.c_layer = nn.Linear(
            in_features=self.c_dim, out_features=self.h_dim, bias=False
        )

    def forward(self, phi_o: Tensor, c_hat: Tensor) -> Tensor:
        return self.non_lin(self.c_layer(c_hat))

    def __repr__(self) -> str:
        return "name:{}-nonlin:{}".format(
            self.__class__.__name__, self.non_lin.__name__
        )


# ---------------------------------------------------------------------------
# Output model — unchanged from graph_networks.py
# ---------------------------------------------------------------------------

class OutputModelFirstLast(nn.Module):

    def __init__(self, h_dim: int, num_tasks: int, hidden_dim: int = 128):
        super().__init__()
        self.h_dim = h_dim
        self.num_tasks = num_tasks
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(in_features=self.h_dim * 2, out_features=self.hidden_dim)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=self.hidden_dim, out_features=self.num_tasks)

    def forward(self, h_0: Tensor, h_T: Tensor) -> Tensor:
        inp = torch.cat((h_0, h_T), dim=1)     # [B x h_dim*2]
        inp = F.relu(self.fc1.forward(inp))     # [B x hidden_dim]
        inp = self.drop(inp)
        return self.fc2.forward(inp)            # [B x num_tasks]

    def __repr__(self) -> str:
        return "name:{}-numhidden:{}".format(self.__class__.__name__, self.hidden_dim)

class GGNN(nn.Module):
    def __init__(
        self,
        initializer: nn.Module,
        aggregator: nn.Module,
        output_model: nn.Module,
        max_steps: int = 3,
        h_dim: int = 128,
        x_dim: int = 128,
        class_dim: int = 90,
    ):
        super().__init__()
        self.initializer = initializer
        self.aggregator = aggregator
        self.output_model = output_model
        self.max_steps = max_steps
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.class_dim = class_dim
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")

        # MobileNet replaces ResNet here — out_channels=128 (64 avg + 64 max)
        self.extractor = ExtractorMobileNet()
        self.propagator = nn.GRUCell(input_size=self.x_dim, hidden_size=self.h_dim)

    def forward(self, o: Tensor, c: Tensor, d: Tensor) -> Tensor:
        phi_o = self.extractor.forward(o)           # [B x h_dim]
        h_0 = self.initializer.forward(phi_o, c)    # [B x h_dim]

        h_t = h_0
        for i in range(self.max_steps):
            x_t = self.aggregator.forward(h_t, phi_o, d)
            h_t = self.propagator.forward(x_t, h_t)

        h_T = h_t
        return self.output_model.forward(h_0, h_T)

    def estimate_probability(self, o: Tensor, c: Tensor, d: Tensor) -> Tensor:
        return torch.sigmoid(self.forward(o, c, d))

    def compute_loss(self, logits: Tensor, t: Tensor, m: Tensor) -> Tensor:
        loss = logits.new_zeros(())
        for ti in range(self.output_model.num_tasks):
            if m[ti]:
                loss += self._compute_single_loss(logits[:, ti], t[:, ti])
        return loss

    def _compute_single_loss(self, logits: Tensor, t: Tensor) -> Tensor:
        return self.loss.forward(logits, t)

    def __repr__(self) -> str:
        return "name:{}-init({})-agg({})-out({})-maxstep:{}".format(
            self.__class__.__name__,
            self.initializer.__repr__(),
            self.aggregator.__repr__(),
            self.output_model.__repr__(),
            self.max_steps,
        )



class GGNNBboxNoImg(nn.Module):


    def __init__(
        self,
        initializer: nn.Module,
        aggregator: nn.Module,
        output_model: nn.Module,
        max_steps: int = 3,
        h_dim: int = 128,
        x_dim: int = 128,
        class_dim: int = 90,
    ):
        super().__init__()
        self.initializer = initializer
        self.aggregator = aggregator
        self.output_model = output_model
        self.max_steps = max_steps
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.class_dim = class_dim
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")

        # small linear — bbox [B x 4] -> [B x 16]
        self.extractor = nn.Linear(in_features=4, out_features=16)
        self.propagator = nn.GRUCell(input_size=self.x_dim, hidden_size=self.h_dim)

    def forward(self, bbox: Tensor, c: Tensor, d: Tensor) -> Tensor:
        phi_o = self.extractor.forward(bbox)        # [B x 16]
        h_0 = self.initializer.forward(phi_o, c)   # [B x h_dim]

        h_t = h_0
        for i in range(self.max_steps):
            x_t = self.aggregator.forward(h_t, phi_o, d)
            h_t = self.propagator.forward(x_t, h_t)

        h_T = h_t
        return self.output_model.forward(h_0, h_T)

    def estimate_probability(self, bbox: Tensor, c: Tensor, d: Tensor) -> Tensor:
        return torch.sigmoid(self.forward(bbox, c, d))

    def compute_loss(self, logits: Tensor, t: Tensor, m: Tensor) -> Tensor:
        loss = logits.new_zeros(())
        for ti in range(self.output_model.num_tasks):
            if m[ti]:
                loss += self._compute_single_loss(logits[:, ti], t[:, ti])
        return loss

    def _compute_single_loss(self, logits: Tensor, t: Tensor) -> Tensor:
        return self.loss.forward(logits, t)

    def __repr__(self) -> str:
        return "name:{}-init({})-agg({})-out({})-maxstep:{}".format(
            self.__class__.__name__,
            self.initializer.__repr__(),
            self.aggregator.__repr__(),
            self.output_model.__repr__(),
            self.max_steps,
        )



class GGNNDiscLoss(nn.Module):

    def __init__(
        self,
        initializer: nn.Module,
        aggregator: nn.Module,
        output_model: nn.Module,
        max_steps: int = 3,
        h_dim: int = 128,
        x_dim: int = 128,
        class_dim: int = 90,
        fusion: str = "none",
    ):
        super().__init__()
        self.initializer = initializer
        self.aggregator = aggregator
        self.output_model = output_model
        self.max_steps = max_steps
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.class_dim = class_dim
        self.fusion = fusion
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")

        self.extractor = ExtractorMobileNet()
        # ----------------------------------------------------------------------

        self.propagator = nn.GRUCell(input_size=self.x_dim, hidden_size=self.h_dim)
        self.drop = nn.Dropout(0.25)

        self.aux_fc = nn.Linear(
            self.extractor.out_channels, self.output_model.num_tasks
        )
        self.task_loss_weights = [
            1.0,  # task 1
            1.0,  # task 2
            1.0,  # task 3
            1.0,  # task 4
            1.0,  # task 5
            2.0,  # task 6
            1.0,  # task 7
            5.0,  # task 8  
            1.0,  # task 9
            1.0,  # task 10
            1.0,  # task 11
            1.0,  # task 12
            1.0,  # task 13
            1.0,  # task 14
        ]

    def forward(self, o: Tensor, c: Tensor, d: Tensor) -> Tuple[Tensor, Tensor]:

        phi_o = self.extractor.forward(o)           # [B x h_dim]  <-- was [B x 2048]
        h_0 = self.initializer.forward(phi_o, c)    # [B x h_dim]

        h_t = h_0
        for i in range(self.max_steps):
            x_t = self.aggregator.forward(h_t, phi_o, d)
            h_t = self.propagator.forward(x_t, h_t)

        h_T = h_t

        aux_logits = self.aux_fc(self.drop(phi_o))  # [B x num_tasks]

        final_logits = self.output_model.forward(h_0, h_T)  # [B x num_tasks]

        return final_logits, aux_logits

    def get_features(self, o: Tensor, c: Tensor, d: Tensor) -> Tuple[Tensor, Tensor]:

        phi_o = self.extractor.forward(o)
        h_0 = self.initializer.forward(phi_o, c)

        h_t = h_0
        for i in range(self.max_steps):
            x_t = self.aggregator.forward(h_t, phi_o, d)
            h_t = self.propagator.forward(x_t, h_t)

        h_T = h_t
        return h_0, h_T

    def estimate_probability(self, o: Tensor, c: Tensor, d: Tensor) -> Tensor:

        final_logits, aux_logits = self.forward(o, c, d)
        if self.fusion == "none":
            return torch.sigmoid(final_logits)
        elif self.fusion == "avg":
            return (torch.sigmoid(final_logits) + torch.sigmoid(aux_logits)) / 2
        else:
            raise Exception("Invalid fusion mode: {}".format(self.fusion))

    def compute_loss(
        self, logits: Tuple[Tensor, Tensor], t: Tensor, m: Tensor
    ) -> Tensor:
        loss = logits[0].new_zeros(())
        for ti in range(self.output_model.num_tasks):
            if m[ti]:
                final_logits = logits[0][:, ti]
                aux_logits   = logits[1][:, ti]
                task_loss = self._compute_single_loss(final_logits, aux_logits, t[:, ti])
                loss += self.task_loss_weights[ti] * task_loss
        return loss

    def _compute_single_loss(
        self, final_logits: Tensor, aux_logits: Tensor, t: Tensor
    ) -> Tensor:
        alpha, beta = 10, 1
        return (
            alpha * self.loss.forward(final_logits, t)
            + beta  * self.loss.forward(aux_logits,   t)
        ) / (alpha + beta)

    def __repr__(self) -> str:
        return "name:{}-init({})-agg({})-out({})-maxstep:{}".format(
            self.__class__.__name__,
            self.initializer.__repr__(),
            self.aggregator.__repr__(),
            self.output_model.__repr__(),
            self.max_steps,
        )
