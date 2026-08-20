import torch
import torch.nn as nn

INPUT_DIM = 137
HIDDEN_DIM = 256


class Discriminator(nn.Module):
    """MLP discriminator, D_phi(s, a, c) -> P(expert | input).

    Per methodology 3.3.1: two hidden layers of 256 units, ReLU activations,
    sigmoid output head. Context is concatenated at the input layer (it is
    already dims 135-136 of the 137-dim feature vector) -- the canonical
    conditional-GAN conditioning mechanism.

    forward() returns the PRE-SIGMOID LOGIT, not the probability. This is
    deliberate and load-bearing:
      - The R1 gradient penalty (3.3.2) is specified on the pre-sigmoid logit.
      - BCEWithLogitsLoss is numerically stable in a way that
        sigmoid-then-BCE is not.
    Call probability() when you actually need P(expert | input) -- e.g. the
    counterfactual swap gate, which is defined as a shift in probability.
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """x: (batch, 137) -> (batch,) pre-sigmoid logits."""
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"expected (batch, {self.input_dim}), got {tuple(x.shape)}")
        return self.net(x).squeeze(-1)

    def probability(self, x):
        """x: (batch, 137) -> (batch,) P(expert | input) in (0, 1)."""
        return torch.sigmoid(self.forward(x))