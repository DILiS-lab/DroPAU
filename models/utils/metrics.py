import torch


class BetaGNLLLoss(torch.nn.Module):
    def __init__(self, beta=0.5, reduction="mean"):
        super().__init__()
        self.beta = beta
        self.gnll = torch.nn.GaussianNLLLoss(reduction="none")
        self.reduction = reduction

    def forward(self, input, target, var):
        loss = self.gnll(input, target, var)

        if self.beta > 0:
            loss = loss * var.detach() ** self.beta
        return (
            torch.mean(loss)
            if self.reduction == "mean"
            else torch.sum(loss)
            if self.reduction == "sum"
            else loss
        )
