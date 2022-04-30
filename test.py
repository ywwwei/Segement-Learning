import torch
from torch import nn

# from models import ContrastiveCriterion

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a batch of square matrix
    # input x: B * n * m
    B, n, m = x.shape
    assert n == m
    return x.view(B, -1)[:, :-1].view(B, n - 1, n + 1)[:, :, 1:].flatten()

class ContrastiveCriterion(nn.Module):
    """ This class computes the contrastive loss.
    """
    def __init__(self, lambd):
        """ Create the criterion.
        Parameters:
            lambd: hyper-parameter trading off the importance of invariance term and redundance term in the loss.
        """
        super().__init__()
        self.lambd = lambd
    
    def forward(self, outputs):
        """This performs the loss computation.
        Parameters:
            outputs: tensors of dimension (B, 2, N, d).
                B: batch size.
                2: one pair of images.
                N: number of queries for each image.
                d: dimension of query output.
        """
        # gather features for two images and normalize
        norm_out = outputs.pow(2).sum(keepdim = True, dim=3).sqrt()
        outputs = outputs / norm_out
        z1 = outputs[:,0] 
        z2 = outputs[:,1]

        # compute similarity matrix
        c = torch.matmul(z1, torch.transpose(z2, dim0=1, dim1=2))

        # sum the cross-correlation matrix between all gpus
        c.div_(outputs.shape[0])
        # torch.distributed.all_reduce(c)
        on_diag = torch.diagonal(c, dim1 = 1, dim2 = 2).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss
 
if __name__ == '__main__':
    # input
    outputs = torch.rand(2,3,4,4)
    # call function
    criterion = ContrastiveCriterion(lambd = 0.5)
    print(criterion(outputs))

    # output
