import torch
from torch.nn.functional import normalize

class MeanShift_loss(torch.nn.Module):
    def __init__(self, use_loss):
        super(MeanShift_loss, self).__init__()
        self.use_loss = use_loss

    def forward(self, pred_pos, true_pos, weight=None, dummy=None, print_flag=False):
        if len(true_pos) == 0:
            return 0.#, 0., 0.
        dists = torch.cdist(true_pos.float() / 15., pred_pos.float() / 15.)
        mins, _ = torch.min(dists, dim=1)
        mins_seeds, _ = torch.min(dists, dim=0)
        
        for pos in true_pos:
            if print_flag:
                print(pos, pred_pos[torch.argmin(torch.cdist(pos.unsqueeze(0), pred_pos))], "true_pos and pred_pos")
        loss = torch.mean(mins) #+ .5 * torch.max(mins)
        # loss = 0.
        # loss = torch.max(mins)
        loss_seeds = torch.mean(mins_seeds) #+ .5 * torch.max(mins_seeds)
        # loss_seeds = torch.max(mins_seeds)
        # return 0., mins_seeds
        just_in_case_losses = (loss, loss_seeds)
        if print_flag:
            print(loss, loss_seeds, "losses")
        if not self.use_loss:
            return 0.#, mins_seeds, just_in_case_losses
        # return loss_seeds, mins_seeds, just_in_case_losses
        return loss + loss_seeds#, mins_seeds, just_in_case_losses

class MeanShift_loss_directional(torch.nn.Module):
    def __init__(self):
        super(MeanShift_loss_directional, self).__init__()

    def forward(self, true_pos, pred_pos, seed_points):
        dists = torch.cdist(true_pos.float(), seed_points.float())
        closest_centers = torch.argmin(dists, dim=1)
        closest_centers = true_pos[0, closest_centers]
        true_directions = closest_centers - seed_points
        true_directions = normalize(true_directions, dim=2)
        pred_directions = pred_pos - seed_points
        pred_directions = normalize(pred_directions, dim=2)
        dot_prod = pred_directions[0] * true_directions[0]  # values in range -1:1 --> 1 is good, -1 is bad
        dot_prod = torch.sum(dot_prod, axis=1)
        dot_prod = 1. - dot_prod
        dot_bkp = dot_prod.clone().detach().cpu()
        dists_pred = torch.cdist(true_pos.float(), pred_pos.float())
        dists_pred, _ = torch.min(dists_pred, dim=1)
        dists_pred = dists_pred[0]
        #dot_prod *= dists_pred
        dot_prod = torch.mean(dot_prod)

        loss = dot_prod #+ torch.mean(dists_pred)

        return loss, dot_bkp.unsqueeze(0)