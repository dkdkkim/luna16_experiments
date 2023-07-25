import torch
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

class LabelSmoothingCrossEntropyLoss(torch.nn.Module): 
    def __init__(self, classes, smoothing=0.0, dim=-1, weight = None):
        super(LabelSmoothingCrossEntropyLoss, self).__init__() 
        self.confidence = 1.0 - smoothing 
        self.smoothing = smoothing 
        self.cls = classes 
        self.dim = dim
        self.weight = weight.cuda()
    
    def forward(self, pred, target): 
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred) 
            true_dist.fill_(self.smoothing / (self.cls - 1)) 
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_pred = -true_dist * pred
        if self.weight is not None:
            if len(true_pred.size()) == 5:
                true_pred = true_pred * self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
            elif len(true_pred.size()) == 2:
                true_pred = true_pred * self.weight.unsqueeze(0)
        return torch.mean(torch.sum(true_pred, dim=self.dim))

class RegressionLoss(torch.nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.l1 = torch.nn.L1Loss()
    def forward(self, pred, target):
        return self.mse(pred, target) + self.l1(pred, target)

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = torch.nn.functional.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class FocalLossWrap(torch.nn.Module):
    def __init__(self, loss_fn, gamma=0, alpha=None, size_average=True):
        super(FocalLossWrap, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.loss_fn = loss_fn

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = torch.nn.functional.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        loss = self.loss_fn(input, torch.squeeze(target, -1))
        
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = (1-pt)**self.gamma * loss ## value from loss_fn is already negative
        if self.size_average: return loss.mean()
        else: return loss.sum()

class AvULoss(torch.nn.Module):
    """
    Calculates Accuracy vs Uncertainty Loss of a model.
    The input to this loss is logits from Monte_carlo sampling of the model, true labels,
    and the type of uncertainty to be used [0: predictive uncertainty (default); 
    1: model uncertainty]
    reference: https://github.com/IntelLabs/AVUC/blob/main/src/avuc_loss.py
    """
    def __init__(self, beta=1):
        super(AvULoss, self).__init__()
        self.beta = beta
        self.eps = 1e-10

    def entropy(self, prob):
        return -1 * torch.sum(prob * torch.log(prob + self.eps), dim=-1)

    def expected_entropy(self, mc_preds):
        return torch.mean(self.entropy(mc_preds), dim=0)

    def predictive_uncertainty(self, mc_preds):
        """
        Compute the entropy of the mean of the predictive distribution
        obtained from Monte Carlo sampling.
        """
        return self.entropy(torch.mean(mc_preds, dim=0))

    def model_uncertainty(self, mc_preds):
        """
        Compute the difference between the entropy of the mean of the
        predictive distribution and the mean of the entropy.
        """
        return self.entropy(torch.mean(
            mc_preds, dim=0)) - self.expected_entropy(mc_preds)

    def accuracy_vs_uncertainty(self, prediction, true_label, uncertainty,
                                optimal_threshold):
        # number of samples accurate and certain
        n_ac = torch.zeros(1, device=true_label.device)
        # number of samples inaccurate and certain
        n_ic = torch.zeros(1, device=true_label.device)
        # number of samples accurate and uncertain
        n_au = torch.zeros(1, device=true_label.device)
        # number of samples inaccurate and uncertain
        n_iu = torch.zeros(1, device=true_label.device)

        avu = torch.ones(1, device=true_label.device)
        avu.requires_grad_(True)

        for i in range(len(true_label)):
            if ((true_label[i].item() == prediction[i].item())
                    and uncertainty[i].item() <= optimal_threshold):
                """ accurate and certain """
                n_ac += 1
            elif ((true_label[i].item() == prediction[i].item())
                  and uncertainty[i].item() > optimal_threshold):
                """ accurate and uncertain """
                n_au += 1
            elif ((true_label[i].item() != prediction[i].item())
                  and uncertainty[i].item() <= optimal_threshold):
                """ inaccurate and certain """
                n_ic += 1
            elif ((true_label[i].item() != prediction[i].item())
                  and uncertainty[i].item() > optimal_threshold):
                """ inaccurate and uncertain """
                n_iu += 1

        print('n_ac: ', n_ac, ' ; n_au: ', n_au, ' ; n_ic: ', n_ic, ' ;n_iu: ',
              n_iu)
        avu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)

        return avu

    def forward(self, logits, labels, optimal_uncertainty_threshold, type=0):

        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, 1)

        if type == 0:
            unc = self.entropy(probs)
        else:
            unc = self.model_uncertainty(probs)

        unc_th = torch.tensor(optimal_uncertainty_threshold,
                              device=logits.device)

        n_ac = torch.zeros(
            1, device=logits.device)  # number of samples accurate and certain
        n_ic = torch.zeros(
            1,
            device=logits.device)  # number of samples inaccurate and certain
        n_au = torch.zeros(
            1,
            device=logits.device)  # number of samples accurate and uncertain
        n_iu = torch.zeros(
            1,
            device=logits.device)  # number of samples inaccurate and uncertain

        avu = torch.ones(1, device=logits.device)
        avu_loss = torch.zeros(1, device=logits.device)

        for i in range(len(labels)):
            if ((labels[i].item() == predictions[i].item())
                    and unc[i].item() <= unc_th.item()):
                """ accurate and certain """
                n_ac += confidences[i] * (1 - torch.tanh(unc[i]))
            elif ((labels[i].item() == predictions[i].item())
                  and unc[i].item() > unc_th.item()):
                """ accurate and uncertain """
                n_au += confidences[i] * torch.tanh(unc[i])
            elif ((labels[i].item() != predictions[i].item())
                  and unc[i].item() <= unc_th.item()):
                """ inaccurate and certain """
                n_ic += (1 - confidences[i]) * (1 - torch.tanh(unc[i]))
            elif ((labels[i].item() != predictions[i].item())
                  and unc[i].item() > unc_th.item()):
                """ inaccurate and uncertain """
                n_iu += (1 - confidences[i]) * torch.tanh(unc[i])

        avu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + self.eps)
        p_ac = (n_ac) / (n_ac + n_ic)
        p_ui = (n_iu) / (n_iu + n_ic)
        #print('Actual AvU: ', self.accuracy_vs_uncertainty(predictions, labels, uncertainty, optimal_threshold))
        avu_loss = -1 * self.beta * torch.log(avu + self.eps)
        return avu_loss.sum()

def generate_loss_function(option):
    if option.loss_fn == 'focal':
        criterion = FocalLoss(gamma=option.loss_gamma, alpha=option.loss_alpha)
    elif option.loss_fn == 'ce':
        if option.num_classes == 2:
            criterion = torch.nn.CrossEntropyLoss(weight = torch.Tensor([1.0, option.loss_weight]).cuda(option.gpu), label_smoothing=option.label_smoothing).cuda(option.gpu)
        elif option.num_classes == 3:    
            criterion = torch.nn.CrossEntropyLoss(weight = torch.Tensor([1.0, option.loss_weight, option.loss_weight]).cuda(option.gpu), label_smoothing=option.label_smoothing).cuda(option.gpu)
    return criterion
