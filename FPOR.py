import sys
sys.path.append("./utils/")

from shared import main, cal_precision, cal_recall
import torch

def objective(s, y):
    # return -(s.T@y) / (torch.sum(s) + 1e-9)
    return -cal_recall(s, y)

def metric_constr(s, y, alpha):
    return torch.maximum(torch.tensor(0), alpha - cal_precision(s, y))


if __name__ == "__main__":
    main(task="FPOR", objective=objective, metric_constr=metric_constr)
