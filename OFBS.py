import sys
sys.path.append("./utils/")
from shared import main, cal_precision, cal_recall, cal_fbs
import torch



## fbeta-score
def objective(s, y):
    return -cal_fbs(s, y)

## dummy constraint
def metric_constr(s, y, alpha):
    return torch.tensor(0)


if __name__ == "__main__":
    main(task="OFBS", objective=objective, metric_constr=metric_constr)
