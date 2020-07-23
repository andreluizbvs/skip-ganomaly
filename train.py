"""
TRAIN SKIP/GANOMALY

. Example: Run the following command from the terminal.
    run train.py                                    \
        --model <skipganomaly, ganomaly>            \
        --dataset cifar10                           \
        --abnormal_class airplane                   \
        --display                                   \
"""

##
# LIBRARIES

from options import Options
from lib.data.dataloader import load_data
from lib.models import load_model

##
def main():
    """ Training
    """
    opt = Options().parse()
    data = load_data(opt)
    model = load_model(opt, data)
    if opt.phase == 'train':
        model.train()
    elif opt.phase == 'test':
        model.test_best_weights()
    elif opt.phase == 'demo':
        model.demo(plot_hist=True,is_best=True)

if __name__ == '__main__':
    main()
