import os
import random

from train_entrance import *
from Data_get import *
from model_init import *


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
#################################################################
seed_everything(0)


#设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('-----device----- '+'cuda' if torch.cuda.is_available() else 'cpu')

#数据集准备
train_set = covid19_Set(filePath='./data/covid.train.csv',mode='train')
val_set = covid19_Set(filePath='./data/covid.train.csv',mode='val')
test_set = covid19_Set(filePath='./data/covid.test.csv',mode='test')

train_loador = torch.utils.data.DataLoader(train_set,batch_size=128,shuffle=True)
val_loador = torch.utils.data.DataLoader(val_set,batch_size=128,shuffle=True)
test_loador  = torch.utils.data.DataLoader(test_set,batch_size=32,shuffle=False)

HW1_M = HW1_Model(train_set.x.size(1))
HW1_M = HW1_M.to(device)


#参数设置
_loss = torch.nn.MSELoss(reduction='mean')
_loss = _loss.to(device)

optim = torch.optim.SGD(HW1_M.parameters(),lr=1e-5,momentum=0.9)
# optim = nn.MSELoss(reduction='mean')

_parameters = {
    'device':device,
    'train_loader':train_loador,
    'train_set':train_set,
    'test_loader':test_loador,
    'test_set':test_set,
    'val_loader':val_loador,
    'val_set':val_set,
    'model':HW1_M,
    'loss':_loss,
    'optim':optim,


}


if __name__ == '__main__':

    train_start(_parameters)





