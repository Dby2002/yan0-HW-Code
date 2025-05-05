import csv
import torch
import torchvision
import torch.nn as nn
import numpy as np

# transForm = torchvision.transforms.Compose([torchvision .transforms.ToTensor()])


class covid19_Set(torch.utils.data.Dataset):
    def __init__(self,filePath,mode):
        if mode not in ['train','val','test']:
            print('模式错误')
            return

        x,y = self._get_item(filePath,mode)

        self.x = torch.from_numpy(np.array(x)).float()
        if y == None :
            self.y=None
        else:
            self.y = torch.from_numpy(np.array(y)).float()




    def __getitem__(self, idx):
        if(self.y == None):
            return self.x[idx]

        else:
            return self.x[idx],self.y[idx]


    def __len__(self):
        return len(self.x)


    def _get_item(self,filePath,mode):
        if( mode == 'train'):
            x = []
            y = []
            with open(filePath) as csvFile:
                csvreader = csv.reader(csvFile)

                step = 0
                for row in csvreader:
                    if step == 0 or step % 10 == 0:
                        step += 1
                        continue

                    row_ = row[1:-1]

                    list_Temp = [float(it) for it in row_]
                    x.append(list_Temp)
                    y_tmp = []
                    y_tmp.append(float(row[-1]))
                    y.append(y_tmp)

                    step += 1
            return x,y

        elif mode == 'val':
            x = []
            y = []
            with open(filePath) as csvFile:
                csvreader = csv.reader(csvFile)

                step = 0
                for row in csvreader:
                    if step == 0 or step % 10 != 0:
                        step += 1
                        continue

                    row_ = row[1:-1]

                    list_Temp = [float(it) for it in row_]
                    x.append(list_Temp)
                    y_tmp = []
                    y_tmp.append(float(row[-1]))
                    y.append(y_tmp)

                    step += 1
            return x, y

        else:
            x=[]

            with open(filePath) as csvFile:
                csvreader = csv.reader(csvFile)

                step = 0
                for row in csvreader:
                    if (step == 0):
                        step += 1
                        continue

                    row_ = row[1:]

                    list_Temp = [float(it) for it in row_]
                    x.append(list_Temp)
            return x,None







if __name__ == '__main__':
    dataSet = covid19_Set('./data/covid.train.csv',mode='train')
    x = dataSet.x
    print(len(dataSet.x[0]))
    print('\n \n')
    print(type(dataSet.y))