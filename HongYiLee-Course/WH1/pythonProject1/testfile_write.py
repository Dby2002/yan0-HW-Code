import csv
from model_init import *
from Data_get import *
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_set = covid19_Set(filePath='./data/covid.test.csv',mode='test')
test_loador  = torch.utils.data.DataLoader(test_set,batch_size=32,shuffle=False)

file_path_row = './data/covid.test.csv'
file_path_new = './data/covid.test_w.csv'


HW1_M_trained = torch.load('./model_save/HW1.path')
HW1_M_trained = HW1_M_trained.to(device)




def write_test_file():
    data_pred = [0]
    rows_r = []

    HW1_M_trained.eval()

    for data in test_loador:
        xs = data.to(device)

        outputs = HW1_M_trained(xs)

        list_temp = outputs.cpu().data.numpy().tolist()

        list_t = [float(it[0]) for it in list_temp]

        data_pred.extend(list_t)

    with open(file_path_row,'r') as file_r:
        reader = csv.reader(file_r)

        for rows in reader:
            rows_r.append(rows)


    step = 0
    with open(file_path_new,'w',newline='') as file_w:
        writer = csv.writer(file_w)

        for row in rows_r:
            if step ==0:
                row.append('tested_positive')
                writer.writerow(row)
                step += 1
                continue

            row.append(data_pred[step])

            writer.writerow(row)
            step += 1


if __name__ == '__main__':
    write_test_file()