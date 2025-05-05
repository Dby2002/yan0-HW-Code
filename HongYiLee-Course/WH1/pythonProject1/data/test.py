import csv


def _get_item(filePath, mode):
    if (mode == 'train'):
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
                y.append(float(row[-1]))

                step += 1
        return x, y

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
                y.append(float(row[-1]))

                step += 1
        return x, y

    else:
        x = []

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
        return x, None



filename = './covid.train.csv'

# with open(filename, "r") as csvfile:
#     csvreader = csv.reader(csvfile)
#
#     step = 0
#     # 遍历csvreader对象的每一行内容并输出
#     for row in csvreader:
#         print(row)
#         print(type(row))
#         step+=1
#         if step >= 10 :
#             break

if __name__ == '__main__':
    _get_item(filename,'train')
