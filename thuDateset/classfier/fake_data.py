import numpy as np
confuse_matrix2_path = r"/media/jsl/ubuntu/result/nwpu/vgg/confuse_matrix 2.txt"
acc2_path = r"/media/jsl/ubuntu/result/nwpu/vgg/acc 2.txt"
confuse_matrix_new_path = r"/media/jsl/ubuntu/result/nwpu/vgg/confuse_matrix_new.txt"
# with open(confuse_matrix2_path, 'r') as f:
#     line = f.readline()
#     while line is not "":
#         print(line)
#         line = f.readline()

# oa = 0
# count = 0
# with open(acc2_path, 'r') as f:
#     line = f.readline()
#     while line is not "":
#         acc = float(line.split(':')[1])
#         print(count, acc)
#         oa += acc
#         count += 1
#         line = f.readline()
# print("OA:", oa/count)
# a = a.split("' '")
#
#
# # print(a)
# b = np.zeros((45,),dtype='int')
# for item in a:
#     item = int(item)
#     b[item] += 1
# str_line = ""
# for item in b:
#     str_line += str(item)+','
# print(str_line)

count = 0
oa = 0
with open(confuse_matrix_new_path, 'r') as f:
    line = f.readline()
    while line is not "":
        num = 0
        line = line.strip().split(',')
        for item in line:
            try:

                num += int(item)
            except:
                pass
        acc = int(line[count])/num
        print(count,":",num,"; acc:", acc)
        oa += acc
        count += 1
        line = f.readline()

print("count:",count)
print("OA:", oa/45)