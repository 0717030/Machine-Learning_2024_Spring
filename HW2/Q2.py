import argparse
import numpy as np
def getBinomial(total_result,one,zero,prob):
    count_smaller = min(one,zero)
    binomial = (one**one)*(zero**zero)/(total_result**total_result)

    mul = total_result
    div = count_smaller
    for i in range(count_smaller):
        binomial *= mul/div
        mul -= 1
        div -= 1

    return binomial



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Homework#02')
    parser.add_argument('--a', type=int, default=0, help="Parameter a for the initial beta prior")
    parser.add_argument('--b', type=int, default=0, help="Parameter b for the initial beta prior ")
    parser.add_argument('--file_path', type=str, default="testfile.txt", help="Online learning data path")

    setting = parser.parse_args()
    a = setting.a
    b = setting.b

    file = open(setting.file_path, 'r')
    count = 0
    one_prob = 1/2
    for line in file.readlines():
        count += 1
        if line != "\n":
            line = line.rstrip('\n')
            one = line.count('1')
            zero = line.count('0')
            likelihood = getBinomial(len(line),one,zero,one_prob)
            print("case",count,": ",line)
            print("Likelihood: ",likelihood)
            print("Beta prior:     a = ",a," b = ",b)
            a += one
            b += zero
            print("Beta posterior: a = ",a," b = ",b)
            print("")

    file.close()