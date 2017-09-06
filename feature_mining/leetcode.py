#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/2 15:19
# @Author  : sunday
# @Site    : 
# @File    : leetcode.py
# @Software: PyCharm

def main():
    lines = raw_input()
    inputs = []
    for i in range(int(lines)):
        input_line = raw_input()
        input_list = input_line.split(" ")
        temp_input = [input_list[0],int(input_list[1])]
        inputs.append(temp_input)

    inputs.sort(key=lambda x: x[1],reverse=False)
    sorted_by_value = inputs
    inputs.sort(key=lambda x: x[0])
    sorted_by_name = inputs
    # sorted(inputs, key=itemgetter(1))
    # inputs.sort(key= lambda x:x[1],x[0],reverse=True)
    #


    output = []
    space = ""
    count= 0
    last_value = 0
    for value_sorted in sorted_by_value:
        if (value_sorted[1] ==  -1):
            output.append(value_sorted[0])
            last_value = value_sorted[1]

            continue
        elif value_sorted[1] == last_value:
            continue
        else:
            for  name_sorrted in sorted_by_name:
                if(name_sorrted[1]==value_sorted[1]):

                    if(value_sorted[1]>last_value ):
                        if(last_value ==-1):
                            output.append(space + "|-- " + name_sorrted[0])
                            count=count+1
                            last_value = name_sorrted[1]
                        else:
                            temp_str = output[count]
                            temp_str.replace('|','`')
                            output[count] = temp_str
                            space = space + "    "
                            output.append(space + "|-- " + name_sorrted[0])
                            count = count + 1
                            last_value = name_sorrted[1]
                    elif(value_sorted[1]==last_value ):
                        output.append(space + "|-- " + name_sorrted[0])
                        count = count + 1
                        last_value = name_sorrted[1]

    temp_str = output[count]
    str(temp_str).replace("|","`")
    output[count] = temp_str
    print "\n".join(output)



main()