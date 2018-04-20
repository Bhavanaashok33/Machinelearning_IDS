import random
import socket
import re
from binascii import hexlify

protocols = {
    'sip':0, 
    'http':1,
    'rdp':2, 
    'snmp':3, 
    'smtp':4, 
    'ssl':5, 
    'other':6, 
    'ssh':7, 
    'dns':8, 
    'smtp,ssl':9,
    'dhcp':10
}

flags = {
    'OTH':0, 
    'SHR':1, 
    'RSTRH':2, 
    'RSTR':3, 
    'S3':4, 
    'S2':5, 
    'S1':6, 
    'S0':7, 
    'RSTOS0':8, 
    'REJ':9, 
    'SH':10, 
    'RSTO':11, 
    'SF':12
}

connection = {
    'udp':0, 
    'icmp':1, 
    'tcp':2
}

def randomSplit(split, array):
    random.shuffle(array)
    return array[:int(split[0]*len(array))], array[int(split[1]*len(array)):]

def find_categories(data, indice):
    categories = set()
    for row in data:
        categories.add(row[indice])
    return categories

def vectorize(x):
    x[1] = str(protocols[x[1]])
    x[13] = str(flags[x[13]])
    # x[17] = str(ipv6_to_int(x[17]))
    # x[19] = str(ipv6_to_int(x[19]))
    x[16] = str(time_to_decimal(x[16]))
    x[17] = str(connection[x[17]])
    for idx in range(len(x) - 1):
        x[idx] = "".join(re.findall('\d+', x[idx]))
    return x


# def ipv6_to_int(ipv6_addr):
#     return int(hexlify(socket.inet_pton(socket.AF_INET6, ipv6_addr)), 16)

def time_to_decimal(time):
    elements = time.split(":")
    return int(elements[0])*3600 + int(elements[1])*60 + int(elements[2])

mode = lambda list: max(set(list), key=list.count)