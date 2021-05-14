import pyspark
import findspark
from pyspark.sql import SQLContext,SparkSession
import json
import sys
import os.path
import time
from itertools import chain
import math
import functools
import heapq

from collections import Counter
from itertools import combinations

findspark.find()
spark_context = pyspark.SparkContext()
"""spark_context =  SparkSession \
    .builder \
    .master('local')\
    .appName("Task 2 HW3") \
    .getOrCreate()"""
sql_context = SQLContext(spark_context)

class Hasher:
    def __init__(self, bins_num, a, b):
        self.a = a
        self.b = b
        self.m = bins_num
        self.p = self.set_prime()

    def set_prime(self):
        several_primes = [367, 677, 967, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223,
                          2687, 2689, 2693, 2699, 7793, 7817, 7823, 7829, 7841, 9643, 9649, 9661, 9677,
                          9679, 11489, 11491, 11497, 11503, 11519, 13399, 13411, 13417, 13421, 13441,
                          23143, 23159, 23167, 23173, 23189, 23197, 27241, 27253, 27259, 27271, 27277,
                          35507, 35509, 35521, 35527, 43963, 43969, 43973, 54767, 54773, 54779, 63527,
                          63533, 63541, 92251, 92269, 92297, 502687]
        return 502687

    def h1(self, x):
        return (self.a * x + self.b) % self.m

    def h2(self, x):
        return ((self.a * x + self.b) % self.p) % self.m

    def dyn_hash(self, x, i):
        return ((i * self.h1(x) + i * self.h2(x) + i * i) % self.p) % self.m

    def get_min_dynamic_hash(self, i, rows):
        current_min_hash = float('inf')
        for row in rows:
            row_hash = self.dyn_hash(row, i)
            if row_hash < current_min_hash:
                current_min_hash = row_hash
        return current_min_hash

def create_signature(rows, hash_num, hasher):
    columns_signature = []
    for hash_index in range(hash_num):
        columns_signature.append(hasher.get_min_dynamic_hash(hash_index, rows))
    return columns_signature

def dump_file(filename, data, mode='w'):
    with open(filename, mode=mode) as f:
        f.write("{}".format(data))

def get_users_businesses_wordmap(model):
    with open(model, 'r',encoding='utf-8') as outfile:
        data = outfile.readlines()
        businesses = json.loads(data[0])
        users = json.loads(data[1])

    businesses = dict(businesses['bprofile'])
    users = dict(users['uprofile'])

    return businesses,users

def map_bands(signature, b, r):
    bands = []
    for b_i in range(b):
        bands.append(hash(tuple(signature[b_i * r:b_i * r + r]+[b_i])))
    return bands

def is_uniq_pair(hastable,user_id,business_id):
    key = (user_id,business_id)
    try:
        exist = hastable[key]
        return False
    except KeyError:
        hastable[key] = 1
        return True

def get_test_data(test_file):
    data = []
    busn = []
    with open(test_file, 'r', encoding='utf-8') as outfile:
        for line in outfile.readlines():
            entry = json.loads(line)
            data.append(entry['user_id'])
            busn.append(entry['business_id'])
    return data,busn

def remove_useless_entires(user_profiles,test_data):
    clean = {}
    for key in user_profiles.keys():
        if key in test_data:
            clean[key] = user_profiles[key]

    return clean

def cosine_Sim(user_id,business_id,user_profiles,business_profiles):
    user_vector = user_profiles[user_id]
    business_vector = business_profiles[business_id]
    numerator = len(set(user_vector).intersection(business_vector))
    denimenator = len(user_vector)
    return numerator/denimenator

def cosineSim(user_id,business_id,user_profiles,business_profiles):
    user_vector = user_profiles[user_id]
    business_vector = business_profiles[business_id]
    if len(business_vector) != len(user_vector):
        if len(user_vector) < len(business_vector):
            while len(user_vector) < len(business_vector):
                user_vector.append(0)
        else:
            while len(business_vector) < len(user_vector):
                business_vector.append(0)
    numerator = sum([user_vector[i]*business_vector[i] for i in range(len(user_vector))])
    a_sqrt = sum([user_vector[i]**2 for i in range(len(user_vector))])
    b_sqrt = sum([business_vector[i]**2 for i in range(len(business_vector))])

    return numerator/(math.sqrt(a_sqrt)*math.sqrt(b_sqrt))

def binary_cos_similarity(user_id,business_id,user_profiles,business_profiles):
    user_vector = user_profiles[user_id]
    business_vector = business_profiles[business_id]



    return len(set(user_vector).intersection(business_vector))/((len(user_vector)**0.5)*(len(business_vector)**0.5))


if __name__ == '__main__':
    start_time = time.time()
    model_file = 'task2.model'
    output_file = 'out'
    test_file = '../test_review.json'
    business_profiles,\
    user_profiles = get_users_businesses_wordmap(model_file)

    pair_similarities = sql_context.read.json(test_file).rdd\
                                .map(lambda x:(x['user_id'],x['business_id']))\
                                .filter(lambda x: x[0] in user_profiles.keys() and x[1] in business_profiles.keys())\
                                .map(lambda x:(x[0],(x[1],binary_cos_similarity(x[0],x[1],user_profiles,business_profiles)))) \
                                .filter(lambda x: x[1][1] >= 0.01) \
                                .collect()
    #
    with open(output_file,'w') as output:
        for pair in pair_similarities:
            output.write('{'+'"user_id":"'+pair[0]+'","business_id":"'+pair[1][0]+'","sim":'+str(pair[1][1])+'}\n')


    print("Duration: {:.2f}".format(time.time()-start_time))
