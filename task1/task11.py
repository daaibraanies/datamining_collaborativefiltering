import pyspark
import findspark
from pyspark.sql import SQLContext
import json
import sys
import os.path
import time
from collections import Counter
from itertools import combinations

findspark.find()
spark_context = pyspark.SparkContext()
sql_context = SQLContext(spark_context)


def dump_file(filename, data, mode='w'):
    with open(filename, mode=mode) as f:
        f.write("{}".format(data))


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
        return 23197

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


def init_task(review_file):
    review_json = sql_context.read.json(review_file).rdd
    real_matrix = review_json.map(lambda x: (x['business_id'], x['user_id'])) \
        .groupByKey()

    uniq_users = list(real_matrix.map(lambda x: set(x[1])) \
                      .reduce(lambda x, y: x.union(y)))

    sparse_matrix = real_matrix.map(lambda x: (x[0], [uniq_users.index(user) for user in x[1]]))
    return uniq_users, sparse_matrix


def compute_bands(expected_accuracy, hash_num):
    for possible_b in range(hash_num, 1, -1):
        if hash_num % possible_b == 0:
            r = hash_num / possible_b
            if (1 / possible_b) ** (1 / r) >= expected_accuracy:
                return possible_b


def map_bands(signature, b, r):
    bands = []
    for b_i in range(b):
        bands.append((b_i, signature[b_i * r:b_i * r + r]))
    return bands

def JaccardSim(id1,id2,sparse_matrix):
    return len(set(sparse_matrix[id1]).intersection(sparse_matrix[id2]))/len(set(sparse_matrix[id1]).union(sparse_matrix[id2]))

def get_uniq_candidates(collection):
    uniq_candidates = []
    for group in collection:
        _,group_members = group
        for member in group_members:
            uniq_candidates.append(member)
    return set(uniq_candidates)


if __name__ == '__main__':
    script_start_time = time.time()
    # review_file = sys.argv[1]
    # output_file = sys.argv[2]
    review_file = '../train_review.json'
    output_file = 'out'
    uniq_users, sparse_matrix = init_task(review_file)
    sparse_matrix.persist()
    hasher = Hasher(bins_num=27351, a=9679, b=23143)
    number_of_hash = 25
    bands = 25
    bands_r = 1


    signature_matrix = sparse_matrix.map(lambda x: (x[0], create_signature(x[1], number_of_hash, hasher)))

    bands_distribution = signature_matrix.flatMap(lambda x: [(str(band_index)+str(band_hash), x[0]) for band_index, band_hash in map_bands(x[1], bands, bands_r)])\
                                         .groupByKey()\
                                         .filter(lambda x: len(x[1])>1)\
                                         .map(lambda x: (x[0],sorted(x[1])))

    candidate_pairs = bands_distribution.flatMap(lambda x: [(id1,id2) for id1,id2 in combinations(x[1],2)])

    sparse_matrix = sparse_matrix.collect()
    sparse_matrix = {key:value for key,value in sparse_matrix}

    similars = set(candidate_pairs.map(lambda x: (x[0],x[1],JaccardSim(x[0],x[1],sparse_matrix)))\
                              .filter(lambda x: x[2] >= 0.05).collect())

    with open(output_file,'w') as output:
        for pair in similars:
            output.write('{'+'"b1":"'+pair[0]+'","b2":"'+pair[1]+'","sim":'+str(pair[2])+'}\n')


    print("Duration: {} s.".format(time.time() - script_start_time))