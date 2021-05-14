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
        return several_primes[-1]

    def h1(self, x):
        return (self.a * x + self.b) % self.m

    def h2(self, x):
        return ((self.a * x + self.b) % self.p) % self.m

    def dyn_hash(self, x, i):
        return ((i * self.h1(x) + i * self.h2(x) + i * i) % self.p) % self.m

    def get_hash_part(self,rows,b_i,r):
        hash_part = []
        for offset in range(r):
            i = b_i*r+offset
            all_hashes = []
            for row in rows:
                all_hashes.append(self.dyn_hash(row,i))
            hash_part.append(min(all_hashes))
        return str(b_i)+str(hash_part)

def init_task(review_file):
    review_json = sql_context.read.json(review_file).rdd
    user_business = review_json.map(lambda x: (x['user_id'],x['business_id']))
    uniq_users = user_business.distinct().map(lambda x:x[0]).collect()

    sparse_matrix = user_business.map(lambda x: (x[1], uniq_users.index(x[0])))\
                                 .groupByKey()
    return sparse_matrix

def map_bands(signature, b, r):
    bands = []
    for b_i in range(b):
        bands.append((b_i, signature[b_i * r:b_i * r + r]))
    return bands

def JaccardSim(id1,id2,sparse_matrix):
    return len(set(sparse_matrix[id1]).intersection(sparse_matrix[id2]))/len(set(sparse_matrix[id1]).union(sparse_matrix[id2]))

if __name__ == '__main__':
    script_start_time = time.time()
    # review_file = sys.argv[1]
    # output_file = sys.argv[2]
    review_file = '../train_review.json'
    output_file = 'out'
    sparse_matrix = init_task(review_file)
    sparse_matrix_collection = {key: value for key, value in sparse_matrix.collect()}
    hasher = Hasher(bins_num=28000, a=677, b=23143)
    number_of_hash = 100
    bands = 50
    bands_r = int(number_of_hash / bands)

    signature_matrix = sparse_matrix.map(lambda x: (x[0], create_signature(x[1], number_of_hash, hasher)))

    bands_distribution = signature_matrix.flatMap(lambda x: [(band_index, (band_hash, x[0])) for band_index, band_hash in map_bands(x[1], bands, bands_r)]) \
        .partitionBy(bands)

    with open(output_file,'w') as output:
        for pair in similars:
            output.write('{'+'"b1":"'+pair[0]+'","b2":"'+pair[1]+'","sim":'+str(pair[2])+'}\n')


    print("Duration: {} s.".format(time.time() - script_start_time))