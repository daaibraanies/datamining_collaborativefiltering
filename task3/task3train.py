import pyspark
import findspark
from pyspark.sql import SQLContext,SparkSession
import json
import sys
import os.path
import time
import statistics
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

def dump_file(filename, data, mode='w'):
    with open(filename, mode=mode) as f:
        f.write("{}".format(data))

def JaccardSim(id1,id2,sparse_matrix):
    return len(set(sparse_matrix[id1]).intersection(sparse_matrix[id2]))/len(set(sparse_matrix[id1]).union(sparse_matrix[id2]))

def pearson_correlation(business_id1,business_id2,corated,raitings):
    coraters = corated[business_id1].intersection(corated[business_id2])
    avg_id1 = statistics.mean([raitings[(business_id1,uid)] for uid in coraters])
    avg_id2 = statistics.mean([raitings[(business_id2, uid)] for uid in coraters])
    numerator,denom_id1,denom_id2=0,0,0
    for uid in coraters:
        arg1 = raitings[(business_id1,uid)]-avg_id1
        arg2 = raitings[(business_id2,uid)]-avg_id2
        numerator += (arg1)*(arg2)
        denom_id1 += arg1**2
        denom_id2 += arg2**2
    return numerator/(math.sqrt(denom_id1)*math.sqrt(denom_id2)+1e-10)

def pearson_correlation_case_2(user_id1,user_id2,corated,raitings):
    coraters = corated[user_id1].intersection(corated[user_id2])
    avg_id1 = statistics.mean([raitings[(user_id1,bid)] for bid in coraters])
    avg_id2 = statistics.mean([raitings[(user_id2, bid)] for bid in coraters])
    numerator,denom_id1,denom_id2=0,0,0
    for bid in coraters:
        arg1 = raitings[(user_id1,bid)]-avg_id1
        arg2 = raitings[(user_id2,bid)]-avg_id2
        numerator += (arg1)*(arg2)
        denom_id1 += arg1**2
        denom_id2 += arg2**2
    return numerator/(math.sqrt(denom_id1)*math.sqrt(denom_id2)+1e-10)

def is_uniq_pair(key,dictionary):
    if key in dictionary.keys():
        return False
    else:
        dictionary[key] = 1
        return True

if __name__ == '__main__':
    start_time = time.time()
    review_file = '../train_review.json'
    output_file = 'task3user.model'
    candidate_pairs = {}
    case = 2
    if case == 1:
        businesses_rdd = spark_context.textFile(review_file).map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], x['user_id']))\
                                                     .groupByKey()\
                                                     .map(lambda x:(x[0],set(x[1])))\
                                                     .filter(lambda x:len(x[1])>=3)
        businesses_rdd.persist()
        distinct_businesses = businesses_rdd.map(lambda x:x[0]).distinct().collect()
        business_reviewers = dict(businesses_rdd.collect())
        raitings = dict(spark_context.textFile(review_file).map(lambda x: json.loads(x)).map(lambda x: ((x['business_id'], x['user_id']),x['stars']))\
                                                     .groupByKey()\
                                                     .map(lambda x: (x[0],statistics.mean(x[1])))
                                                     .collect())
        businesses_pairs = businesses_rdd.partitionBy(20)\
                                         .flatMap(lambda x:[(x[0],other) for other in distinct_businesses if\
                                                            x[0] != other and len(business_reviewers[x[0]].intersection(business_reviewers[other]))>=3])\
                                         .map(lambda x:(x[0],(x[1],pearson_correlation(x[0],x[1],business_reviewers,raitings))) if pearson_correlation(x[0],x[1],business_reviewers,raitings)>0 else None)\
                                         .filter(lambda x: x is not None)\
                                         .collect()

        with open(output_file,'w') as output:
            for pair in businesses_pairs:
                output.write('{'+'"b1":"'+pair[0]+'","b2":"'+pair[1][0]+'","sim":'+str(pair[1][1])+'}\n')
    else:
        uniq_pairs = {}
        users_rdd = spark_context.textFile(review_file).map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], x['business_id'])) \
                                                            .groupByKey() \
                                                            .map(lambda x: (x[0], set(x[1]))) \
                                                            .filter(lambda x: len(x[1]) >= 3)
        users_rdd.persist()
        distinct_users = users_rdd.map(lambda x: x[0]).distinct().collect()
        reviewed_businesses = dict(users_rdd.collect())
        raitings = dict(spark_context.textFile(review_file).map(lambda x: json.loads(x)).map(lambda x: ((x['user_id'], x['business_id']),x['stars']))\
                                                     .groupByKey()\
                                                     .map(lambda x: (x[0],statistics.mean(x[1])))
                                                     .collect())

        similar_users = users_rdd.partitionBy(20).flatMap(lambda x: [tuple(sorted((x[0],other))) for other in distinct_users if \
                                                                     x[0] != other \
                                                                     and len(reviewed_businesses[x[0]].intersection(reviewed_businesses[other])) >= 3 \
                                                                     and JaccardSim(x[0],other,reviewed_businesses)>=0.01])\
                                                 .distinct()\
                                                 .map(lambda x:(x[0],(x[1],pearson_correlation(x[0],x[1],reviewed_businesses,raitings))))\
                                                 .collect()

        with open(output_file,'w') as output:
            for pair in similar_users:
                output.write('{'+'"u1":"'+pair[0]+'","u2":"'+pair[1][0]+'","sim":'+str(pair[1][1])+'}\n')

    print("Duration: {:.2f}".format(time.time() - start_time))
    ddd=3