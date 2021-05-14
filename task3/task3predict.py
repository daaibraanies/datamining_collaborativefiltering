import pyspark
import findspark
import json
import os.path
import time
import statistics
import math
import heapq


findspark.find()
spark_context = pyspark.SparkContext()
"""spark_context =  SparkSession \
    .builder \
    .master('local')\
    .appName("Task 2 HW3") \
    .getOrCreate()"""


def dump_file(filename, data, mode='w'):
    with open(filename, mode=mode) as f:
        f.write("{}".format(data))

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

def get_business_similarity(b1,b2,similar_businesses):
    try:
        similarity = similar_businesses[(b1,b2)]
        return similarity
    except KeyError:
        try:
            similarity = similar_businesses[(b2,b1)]
            return similarity
        except KeyError:
            return 0


def case1_predict(uid,bid,user_revs,sim_bus,raitings,neighbors):
    all_user_reviews = user_revs[uid]
    kn = [(business,get_business_similarity(bid,business,sim_bus)) for business in all_user_reviews]
    top_n_similars = heapq.nlargest(neighbors,kn,key=lambda x:x[1])
    numerator,denominator = 0,sum([math.fabs(top_similar[1]) for top_similar in top_n_similars])
    if denominator == 0:
        return None
    else:
        for n in top_n_similars:
            n_id = n[0]
            w = n[1]
            numerator += raitings[(n_id,uid)]*w
        return numerator/denominator

def case2_predict(uid,bid,users_avg,bus_rev,weights,raitings,neighbors):
    try:
        users_reviewed_item = bus_rev[bid]
        uid_avg = users_avg[uid]
        kn = [(business, get_business_similarity(bid, business, weights)) for business in users_reviewed_item]
        top_n_similars = heapq.nlargest(neighbors, kn, key=lambda x: x[1])
        denominator = sum([math.fabs(w) for other_uid,w in top_n_similars])
        if denominator == 0:
            return users_avg[uid]
        numerator = sum([(raitings[(other_uid,bid)]-users_avg[other_uid])*w for other_uid,w in top_n_similars])
        return uid_avg + (numerator/(denominator))
    except KeyError:
        return None

def output_prediction(prediciton):
    with open(output_file, 'w') as output:
        for pair in prediciton:
            output.write('{' + '"user_id":"' + pair[0][0] + '","business_id":"' + pair[0][1] + '","stars":' + str(pair[1]) + '}\n')

if __name__ == '__main__':
    start_time = time.time()
    train_file = '../train_review.json'
    test_file = '../test_review.json'
    model_file = 'task3user.model'
    output_file = 'task3user.predict'
    avg_users_file = os.path.dirname(train_file)+'/user_avg.json'
    avg_business_file = os.path.dirname(train_file) + '/business_avg.json'
    candidate_pairs = {}
    case = 2
    if case == 1:
        N = 2
        #[(business,user)]:star
        raitings = dict(spark_context.textFile(train_file).map(lambda x: json.loads(x)).map(lambda x: ((x['business_id'], x['user_id']),x['stars']))\
                                                     .groupByKey()\
                                                     .map(lambda x: (x[0],statistics.mean(x[1])))
                                                     .collect())
        #[(business,business)]:similarity
        weights = dict(spark_context.textFile(model_file).map(lambda x: json.loads(x)).map(lambda x:((x['b1'],x['b2']),x['sim'])).collect())

        #[user]:[businesses rated by user]
        user_reviews = dict(spark_context.textFile(train_file).map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], x['business_id']))\
                                                              .groupByKey()\
                                                              .map(lambda x:(x[0],list(x[1])))\
                                                              .collect())
        prediciton = spark_context.textFile(test_file).map(lambda x: json.loads(x)).map(lambda x:(x['user_id'],x['business_id']))\
                                                    .map(lambda x: ((x[0],x[1]),case1_predict(x[0],x[1],user_reviews,weights,raitings,N)))\
                                                    .filter(lambda x:x[1] is not None)\
                                                    .collect()
        output_prediction(prediciton)

    else:
        N = 2
        with open(avg_users_file, 'r',encoding='utf-8') as uf:
            user_avg = dict(json.load(uf))

        rdd = spark_context.textFile(train_file).map(lambda x: json.loads(x)).map(lambda x: ((x['user_id'], x['business_id']), x['stars']))

        # [(user,user)]:similarity
        weights = dict(spark_context.textFile(model_file).map(lambda x: json.loads(x)).map(lambda x: ((x['u1'], x['u2']), x['sim'])).collect())

        # [(user,business)]:star
        raitings = dict(rdd \
                        .groupByKey() \
                        .map(lambda x: (x[0], statistics.mean(x[1])))
                        .collect())

        # [business]:[users rated the business]
        business_reviews = dict(rdd.map(lambda x: (x[0][1], x[0][0])) \
                                .groupByKey() \
                                .collect())

        prediction = spark_context.textFile(test_file).map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], x['business_id'])) \
            .map(lambda x: ((x[0], x[1]), case2_predict(x[0], x[1], user_avg, business_reviews, weights, raitings, N))) \
            .collect()

        with open(output_file,'w') as output:
            for pred in prediction:
                answer = {"user_id":pred[0][0],"business_id":pred[0][1],"stars":pred[1]}
                output.write(json.dumps(answer))
                output.write('\n')

    print("Duration: {:.2f}".format(time.time() - start_time))