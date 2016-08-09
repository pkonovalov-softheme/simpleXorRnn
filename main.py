import tensorflow as tf
# from tensorflow.models.rnn import rnn

import bitarray
import os
import numpy as np

CONST_BYTES_TO_GENERATE = 5000
CONST_BATCH_SIZE = 2
assert(CONST_BYTES_TO_GENERATE % CONST_BATCH_SIZE == 0)
CONST_NO_OF_BATCHES = int(CONST_BYTES_TO_GENERATE / CONST_BATCH_SIZE);
CONST_INPUT_OTPUT_SIZE = 8  # 8 bits in byte
CONST_NUM_OF_HIDDEN_STATES = 24

CONST_EPOCH = 1


def byteToBitsArray(byte):
    for i in range(8):
        yield (byte >> i) & 1


def getDataAndResutls():
    byte_array = np.array(bytearray(os.urandom(CONST_BYTES_TO_GENERATE)))
    np.random.shuffle(byte_array)

    results = []
    results.append(byte_array[0])

    for i in range(1, len(byte_array)):
        curByte = byte_array[i]
        prevByte = byte_array[i - 1]
        result = curByte ^ prevByte
        results.append(result)

    return byte_array, results

def byteArrayToBitTensor(byteArray):
    resultList = []

    for curBatch in range(CONST_NO_OF_BATCHES):
        curTensor = np.empty([CONST_BATCH_SIZE, CONST_INPUT_OTPUT_SIZE])

        for curIdInsideBatch in range(CONST_BATCH_SIZE):
            curByteId = curBatch * CONST_BATCH_SIZE + curIdInsideBatch
            curByte = byteArray[curByteId]

            for curBitInByte in range(8):
                curBit = (curByte >> curBitInByte) & 1
                curTensor[curIdInsideBatch, curBitInByte] = bool(curBit)

        resultList.append(curTensor)

    return resultList

    # bits = np.unpackbits(byteArray)
    # #assert(len(bits) %  8 == 0)
    # assert(len(bits) == CONST_BYTES_TO_GENERATE * 8)
    # assert(len(bits) % CONST_BATCH_SIZE == 0)
    # assert(len(bits) % 8 == 0)
    #
    # print(len(bits))
    # #nputSize = len(bits) / 8
    # bitTensor = np.reshape(bits, (CONST_BATCH_SIZE, 8, -1))
    # print(bitTensor)
    # return bitTensor.astype(bool)

def main():
    byte_array, results = getDataAndResutls()

    byte_array = byteArrayToBitTensor(byte_array)
    results = byteArrayToBitTensor(results)

    training_idx = int(len(byte_array) * 0.8)

    test_input, train_input = byte_array[:training_idx], byte_array[training_idx:]
    test_output, train_output = results[:training_idx], results[training_idx:]


    #data = tf.placeholder(tf.bool, [None, CONST_INPUT_OTPUT_SIZE])  # Number of examples, number of input, dimension of each input
    data = []
    for _ in range(0, len(train_input)):
        data.append(tf.placeholder(tf.bool, [None, CONST_INPUT_OTPUT_SIZE]))

    target = tf.placeholder(tf.bool, [None, CONST_INPUT_OTPUT_SIZE])

    lstm = tf.nn.rnn_cell.BasicLSTMCell(CONST_NUM_OF_HIDDEN_STATES)

    val, state = tf.nn.rnn(lstm, data, dtype=tf.bool)
    #val, state = tf.nn.dynamic_rnn(lstm, data, dtype=tf.bool)
    #val = tf.transpose(val, [1, 0, 2])

    weight = tf.Variable(tf.truncated_normal([CONST_NUM_OF_HIDDEN_STATES, int(target.get_shape()[1])]))
    bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

    prediction = tf.nn.softmax(tf.matmul(val, weight) + bias)
    cross_entropy = -tf.reduce_sum(target * tf.log(prediction))

    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(cross_entropy)

    mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)

    print(train_input)
    for i in range(CONST_EPOCH):
        sess.run(minimize, {data: train_input, target: train_output})
        # ptr = 0
        # for j in range(CONST_NO_OF_BATCHES):
        #     inp, out = train_input[ptr:ptr + CONST_BATCH_SIZE], train_output[ptr:ptr + CONST_BATCH_SIZE]
        #     ptr += CONST_BATCH_SIZE
        #     sess.run(minimize, {data: inp, target: out})
        # print
        # "Epoch - ", str(i)
    incorrect = sess.run(error, {data: test_input, target: test_output})
    print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
    sess.close()

    print(len(results))


if __name__ == "__main__":
    main()
