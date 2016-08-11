import tensorflow as tf
import numpy as np
import os

CONST_BYTES_TO_GENERATE = 5000
CONST_BATCH_SIZE = 2
assert(CONST_BYTES_TO_GENERATE % CONST_BATCH_SIZE == 0)
CONST_NO_OF_BATCHES = int(CONST_BYTES_TO_GENERATE / CONST_BATCH_SIZE);
print("CONST_NO_OF_BATCHES")
print(CONST_NO_OF_BATCHES)

CONST_INPUT_OTPUT_SIZE = 8  # 8 bits in byte
CONST_NUM_OF_HIDDEN_STATES = 24

CONST_EPOCH = 1

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
                curTensor[curIdInsideBatch, curBitInByte] = curBit

        resultList.append(curTensor)

    return resultList


def main():
    byte_array, results = getDataAndResutls()

    byte_array = byteArrayToBitTensor(byte_array)
    results = byteArrayToBitTensor(results)

    training_idx = int(len(byte_array) * 0.8)

    test_input, train_input = byte_array[:training_idx], byte_array[training_idx:]
    test_output, train_output = results[:training_idx], results[training_idx:]


    # data = []
    # for _ in range(0, len(train_input)):
    #     data.append(tf.placeholder(tf.float32, [CONST_BATCH_SIZE, CONST_INPUT_OTPUT_SIZE]))

    data = [tf.placeholder(tf.float32, [CONST_BATCH_SIZE, CONST_INPUT_OTPUT_SIZE]) for _ in range(len(train_input))]
    lstm = tf.nn.rnn_cell.BasicLSTMCell(CONST_NUM_OF_HIDDEN_STATES)

    val, state = tf.nn.rnn(lstm, data, dtype=tf.float32)
    print("Val")
    print(val[0].get_shape())
    print(len(val))

    weight = tf.Variable(tf.zeros([CONST_NUM_OF_HIDDEN_STATES, CONST_INPUT_OTPUT_SIZE]))

    for i in val:
        mult = tf.matmul(i, weight)

    bias = tf.Variable(tf.zeros([CONST_INPUT_OTPUT_SIZE]))

    prediction = tf.nn.softmax(mult + bias)

    #target = tf.placeholder(tf.float32, [None, CONST_INPUT_OTPUT_SIZE])
    target = [tf.placeholder(tf.float32, [None, CONST_INPUT_OTPUT_SIZE]) for _ in range(len(train_output))]

    cross_entropy = -tf.reduce_sum(target * tf.log(prediction))

    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(cross_entropy)

    mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)

    for i in range(CONST_EPOCH):
        #sess.run(minimize, {data: train_input, target: train_output})
        dict = {i: d for i, d in zip(data, train_input)}
        dict2 = {i: d for i, d in zip(target, train_output)}
        feed = {**dict, **dict2}
        sess.run(minimize, feed_dict= feed)
    incorrect = sess.run(error, {data: test_input, target: test_output})
    print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
    sess.close()


if __name__ == "__main__":
    main()
