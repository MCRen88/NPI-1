# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

class AdditionEnv:

    eye10 = np.eye(10)

    LEFT = 0
    RIGHT = 1
    WRITE = 2

    PROGRAM_NAMES = 'ADD ADD1 CARRY LSHIFT'.split()

    def __init__(self, a=1234, b=5678):
        self.a, self.b = a, b
        self.q = [
            self.encode(n)
            for n in [self.a, self.b, 0, 0]
        ]
        self.pointers = [4, 4, 4, 4]

    def encode(self, n):
        return [
            n // 10000 % 10,
            n // 1000 % 10,
            n // 100 % 10,
            n // 10 % 10,
            n // 1 % 10]

    @staticmethod
    def build_f_enc(npi):
        input = tf.placeholder(tf.float32, shape=(None, 10 * 4 + npi.arg_size))
        layer1 = tf.layers.dense(input, 64, activation=tf.nn.relu)
        layer2 = tf.layers.dense(layer1, npi.d)
        return input, layer2

    def make_f_enc_input(self, arg):
        return np.concatenate([
            self.eye10[self.q[0][self.pointers[0]]],
            self.eye10[self.q[1][self.pointers[1]]],
            self.eye10[self.q[2][self.pointers[2]]],
            self.eye10[self.q[3][self.pointers[3]]],
            arg
        ])

    def act(self, prog_id, arg):
        assert prog_id == 0

        # ACT(LEFT, row, _), ACT(RIGHT, row, _), ACT(WRITE, row, value)
        #op, row, value = [int(round(x)) for x in arg]
        op, row, value = np.argmax(arg[0:10]), np.argmax(arg[10:20]), np.argmax(arg[20:30])

        print(op, row, value)

        if row < 0 or 3 < row:
            return
        if op == self.LEFT:
            self.pointers[row] = max(0, self.pointers[row] - 1)
        elif op == self.RIGHT:
            self.pointers[row] = min(4, self.pointers[row] + 1)
        elif op == self.WRITE:
            self.q[row][self.pointers[row]] = value

    def initial_arg(self, arg_size):
        return [0.] * arg_size

    def __repr__(self):
        return '<AdditionEnv {}>'.format(self.q)

def operation(func):
    def wrapper(self, eop, *args, **kwargs):
        yield func.__name__.upper(), eop
        for x in func(self, *args, **kwargs): yield x
    return wrapper

class AdditionPlayer:

    eye10 = np.eye(10)

    def __init__(self, env):
        self.env = env

    def play(self):
        for op, eop in self.add(True):
            yield self.env, op, eop
            if not isinstance(op, str):
                self.env.act(0, op)

    @operation
    def add(self):
        l = len(str(self.env.a + self.env.b))
        for i in range(l):
            for x in self.add1(False): yield x
            for x in self.lshift(i == l-1): yield x

    @operation
    def add1(self):
        v = self.env.q[0][self.env.pointers[0]] + self.env.q[1][self.env.pointers[1]] + self.env.q[2][self.env.pointers[2]]
        s, c = v % 10, v // 10
        yield self.arg(AdditionEnv.WRITE, 3, s), c == 0
        if c > 0:
            for x in self.carry(True): yield x

    @operation
    def carry(self):
        yield self.arg(AdditionEnv.LEFT, 2, 0), False
        yield self.arg(AdditionEnv.WRITE, 2, 1), False
        yield self.arg(AdditionEnv.RIGHT, 2, 0), True

    @operation
    def lshift(self):
        yield self.arg(AdditionEnv.LEFT, 0, 0), False
        yield self.arg(AdditionEnv.LEFT, 1, 0), False
        yield self.arg(AdditionEnv.LEFT, 2, 0), False
        yield self.arg(AdditionEnv.LEFT, 3, 0), True

    def arg(self, x, y, z):
        return np.concatenate([self.eye10[x], self.eye10[y], self.eye10[z], [0., 0.]])

class NPI:

    MAX_NUM_PROGS = 32

    def __init__(self, Env):
        self.prog_key_size = 5
        self.prog_emb_size = 10
        self.arg_size = 32
        self.eop_threshold = 0.5
        self.d = 64 # XXX
        self.m = 256

        self.prog_names = ['ACT'] + Env.PROGRAM_NAMES
        self.prog_store = []
        self.build_core(Env)

    def build_core(self, Env):
        self.prog_keys = tf.Variable(tf.random_normal([self.MAX_NUM_PROGS, self.prog_key_size], stddev=0.35))
        self.prog_embs = tf.Variable(tf.random_normal([self.MAX_NUM_PROGS, self.prog_emb_size], stddev=0.35))

        self.prog_id = tf.placeholder(tf.int32, shape=(None,))
        self.f_enc_input, self.s = Env.build_f_enc(self)
        #self.prog_arg = tf.placeholder(tf.float32, shape=(None, self.prog_arg_size)) # TODO
        self.hidden_state = tf.placeholder(tf.float32, shape=(None, self.m))

        self.prog_emb = tf.nn.embedding_lookup(self.prog_embs, self.prog_id)
        self.prog_emb = tf.reshape(self.prog_emb, [-1, self.prog_emb_size])
        self.input = tf.layers.dense(tf.concat([self.prog_emb, self.s], axis=1), self.m)
        self.rnn_cell = tf.contrib.rnn.GRUCell(self.m)
        self.output_state, self.next_hidden_state = self.rnn_cell(self.input, self.hidden_state)

        self.next_prog_key = tf.layers.dense(self.output_state, 32, activation=tf.nn.relu)
        self.next_prog_key = tf.layers.dense(self.next_prog_key, self.prog_key_size)
        self.next_prog_prods = tf.matmul(tf.expand_dims(self.prog_keys, 0), tf.expand_dims(self.next_prog_key, -1))
        self.next_prog_prods = tf.squeeze(self.next_prog_prods, axis=2)
        self.next_prog_id = tf.argmax(self.next_prog_prods, axis=1)
        self.next_prog_arg = tf.layers.dense(self.output_state, 32, activation=tf.nn.relu) # env-specific
        self.next_prog_arg = tf.layers.dense(self.next_prog_arg, self.arg_size) # env-specific
        self.eop_prob = tf.layers.dense(self.output_state, 1)
        self.eop = tf.squeeze(self.eop_prob, axis=1)
        self.eop = tf.greater(self.eop, self.eop_threshold)


        self.y_next_prog_id = tf.placeholder(tf.int32, shape=(None,))
        self.y_next_prog_arg = tf.placeholder(tf.float32, shape=(None, self.arg_size))
        self.y_eop = tf.placeholder(tf.bool, shape=(None,))

        self.loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.y_next_prog_id, self.MAX_NUM_PROGS),
            logits=self.next_prog_prods) + \
            tf.nn.l2_loss(self.y_next_prog_arg - self.next_prog_arg) + \
            tf.nn.l2_loss(tf.to_float(self.y_eop) - self.eop_prob)

        self.loss_without_arg = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.y_next_prog_id, self.MAX_NUM_PROGS),
            logits=self.next_prog_prods) + \
            tf.nn.l2_loss(tf.to_float(self.y_eop) - self.eop_prob)

        self.lr = tf.placeholder(tf.float32)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.optimize = optimizer.minimize(self.loss)
        self.optimize_without_arg = optimizer.minimize(self.loss_without_arg)

    def initial_hidden_state(self, sess):
        return sess.run(self.rnn_cell.zero_state(1, tf.float32))

    def reset(self, sess):
        init = tf.global_variables_initializer()
        sess.run(init)

    def interpret(self, sess, env, prog_id, arg=None, depth=0):
        if arg is None:
            arg = env.initial_arg(self.arg_size)

        print('{}{}'.format('  ' * depth, self.prog_names[prog_id]))
        h = self.initial_hidden_state(sess)
        while True:
            feed_dict = {
                self.prog_id: np.array([prog_id], dtype=np.int32),
                self.f_enc_input: [env.make_f_enc_input(arg)],
                self.hidden_state: h
            }
            sub_prog_id, sub_prog_arg, eop, h = sess.run(
                [self.next_prog_id, self.next_prog_arg, self.eop, self.next_hidden_state],
                feed_dict=feed_dict)
            sub_prog_id, sub_prog_arg, eop = sub_prog_id[0], sub_prog_arg[0], eop[0]
            #print(sub_prog_id, sub_prog_arg, eop)

            if sub_prog_id == 0: # is_act
                #print('{}act {}'.format('  ' * depth, sub_prog_arg))
                env.act(sub_prog_id, sub_prog_arg)
            else:
                self.interpret(sess, env, sub_prog_id, sub_prog_arg, depth + 1)

            if eop:
                print('{}{} end.'.format('  ' * depth, self.prog_names[prog_id]))
                break

    def interpret_(self, sess, env, prog_id, arg=None, depth=0):
        if arg is None:
            arg = env.initial_arg(self.arg_size)

        print('{}{}'.format('  ' * depth, prog_id))
        h = self.initial_hidden_state(sess)
        for _ in range(50):
            feed_dict = {
                self.prog_id: np.array([prog_id], dtype=np.int32),
                self.f_enc_input: [env.make_f_enc_input(arg)],
                self.hidden_state: h,
                self.y_next_prog_id: [1],
                self.y_next_prog_arg: [(2., 3., 4.)],
                self.y_eop: [True],
                self.lr: 0.0001,
            }
            sub_prog_id, sub_prog_arg, eop, h = sess.run(
                [self.next_prog_id, self.next_prog_arg, self.eop, self.next_hidden_state],
                feed_dict=feed_dict)
            sub_prog_id, sub_prog_arg, eop = sub_prog_id[0], sub_prog_arg[0], eop[0]
            print(sub_prog_id, sub_prog_arg, eop)

            sess.run(self.optimize, feed_dict)

    def learn_(self, sess, env, play, prog_id, arg=None):
        if arg is None:
            arg = env.initial_arg(self.arg_size)

        h = self.initial_hidden_state(sess)
        while True:
            env, op, eop = next(play)
            if not isinstance(op, str):
                next_prog_arg = np.array(op, dtype=np.float32)
                feed_dict = {
                    self.prog_id: np.array([prog_id], dtype=np.int32),
                    self.f_enc_input: [env.make_f_enc_input(arg)],
                    self.hidden_state: h,
                    self.y_next_prog_id: [0.],
                    self.y_next_prog_arg: [next_prog_arg],
                    self.y_eop: [eop],
                    self.lr: 0.0001,
                }
                _, loss = sess.run([self.optimize, self.loss], feed_dict)
                print(loss)

                env.act(0, next_prog_arg)
            else:
                sub_prog_id = self.prog_names.index(op)
                feed_dict = {
                    self.prog_id: np.array([prog_id], dtype=np.int32),
                    self.f_enc_input: [env.make_f_enc_input(arg)],
                    self.hidden_state: h,
                    self.y_next_prog_id: [sub_prog_id],
                    self.y_eop: [eop],
                    self.lr: 0.0001,
                }
                _, sub_prog_arg, loss = sess.run([self.optimize_without_arg, self.next_prog_arg, self.loss_without_arg], feed_dict)
                sub_prog_arg = sub_prog_arg[0]
                print(loss)

                self.learn_(sess, env, play, sub_prog_id, sub_prog_arg)

            if eop:
                break

    def learn(self, sess, env, player):
        self.learn_(sess, env, player.play(), self.prog_names.index('ADD'))

def main():
    sess = tf.Session()
    npi = NPI(AdditionEnv)
    env = AdditionEnv()
    print(env)
    #npi.reset(sess)
    #npi.interpret(sess, env, 1)

def train():
    sess = tf.Session()
    npi = NPI(AdditionEnv)
    npi.reset(sess)
    for i in range(300):
        env = AdditionEnv(np.random.randint(1, 9), np.random.randint(1, 9))
        print((env.a, env.b))
        player = AdditionPlayer(env)
        npi.learn(sess, env, player)

        if i % 10 == 0:
            print(i)

    env = AdditionEnv(1, 2)
    print(env)
    npi.interpret(sess, env, 1)

    env = AdditionEnv(5, 6)
    print(env)
    npi.interpret(sess, env, 1)

    env = AdditionEnv(123, 45)
    print(env)
    npi.interpret(sess, env, 1)

if __name__ == '__main__':
    train()
    #main()
    '''
    env = AdditionEnv(123, 45)
    player = AdditionPlayer(env)
    for x in player.play():
        print(x)
    '''
