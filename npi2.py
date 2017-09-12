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
        with tf.name_scope('f_enc'):
            input = tf.placeholder(tf.float32, shape=(None, 10 * 4 + npi.arg_size))
            layer1 = tf.layers.dense(input, 200, activation=tf.nn.elu)
            layer2 = tf.layers.dense(layer1, npi.d, activation=tf.nn.elu)
        return input, layer2

    def make_f_enc_input(self, arg):
        return np.concatenate([
            self.eye10[self.q[0][self.pointers[0]]],
            self.eye10[self.q[1][self.pointers[1]]],
            self.eye10[self.q[2][self.pointers[2]]],
            self.eye10[self.q[3][self.pointers[3]]],
            arg
        ])

    def act(self, prog_id, arg, show=False):
        assert prog_id == 0

        # ACT(LEFT, row, _), ACT(RIGHT, row, _), ACT(WRITE, row, value)
        #op, row, value = [int(round(x)) for x in arg]
        op, row, value = np.argmax(arg[0:10]), np.argmax(arg[10:20]), np.argmax(arg[20:30])

        if show:
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
        return '<AdditionEnv {}>'.format('|'.join(' '.join(str(x)) for x in self.q))

def operation(func):
    def wrapper(self, *args, **kwargs):
        yield func.__name__.upper()
        for x in func(self, *args, **kwargs): yield x
        yield 'RETURN'
    return wrapper

class AdditionPlayer:

    eye10 = np.eye(10)

    def __init__(self, env):
        self.env = env

    def play(self):
        for op in self.add():
            yield self.env, op
            if not isinstance(op, str):
                self.env.act(0, op)

    @operation
    def add(self):
        s = self.env.a + self.env.b
        if s > 0:
            for i in range(len(str(s))):
                for x in self.add1(): yield x
                for x in self.lshift(): yield x

    @operation
    def add1(self):
        v = self.env.q[0][self.env.pointers[0]] + self.env.q[1][self.env.pointers[1]] + self.env.q[2][self.env.pointers[2]]
        s, c = v % 10, v // 10
        yield self.arg(AdditionEnv.WRITE, 3, s)
        if c > 0:
            for x in self.carry(): yield x

    @operation
    def carry(self):
        yield self.arg(AdditionEnv.LEFT, 2, 0)
        yield self.arg(AdditionEnv.WRITE, 2, 1)
        yield self.arg(AdditionEnv.RIGHT, 2, 0)

    @operation
    def lshift(self):
        yield self.arg(AdditionEnv.LEFT, 0, 0)
        yield self.arg(AdditionEnv.LEFT, 1, 0)
        yield self.arg(AdditionEnv.LEFT, 2, 0)
        yield self.arg(AdditionEnv.LEFT, 3, 0)

    def arg(self, x, y, z):
        return np.concatenate([self.eye10[x], self.eye10[y], self.eye10[z], [0., 0.]]) # hard-coded

class NPI:

    MAX_NUM_PROGS = 32

    def __init__(self, Env):
        self.prog_key_size = 5
        self.prog_emb_size = 16
        self.arg_size = 32
        self.eop_threshold = 0.5
        self.d = 32 # XXX
        self.m = 256

        self.prog_names = ['ACT', 'RETURN'] + Env.PROGRAM_NAMES
        self.prog_store = []

        initializer = None #tf.random_uniform_initializer(-0.01, 0.01)
        with tf.variable_scope('model', initializer=initializer):
            self.build_core(Env)

    def build_core(self, Env):
        self.prog_keys = tf.Variable(tf.random_normal([self.MAX_NUM_PROGS, self.prog_key_size], stddev=0.35), name='prog_keys')
        self.prog_embs = tf.Variable(tf.random_normal([self.MAX_NUM_PROGS, self.prog_emb_size], stddev=0.35), name='prog_embs')

        tf.summary.histogram('prog_keys', self.prog_embs)

        self.prog_id = tf.placeholder(tf.int32, shape=(None,))
        self.f_enc_input, self.s = Env.build_f_enc(self)
        tf.summary.histogram('s', self.s)
        self.hidden_state = tf.placeholder(tf.float32, shape=(None, self.m))
        tf.summary.histogram('hidden_state', self.hidden_state)

        self.prog_emb = tf.nn.embedding_lookup(self.prog_embs, self.prog_id)
        self.prog_emb = tf.reshape(self.prog_emb, [-1, self.prog_emb_size])
        self.input = tf.layers.dense(tf.concat([self.prog_emb, self.s], axis=1), self.m, name='hoge')
        self.rnn_cell = tf.contrib.rnn.GRUCell(self.m)
        self.output_state, self.next_hidden_state = self.rnn_cell(self.input, self.hidden_state)
        self.output_state = tf.layers.dense(self.output_state, self.m)

        with tf.name_scope('prog_id'):
            self.next_prog_key = tf.layers.dense(self.output_state, 5, activation=tf.nn.elu)
            self.next_prog_prods = tf.matmul(tf.expand_dims(self.prog_keys, 0), tf.expand_dims(self.next_prog_key, -1))
            self.next_prog_prods = tf.squeeze(self.next_prog_prods, axis=2)
            self.next_prog_id = tf.argmax(self.next_prog_prods, axis=1)
        with tf.name_scope('prog_arg'):
            self.next_prog_arg = tf.layers.dense(self.output_state, 32, activation=tf.nn.elu) # env-specific
            self.next_prog_arg = tf.layers.dense(self.next_prog_arg, self.arg_size) # env-specific

        tf.summary.histogram('next_prog_id', self.next_prog_id)

        self.y_next_prog_id = tf.placeholder(tf.int32, shape=(None,))
        self.y_next_prog_arg = tf.placeholder(tf.float32, shape=(None, self.arg_size))

        with tf.name_scope('loss'):
            '''
            self.loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.y_next_prog_id, self.MAX_NUM_PROGS),
            logits=self.next_prog_prods) + \
            tf.nn.l2_loss(self.y_next_prog_arg - self.next_prog_arg) + \
            tf.nn.l2_loss(tf.to_float(self.y_eop) - self.eop_prob)

            self.loss_without_arg = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.y_next_prog_id, self.MAX_NUM_PROGS),
            logits=self.next_prog_prods) + \
            tf.nn.l2_loss(tf.to_float(self.y_eop) - self.eop_prob)
            '''
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.concat([
                    tf.one_hot(self.y_next_prog_id, self.MAX_NUM_PROGS),
                    self.y_next_prog_arg], axis=1),
                logits=tf.concat([
                    self.next_prog_prods,
                    self.next_prog_arg], axis=1)))

            self.loss_without_arg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.concat([
                    tf.one_hot(self.y_next_prog_id, self.MAX_NUM_PROGS)], axis=1),
                logits=tf.concat([
                    self.next_prog_prods], axis=1)))

            ##tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('loss_without_arg', self.loss_without_arg)
            #'''

        self.lr = tf.placeholder(tf.float32)
        optimizer = tf.train.AdamOptimizer(self.lr)
        #self.optimize = optimizer.minimize(self.loss)
        #self.optimize_without_arg = optimizer.minimize(self.loss_without_arg)

        gvs = optimizer.compute_gradients(self.loss)
        self.optimize = optimizer.apply_gradients(gvs, tf.train.get_or_create_global_step())
        gvs_without_arg = optimizer.compute_gradients(self.loss_without_arg)
        self.optimize_without_arg = optimizer.apply_gradients(gvs_without_arg, tf.train.get_or_create_global_step())

        self.summary_op = tf.summary.merge_all()

    def reset(self, sess):
        init = tf.global_variables_initializer()
        sess.run(init)

    def interpret(self, sess, env, prog_id, arg=None, depth=0, step=0):
        if arg is None:
            arg = env.initial_arg(self.arg_size)

        print('{}{}'.format('  ' * depth, self.prog_names[prog_id]))
        h = np.zeros([1, self.rnn_cell.state_size])
        while True:
            step += 1
            if step > 100:
                raise Exception('step too long')

            feed_dict = {
                self.prog_id: np.array([prog_id], dtype=np.int32),
                self.f_enc_input: [env.make_f_enc_input(arg)],
                self.hidden_state: h
            }
            sub_prog_id, sub_prog_arg, h = sess.run(
                [self.next_prog_id, self.next_prog_arg, self.next_hidden_state],
                feed_dict=feed_dict)
            sub_prog_id, sub_prog_arg = sub_prog_id[0], sub_prog_arg[0]
            #print(sub_prog_id, sub_prog_arg)

            if sub_prog_id == 0: # is_act
                #print('{}act {}'.format('  ' * depth, sub_prog_arg))
                env.act(sub_prog_id, sub_prog_arg, show=True)
            elif sub_prog_id == 1: # tarminate
                print('{}{} end.'.format('  ' * depth, self.prog_names[prog_id]))
                break
            else:
                step = self.interpret(sess, env, sub_prog_id, sub_prog_arg, depth + 1, step)

        return step

    def interpret_(self, sess, env, prog_id, arg=None, depth=0):
        if arg is None:
            arg = env.initial_arg(self.arg_size)

        print('{}{}'.format('  ' * depth, prog_id))
        h = np.zeros([1, self.rnn_cell.state_size])
        for _ in range(50):
            feed_dict = {
                self.prog_id: np.array([prog_id], dtype=np.int32),
                self.f_enc_input: [env.make_f_enc_input(arg)],
                self.hidden_state: h,
                self.y_next_prog_id: [15],
                self.y_next_prog_arg: [[0.]*9+[1.]+[0.]*9+[1.]+[0.]*9+[1.]+[0., 0.]],
                self.lr: 0.0001,
            }
            sub_prog_id, sub_prog_arg, h = sess.run(
                [self.next_prog_id, self.next_prog_arg, self.next_hidden_state],
                feed_dict=feed_dict)
            sub_prog_id, sub_prog_arg = sub_prog_id[0], sub_prog_arg[0]
            print(sub_prog_id, np.argmax(sub_prog_arg[0:10]), np.argmax(sub_prog_arg[10:20]), np.argmax(sub_prog_arg[20:30]))

            sess.run(self.optimize, feed_dict)

    def learn_(self, sess, env, play, prog_id, arg=None):
        global global_steps

        if arg is None:
            arg = env.initial_arg(self.arg_size)

        h = np.zeros([1, self.rnn_cell.state_size])
        while True:
            env, op = next(play)

            if not isinstance(op, str):
                next_prog_arg = np.array(op, dtype=np.float32)
                feed_dict = {
                    self.prog_id: np.array([prog_id], dtype=np.int32),
                    self.f_enc_input: [env.make_f_enc_input(arg)],
                    self.hidden_state: h,
                    self.y_next_prog_id: [0.],
                    self.y_next_prog_arg: [next_prog_arg],
                    self.lr: 0.0001,
                }
                #_, loss, h = sess.run([self.optimize, self.loss, self.next_hidden_state], feed_dict)
                _, loss, w_summary, h = sess.run([self.optimize, self.loss, self.summary_op, self.next_hidden_state], feed_dict)

                global_steps += 1
                self.writer.add_summary(w_summary, global_steps)
            else:
                sub_prog_id = self.prog_names.index(op)
                feed_dict = {
                    self.prog_id: np.array([prog_id], dtype=np.int32),
                    self.f_enc_input: [env.make_f_enc_input(arg)],
                    self.hidden_state: h,
                    self.y_next_prog_id: [sub_prog_id],
                    self.lr: 0.0001,
                }

                #_, sub_prog_arg, loss, h = sess.run([self.optimize_without_arg, self.next_prog_arg, self.loss_without_arg, self.next_hidden_state], feed_dict)
                _, sub_prog_arg, loss, w_summary, h = sess.run([self.optimize_without_arg, self.next_prog_arg, self.loss_without_arg, self.summary_op, self.next_hidden_state], feed_dict)
                sub_prog_arg = sub_prog_arg[0]

                global_steps += 1
                self.writer.add_summary(w_summary, global_steps)

                if op == 'RETURN':
                    break

                self.learn_(sess, env, play, sub_prog_id, sub_prog_arg)

    def learn(self, sess, env, player):
        play = player.play()
        env, op = next(play)
        assert isinstance(op, str)
        self.learn_(sess, env, play, self.prog_names.index(op))

global_steps = 0

def main():
    sess = tf.Session()
    npi = NPI(AdditionEnv)
    env = AdditionEnv()
    print(env)
    #npi.reset(sess)
    #npi.interpret(sess, env, npi.prog_names.index('ADD'))

def train_test():
    sess = tf.Session()
    npi = NPI(AdditionEnv)
    npi.reset(sess)
    env = AdditionEnv()
    try:
        npi.interpret_(sess, env, npi.prog_names.index('ADD'))
    except:
        pass

def train():

    def test():
        env = AdditionEnv(1, 2)
        print(env)
        try:
            npi.interpret(sess, env, npi.prog_names.index('ADD'))
            print(env)
        except: pass

        env = AdditionEnv(5, 6)
        print(env)
        try:
            npi.interpret(sess, env, npi.prog_names.index('ADD'))
            print(env)
        except: pass

        env = AdditionEnv(123, 45)
        print(env)
        try:
            npi.interpret(sess, env, npi.prog_names.index('ADD'))
            print(env)
        except: pass

    npi = NPI(AdditionEnv)
    sess = tf.Session()
    npi.writer = tf.summary.FileWriter('tflog/f', sess.graph)
    npi.reset(sess)
    for i in range(10000):
        if i < 200:
            n = 10000
        elif i < 300:
            n = 5
        elif i < 1000:
            n = 10
        elif i < 2000:
            n = 1000
        elif i < 4000:
            n = 10000
        env = AdditionEnv(np.random.randint(0, n), np.random.randint(0, n))

        if i % 25 == 0:
            print((env.a, env.b))
            print(i)

        player = AdditionPlayer(env)
        npi.learn(sess, env, player)

        if i % 100 == 99:
            test()

if __name__ == '__main__':
    train()
    #train_test()
    #main()
    '''
    env = AdditionEnv(123, 45)
    player = AdditionPlayer(env)
    for x in player.play():
        print(x)
    #'''
