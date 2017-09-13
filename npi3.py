# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import math

N = 5 # radix, originally 10.

class AdditionEnv:

    EYE_N = np.eye(N)

    LEFT, RIGHT, WRITE = range(3)

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
            n // (N ** 4) % N,
            n // (N ** 3) % N,
            n // (N ** 2) % N,
            n // (N ** 1) % N,
            n // (N ** 0) % N]

    @staticmethod
    def f_enc_input_size(npi):
        return N * 4 + npi.arg_size

    @staticmethod
    def build_f_enc(npi):
        with tf.name_scope('f_enc'):
            input_size = AdditionEnv.f_enc_input_size(npi)
            input = tf.placeholder(tf.float32, shape=(None, None, input_size))
            flat_input = tf.reshape(input, [-1, input_size])
            layer1 = tf.layers.dense(flat_input, 256, activation=tf.nn.elu, name='f_enc_dense_1',
                                     kernel_regularizer=tf.contrib.layers.l1_regularizer(0.01),
                                     bias_regularizer=tf.contrib.layers.l1_regularizer(0.01))
        return input, layer1

    def make_f_enc_input(self, arg):
        return np.concatenate([
            self.EYE_N[self.q[0][self.pointers[0]]],
            self.EYE_N[self.q[1][self.pointers[1]]],
            self.EYE_N[self.q[2][self.pointers[2]]],
            self.EYE_N[self.q[3][self.pointers[3]]],
            arg
        ])

    def act(self, prog_id, arg, show=False):
        assert prog_id == 0

        # ACT(LEFT, row, _), ACT(RIGHT, row, _), ACT(WRITE, row, value)
        #op, row, value = [int(round(x)) for x in arg]
        op, row, value = np.argmax(arg[0:N]), np.argmax(arg[N:2*N]), np.argmax(arg[2*N:3*N])

        if show:
            print('ACT({}, {}, {})'.format(op, row, value))

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
        return '<AdditionEnv {}>'.format('|'.join(' '.join(str(y) for y in x) for x in self.q))

class AdditionPlayer:

    EYE_N = np.eye(N)

    def __init__(self, env):
        self.env = env
        self.data = []

    def make_data(self):
        self.call(self.add, (0, 0, 0))
        return self.data

    def call(self, f, arg):
        prog = f.__name__.upper()
        encoded_arg = self.arg_encode(*arg)
        seq = []
        self.data.append(seq)
        def act(x, y, z):
            encoded_sub_arg = self.arg_encode(x, y, z)
            seq.append((prog, self.env.make_f_enc_input(encoded_arg), 'ACT', encoded_sub_arg))
            self.env.act(0, encoded_sub_arg)
        def call(f, arg):
            seq.append((prog, self.env.make_f_enc_input(encoded_arg), f.__name__.upper(), self.arg_encode(*arg)))
            self.call(f, arg)
        f(act, call)
        seq.append((prog, self.env.make_f_enc_input(encoded_arg), 'RETURN', self.arg_encode(0, 0, 0)))

    def add(self, act, call):
        s = self.env.a + self.env.b
        if s > 0:
            for i in range(math.floor(math.log(s, 10)) + 1):
                call(self.add1, (0, 0, 0))
                call(self.lshift, (0, 0, 0))

    def add1(self, act, call):
        v = self.env.q[0][self.env.pointers[0]] + self.env.q[1][self.env.pointers[0]] + self.env.q[2][self.env.pointers[0]]
        s, c = v % N, v // N
        act(AdditionEnv.WRITE, 3, s)
        if c > 0:
            call(self.carry, (0, 0, 0))

    def carry(self, act, call):
        act(AdditionEnv.LEFT, 2, 0)
        act(AdditionEnv.WRITE, 2, 1)
        act(AdditionEnv.RIGHT, 2, 0)

    def lshift(self, act, call):
        act(AdditionEnv.LEFT, 0, 0)
        act(AdditionEnv.LEFT, 1, 0)
        act(AdditionEnv.LEFT, 2, 0)
        act(AdditionEnv.LEFT, 3, 0)

    @staticmethod
    def arg_encode(x, y, z):
        return np.concatenate([AdditionPlayer.EYE_N[x], AdditionPlayer.EYE_N[y], AdditionPlayer.EYE_N[z], [0.] * (32 - N*3)]) # hard-coded

    def arg_decode(self, x):
        return (np.argmax(x[0:N]), np.argmax(x[N:2*N]), np.argmax(x[2*N:3*N]))

class NPI:

    num_progs_max = 32
    prog_key_size = 5
    prog_emb_size = 10
    arg_size = 32
    d = 32 # XXX
    m = 256

    def __init__(self, Env):
        self.Env = Env
        self.prog_names = ['ACT', 'RETURN'] + Env.PROGRAM_NAMES

        self.build_core()

    def build_core(self):
        self.prog_keys = tf.get_variable('prog_keys', [self.prog_key_size, self.num_progs_max],
                                         initializer=tf.random_normal_initializer(stddev=0.35),
                                         regularizer=tf.contrib.layers.l2_regularizer(0.05))
        self.prog_embs = tf.get_variable('prog_embs', [self.num_progs_max, self.prog_emb_size],
                                         initializer=tf.random_normal_initializer(stddev=0.35))

        tf.summary.histogram('prog_keys', self.prog_keys)
        tf.summary.histogram('prog_embs', self.prog_embs)

        self.prog_id = tf.placeholder(tf.int32, shape=(None, None))
        flat_prog_id = tf.reshape(self.prog_id, [-1])
        self.f_enc_input, self.s = self.Env.build_f_enc(self)
        tf.summary.histogram('s', self.s)
        self.times = tf.placeholder(tf.int32, shape=(None,))
        self.keep_prob = tf.placeholder(tf.float32, shape=None)

        self.batch_size = tf.shape(self.prog_id)[0]
        self.time_max = tf.shape(self.prog_id)[1]

        prog_emb = tf.nn.embedding_lookup(self.prog_embs, flat_prog_id)

        inputs = tf.layers.dense(tf.concat([prog_emb, self.s], axis=1), self.m, name='hoge')
        inputs = tf.reshape(inputs, [-1, self.time_max, self.m])

        cells = [tf.contrib.rnn.GRUCell(self.m) for _ in range(2)]
        cells = [tf.contrib.rnn.DropoutWrapper(cell, self.keep_prob, state_keep_prob=self.keep_prob, variational_recurrent=True, dtype=tf.float32, input_size=self.m) for cell in cells]
        self.cell = tf.contrib.rnn.MultiRNNCell(cells)
        self.hidden_state1 = tf.placeholder(tf.float32, shape=(None, self.m))
        self.hidden_state2 = tf.placeholder(tf.float32, shape=(None, self.m))
        hidden_state = (self.hidden_state1, self.hidden_state2)

        outputs, (self.next_hidden_state1, self.next_hidden_state2) = tf.nn.dynamic_rnn(self.cell, inputs, sequence_length=self.times, initial_state=hidden_state)

        flat_outputs = tf.reshape(outputs, [-1, self.m])

        flat_outputs = tf.layers.dense(flat_outputs, 64,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                                       bias_regularizer=tf.contrib.layers.l2_regularizer(0.1))

        with tf.name_scope('prog_id'):
            flat_next_prog_key = tf.layers.dense(flat_outputs, self.prog_key_size,
                                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                                                 bias_regularizer=tf.contrib.layers.l2_regularizer(0.1))
            flat_next_prog_prods = tf.matmul(flat_next_prog_key, self.prog_keys)
            flat_next_prog_id = tf.argmax(flat_next_prog_prods, axis=1)
            self.next_prog_id = tf.reshape(flat_next_prog_id, [-1, self.time_max])
        with tf.name_scope('prog_arg'):
            flat_next_prog_arg = tf.layers.dense(flat_outputs, self.arg_size,
                                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                                                 bias_regularizer=tf.contrib.layers.l2_regularizer(0.1)) # env-specific
            #flat_next_prog_arg = tf.layers.dense(flat_outputs, 64, activation=tf.nn.elu) # env-specific
            #flat_next_prog_arg = tf.layers.dense(flat_next_prog_arg, self.arg_size) # env-specific
            self.next_prog_arg = tf.reshape(flat_next_prog_arg, [-1, self.time_max, self.arg_size])

        tf.summary.histogram('next_prog_id', self.next_prog_id)

        self.true_next_prog_id = tf.placeholder(tf.int32, shape=(None, None))
        self.true_next_prog_arg = tf.placeholder(tf.float32, shape=(None, None, self.arg_size))

        with tf.name_scope('loss'):
            next_prog_prods = tf.reshape(flat_next_prog_prods, [-1, self.time_max, self.num_progs_max])
            mask = tf.to_float(tf.sequence_mask(self.times, self.time_max))
            self.loss1 = tf.contrib.seq2seq.sequence_loss(
               next_prog_prods,
               self.true_next_prog_id,
               mask)
            # self.loss1 = tf.reduce_mean(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #     labels=tf.one_hot(self.true_next_prog_id, self.num_progs_max),
            #     logits=next_prog_prods), 2) * mask)
            self.loss2 = tf.reduce_mean(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.true_next_prog_arg,
                logits=self.next_prog_arg), 2) * mask)
            self.regularization_loss = 0.0001 * tf.losses.get_regularization_loss()
            self.loss = self.loss1 + self.loss2 + self.regularization_loss

            tf.summary.scalar('loss1', self.loss1)
            tf.summary.scalar('loss2', self.loss2)
            tf.summary.scalar('regularization_loss', self.regularization_loss)
            tf.summary.scalar('loss', self.loss)

        self.lr = tf.placeholder(tf.float32)
        tf.summary.scalar('learning_rate', self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)

        gvs = optimizer.compute_gradients(self.loss)
        self.optimize = optimizer.apply_gradients(gvs, tf.train.get_or_create_global_step())

        self.summary_op = tf.summary.merge_all()

    def reset(self, sess):
        sess.run(tf.global_variables_initializer())

    def interpret(self, sess, env, prog_id, arg=None, depth=0, step=0):
        if arg is None:
            arg = env.initial_arg(self.arg_size)

        if prog_id >= len(self.prog_names):
            raise Exception('call unknown program {}'.format(prog_id))
        prog_name = self.prog_names[prog_id]

        print('{}{}'.format('  ' * depth, prog_name))
        h1 = np.zeros([1, self.cell.state_size[0]])
        h2 = np.zeros([1, self.cell.state_size[1]])

        while True:
            step += 1
            if step > 100:
                raise Exception('step too long')

            [[sub_prog_id]], [[sub_prog_arg]], h1, h2 = sess.run(
                [self.next_prog_id, self.next_prog_arg, self.next_hidden_state1, self.next_hidden_state2], {
                    self.prog_id: np.array([[prog_id]], dtype=np.int32),
                    self.f_enc_input: [[env.make_f_enc_input(arg)]],
                    self.times: [1],
                    self.hidden_state1: h1,
                    self.hidden_state2: h2,
                    self.keep_prob: 1.0
                })
            #print(sub_prog_id, sub_prog_arg)

            if sub_prog_id == 0: # is_act
                #print('{}act {}'.format('  ' * depth, sub_prog_arg))
                print('  ' * (depth + 1), end='')
                env.act(sub_prog_id, sub_prog_arg, show=True)
            elif sub_prog_id == 1: # tarminate
                print('{}{} end.'.format('  ' * depth, prog_name))
                break
            else:
                step = self.interpret(sess, env, sub_prog_id, sub_prog_arg, depth + 1, step)

        return step

    def learn(self, sess, data):
        global global_steps

        prog_id = []
        f_enc_input = []
        next_prog_id = []
        next_prog_arg = []
        times = []
        time_max = max(map(len, data))
        for seq in data:
            prog_id_, f_enc_input_, next_prog_id_, next_prog_arg_ = zip(*seq)
            prog_id_ = list(map(self.prog_names.index, prog_id_))
            next_prog_id_ = list(map(self.prog_names.index, next_prog_id_))
            time = len(prog_id_)
            times.append(time)
            prog_id.append(prog_id_ + [0] * (time_max - time))
            f_enc_input.append(list(f_enc_input_) + [[0.] * self.Env.f_enc_input_size(self)] * (time_max - time))
            next_prog_id.append(next_prog_id_ + [0] * (time_max - time))
            next_prog_arg.append(list(next_prog_arg_) + [[0.] * self.arg_size] * (time_max - time))

        '''
        print(data)
        print(prog_id)
        print(f_enc_input)
        print(next_prog_id)
        print(next_prog_arg)
        print(times)
        exit(0)
        #'''

        h1 = np.zeros([len(data), self.cell.state_size[0]])
        h2 = np.zeros([len(data), self.cell.state_size[1]])

        loss, _, w_summary = sess.run(
            [self.loss, self.optimize, self.summary_op], {
                self.prog_id: prog_id,
                self.f_enc_input: f_enc_input,
                self.true_next_prog_id: next_prog_id,
                self.true_next_prog_arg: next_prog_arg,
                self.times: times,
                self.hidden_state1: h1,
                self.hidden_state2: h2,
                #self.lr: 0.001 * 0.998 ** global_steps,
                self.lr: 0.0002 * 0.99995 ** global_steps,
                self.keep_prob: 0.5
            })

        global_steps += 1
        self.writer.add_summary(w_summary, global_steps)


global_steps = 0

def train():

    def test():
        env = AdditionEnv(1, 2)
        print(env)
        try:
            npi.interpret(sess, env, npi.prog_names.index('ADD'))
            print(env)
        except Exception as e:
            print(e)

        env = AdditionEnv(5, 6)
        print(env)
        try:
            npi.interpret(sess, env, npi.prog_names.index('ADD'))
            print(env)
        except Exception as e:
            print(e)

        env = AdditionEnv(123, 45)
        print(env)
        try:
            npi.interpret(sess, env, npi.prog_names.index('ADD'))
            print(env)
        except Exception as e:
            print(e)

    npi = NPI(AdditionEnv)
    sess = tf.Session()

    import glob
    npi.writer = tf.summary.FileWriter('tflog/{}'.format(len(glob.glob('tflog/*'))), sess.graph)
    npi.reset(sess)

    def make_batch(n, batch_size):
        data = []
        for _ in range(batch_size):
            a, b = np.random.randint(0, n, 2)
            env = AdditionEnv(a, b)
            player = AdditionPlayer(env)
            data.extend(player.make_data())
        return data

    for i in range(100000):
        if i % 25 == 0:
            print(i)

        n = 100

        npi.learn(sess, make_batch(n, 10))

        if i % 100 == 99:
            test()

if __name__ == '__main__':
    train()

    '''
    env = AdditionEnv(1, 2)
    print(env)
    player = AdditionPlayer(env)
    print('\n'.join(repr(l) for l in player.make_data()))

    print()

    env = AdditionEnv(123, 45)
    print(env)
    player = AdditionPlayer(env)
    print('\n'.join(repr(l) for l in player.make_data()))
    #'''
