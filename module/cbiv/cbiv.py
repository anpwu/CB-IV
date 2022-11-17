import os
import random
import numpy as np
try:
    import tensorflow as tf
    import tensorflow.contrib.layers as layers
except:
    pass

from .imbFun import *

def get_FLAGS():
    ''' Define parameter flags '''
    FLAGS = tf.app.flags.FLAGS

    tf.app.flags.DEFINE_integer('lrate_decay_num', 100, """NUM_ITERATIONS_PER_DECAY. """)
    tf.app.flags.DEFINE_integer('seed', 2021, """Seed. """)
    tf.app.flags.DEFINE_integer('debug', 0, """Debug mode. """)
    tf.app.flags.DEFINE_integer('save_rep', 0, """Save representations after training. """)
    tf.app.flags.DEFINE_integer('output_csv',0,"""Whether to save a CSV file with the results""")
    tf.app.flags.DEFINE_integer('output_delay', 100, """Number of iterations between log/loss outputs. """)
    tf.app.flags.DEFINE_string('x_key', 'x', """Which key to use (x/xu/vxu)""")
    tf.app.flags.DEFINE_string('loss', 'l2', """Which loss function to use (l1/l2/log)""")
    tf.app.flags.DEFINE_integer('n_in', 3, """Number of representation layers. """)
    tf.app.flags.DEFINE_integer('n_out', 5, """Number of regression layers. """)
    tf.app.flags.DEFINE_float('p_alpha', 1, """Imbalance regularization param. """)
    tf.app.flags.DEFINE_float('p_lambda', 1e-4, """Weight decay regularization parameter. """)
    tf.app.flags.DEFINE_integer('rep_weight_decay', 0, """Whether to penalize representation layers with weight decay""")
    tf.app.flags.DEFINE_float('dropout_in', 1.0, """Input layers dropout keep rate. """)
    tf.app.flags.DEFINE_float('dropout_out', 1.0, """Output layers dropout keep rate. """)
    tf.app.flags.DEFINE_string('nonlin', 'elu', """Kind of non-linearity. Default relu. """)
    tf.app.flags.DEFINE_float('lrate', 5e-4, """Learning rate. """)
    tf.app.flags.DEFINE_float('decay', 0.3, """RMSProp decay. """)
    tf.app.flags.DEFINE_integer('batch_size', 256, """Batch size. """)
    tf.app.flags.DEFINE_integer('dim_in', 256, """Pre-representation layer dimensions. """)
    tf.app.flags.DEFINE_integer('dim_out', 256, """Post-representation layer dimensions. """)
    tf.app.flags.DEFINE_integer('batch_norm', 0, """Whether to use batch normalization. """)
    tf.app.flags.DEFINE_string('normalization', 'none', """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
    tf.app.flags.DEFINE_float('rbf_sigma', 0.1, """RBF MMD sigma """)
    tf.app.flags.DEFINE_integer('experiments', 2, """Number of experiments. """)
    tf.app.flags.DEFINE_integer('iterations', 3000, """Number of iterations. """)
    tf.app.flags.DEFINE_float('weight_init', 0.1, """Weight initialization scale. """)
    tf.app.flags.DEFINE_float('lrate_decay', 0.97, """Decay of learning rate every 100 iterations """)
    tf.app.flags.DEFINE_integer('wass_iterations', 10, """Number of iterations in Wasserstein computation. """)
    tf.app.flags.DEFINE_float('wass_lambda', 10.0, """Wasserstein lambda. """)
    tf.app.flags.DEFINE_integer('wass_bpt', 1, """Backprop through T matrix? """)
    tf.app.flags.DEFINE_integer('varsel', 0, """Whether the first layer performs variable selection. """)
    tf.app.flags.DEFINE_string('outdir', '/share/home/wuanpeng/Code/Jupyter/Data/DRCFR/results/', """Output directory. """)
    tf.app.flags.DEFINE_string('datadir', '/share/home/wuanpeng/Code/Jupyter/Data/DRCFR/data/Syn_1.0_1.0_0/2_4_4/', """Data directory. """)
    tf.app.flags.DEFINE_string('dataform', 'train_0.csv', """Training data filename form. """)
    tf.app.flags.DEFINE_string('data_val', 'val_0.csv', """Valid data filename form. """)
    tf.app.flags.DEFINE_string('data_test', 'test_0.csv', """Test data filename form. """)
    tf.app.flags.DEFINE_integer('sparse', 0, """Whether data is stored in sparse format (.x, .y). """)
    tf.app.flags.DEFINE_integer('repetitions', 1, """Repetitions with different seed.""")
    tf.app.flags.DEFINE_integer('use_p_correction', 0, """Whether to use population size p(t) in mmd/disc/wass.""")
    tf.app.flags.DEFINE_string('optimizer', 'Adam', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
    tf.app.flags.DEFINE_string('imb_fun', 'wass', """Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
    tf.app.flags.DEFINE_integer('pred_output_delay', 200, """Number of iterations between prediction outputs. (-1 gives no intermediate output). """)
    tf.app.flags.DEFINE_float('val_part', 0.3, """Validation part. """)
    tf.app.flags.DEFINE_boolean('split_output', 1, """Whether to split output layers between treated and control. """)
    tf.app.flags.DEFINE_boolean('reweight_sample', 1, """Whether to reweight sample for prediction loss with average treatment probability. """)
    tf.app.flags.DEFINE_boolean('twoStage', 1, """twoStage. """)
    tf.app.flags.DEFINE_integer('itr_balance', 2, """twoStage. """)
    tf.app.flags.DEFINE_string('f', '', 'kernel')

    if FLAGS.sparse:
        import scipy.sparse as sparse

    return FLAGS

class Net(object):

    def __init__(self, x, t, y_ , p_t, FLAGS, r_alpha, r_lambda, do_in, do_out, dims):
        self.variables = {}
        self.wd_loss = 0
        
        self.init = tf.contrib.layers.xavier_initializer()

        if FLAGS.nonlin.lower() == 'elu':
            self.nonlin = tf.nn.elu
        else:
            self.nonlin = tf.nn.relu

        self._build_graph(x, t, y_ , p_t, FLAGS, r_alpha, r_lambda, do_in, do_out, dims)

    def _add_variable(self, var, name):
        ''' Adds variables to the internal track-keeper '''
        basename = name
        i = 0
        while name in self.variables:
            name = '%s_%d' % (basename, i) #@TODO: not consistent with TF internally if changed
            i += 1

        self.variables[name] = var

    def _create_variable(self, var, name):
        ''' Create and adds variables to the internal track-keeper '''

        var = tf.Variable(var, name=name)
        self._add_variable(var, name)
        return var

    def _create_variable_with_weight_decay(self, initializer, name, wd):
        ''' Create and adds variables to the internal track-keeper
            and adds it to the list of weight decayed variables '''
        var = self._create_variable(initializer, name)
        self.wd_loss += wd*tf.nn.l2_loss(var)
        return var

    def _build_graph(self, x, t, y_ , p_t, FLAGS, r_alpha, r_lambda, do_in, do_out, dims):
        """
        Constructs a TensorFlow subgraph for counterfactual regression.
        Sets the following member variables (to TF nodes):

        self.output         The output prediction "y"
        self.tot_loss       The total objective to minimize
        self.imb_loss       The imbalance term of the objective
        self.pred_loss      The prediction term of the objective
        self.weights_in     The input/representation layer weights
        self.weights_out    The output/post-representation layer weights
        self.weights_pred   The (linear) prediction layer weights
        self.h_rep          The layer of the penalized representation
        """

        self.x = x
        self.t = t
        self.y_ = y_
        self.p_t = p_t
        self.r_alpha = r_alpha
        self.r_lambda = r_lambda
        self.do_in = do_in
        self.do_out = do_out

        dim_input = dims[0]
        dim_in = dims[1]
        dim_out = dims[2]

        weights_in = []; biases_in = []

        if FLAGS.n_in == 0 or (FLAGS.n_in == 1 and FLAGS.varsel):
            dim_in = dim_input
        if FLAGS.n_out == 0:
            if FLAGS.split_output == False:
                dim_out = dim_in+1
            else:
                dim_out = dim_in

        if FLAGS.batch_norm:
            bn_biases = []
            bn_scales = []

        ''' Construct input/representation layers '''
        with tf.compat.v1.variable_scope('representation'):
            h_in = [x]
            for i in range(0, FLAGS.n_in):
                if i==0:
                    ''' If using variable selection, first layer is just rescaling'''
                    if FLAGS.varsel:
                        weights_in.append(tf.Variable(1.0/dim_input*tf.ones([dim_input])))
                    else:
                        weights_in.append(tf.Variable(tf.random_normal([dim_input, dim_in], stddev=FLAGS.weight_init/np.sqrt(dim_input))))
                else:
                    weights_in.append(tf.Variable(tf.random_normal([dim_in,dim_in], stddev=FLAGS.weight_init/np.sqrt(dim_in))))

                ''' If using variable selection, first layer is just rescaling'''
                if FLAGS.varsel and i==0:
                    biases_in.append([])
                    h_in.append(tf.mul(h_in[i],weights_in[i]))
                else:
                    biases_in.append(tf.Variable(tf.zeros([1,dim_in])))
                    z = tf.matmul(h_in[i], weights_in[i]) + biases_in[i]

                    if FLAGS.batch_norm:
                        batch_mean, batch_var = tf.nn.moments(z, [0])

                        if FLAGS.normalization == 'bn_fixed':
                            z = tf.nn.batch_normalization(z, batch_mean, batch_var, 0, 1, 1e-3)
                        else:
                            bn_biases.append(tf.Variable(tf.zeros([dim_in])))
                            bn_scales.append(tf.Variable(tf.ones([dim_in])))
                            z = tf.nn.batch_normalization(z, batch_mean, batch_var, bn_biases[-1], bn_scales[-1], 1e-3)

                    h_in.append(self.nonlin(z))
                    h_in[i+1] = tf.nn.dropout(h_in[i+1], do_in)

            h_rep = h_in[len(h_in)-1]

            if FLAGS.normalization == 'divide':
                h_rep_norm = h_rep / safe_sqrt(tf.reduce_sum(tf.square(h_rep), axis=1, keep_dims=True))
            else:
                h_rep_norm = 1.0*h_rep

        ''' Construct ouput layers '''
        with tf.compat.v1.variable_scope('outcome'):
            y, y0, weights_out, weights_pred, weights_out0, weights_pred0 = self._build_output_graph(h_rep_norm, t, dim_in, dim_out, do_out, FLAGS)

        ''' Compute sample reweighting '''
        sample_weight = 1.0
        self.sample_weight = sample_weight

        ''' Construct factual loss function '''
        risk = tf.reduce_mean(tf.square(y_ - y))
        pred_error = tf.sqrt(tf.reduce_mean(tf.square(y_ - y)))

        ''' Regularization '''
        if FLAGS.p_lambda>0 and FLAGS.rep_weight_decay:
            for i in range(0, FLAGS.n_in):
                if not (FLAGS.varsel and i==0): # No penalty on W in variable selection
                    self.wd_loss += tf.nn.l2_loss(weights_in[i])

        ''' Imbalance error '''
        if True:
            """ Minimize MI between z and c. """
            with tf.compat.v1.variable_scope('balance'):
                self.lld, self.bound, self.mu, self.logvar = self.mi_net(
                    inp=h_rep_norm,
                    outp=t,
                    dim_in=dim_in,
                    dim_out=1,
                    mi_min_max='min')
            
            imb_dist = self.bound ** 2
        else:      
            imb_dist = tf.Variable(tf.zeros([1])) ** 2
        imb_error = r_alpha * imb_dist

        ''' Total error '''
        tot_error = risk

        if FLAGS.p_alpha>0:
            tot_error = tot_error + imb_error

        if FLAGS.p_lambda>0:
            tot_error = tot_error + r_lambda*self.wd_loss


        if FLAGS.varsel:
            self.w_proj = tf.placeholder("float", shape=[dim_input], name='w_proj')
            self.projection = weights_in[0].assign(self.w_proj)

        self.output = y
        self.tot_loss = tot_error
        self.imb_loss = imb_error
        self.imb_dist = imb_dist
        self.pred_loss = pred_error
        self.weights_in = weights_in
        self.weights_out = weights_out
        self.weights_pred = weights_pred
        self.h_rep = h_rep
        self.h_rep_norm = h_rep_norm
        
    def fc_net(self, inp, dim_out, act_fun, init):
        """ Fully-connected network. """
        return layers.fully_connected(inputs=inp,
                                      num_outputs=dim_out,
                                      activation_fn=act_fun,
                                      weights_initializer=init)

    def mi_net(self, inp, outp, dim_in, dim_out, mi_min_max, name=None):
        """ Mutual information network. """
        
        h_mu = self.fc_net(inp, dim_in // 2, tf.nn.elu, self.init)
        mu = self.fc_net(h_mu, dim_out, None, self.init)
        h_var = self.fc_net(inp, dim_in // 2, tf.nn.elu, self.init)
        logvar = self.fc_net(h_var, dim_out, tf.nn.tanh, self.init)

        # new_order = tf.random_shuffle(tf.range(self.num))
        # outp_rand = tf.gather(outp, new_order)
        outp_rand = tf.random_shuffle(outp)

        """ Get likelihood. """
        loglikeli = -tf.reduce_mean(tf.reduce_sum(-(outp - mu) ** 2 / tf.exp(logvar) - logvar, axis=-1))

        """ Get positive and negative U."""
        pos = - (mu - outp) ** 2 / tf.exp(logvar)
        neg = - (mu - outp_rand) ** 2 / tf.exp(logvar)

        """ Get estimation of mutual information. """
        if mi_min_max == 'min':
            pn = 1.
        elif mi_min_max == 'max':
            pn = -1.
        else:
            raise ValueError
        bound = pn * tf.reduce_mean( (pos - neg))

        return loglikeli, bound, mu, logvar

    def _build_output(self, h_input, dim_in, dim_out, do_out, FLAGS):
        h_out = [h_input]
        dims = [dim_in] + ([dim_out]*FLAGS.n_out)

        weights_out = []; biases_out = []

        for i in range(0, FLAGS.n_out):
            wo = self._create_variable_with_weight_decay(
                    tf.random_normal([dims[i], dims[i+1]],
                        stddev=FLAGS.weight_init/np.sqrt(dims[i])),
                    'w_out_%d' % i, 1.0)
            weights_out.append(wo)

            biases_out.append(tf.Variable(tf.zeros([1,dim_out])))
            z = tf.matmul(h_out[i], weights_out[i]) + biases_out[i]
            # No batch norm on output because p_cf != p_f

            h_out.append(self.nonlin(z))
            h_out[i+1] = tf.nn.dropout(h_out[i+1], do_out)

        weights_pred = self._create_variable(tf.random_normal([dim_out,1],
            stddev=FLAGS.weight_init/np.sqrt(dim_out)), 'w_pred')
        bias_pred = self._create_variable(tf.zeros([1]), 'b_pred')

        if FLAGS.varsel or FLAGS.n_out == 0:
            self.wd_loss += tf.nn.l2_loss(tf.slice(weights_pred,[0,0],[dim_out-1,1])) #don't penalize treatment coefficient
        else:
            self.wd_loss += tf.nn.l2_loss(weights_pred)

        ''' Construct linear classifier '''
        h_pred = h_out[-1]
        y = tf.matmul(h_pred, weights_pred)+bias_pred

        return y, weights_out, weights_pred

    def _build_output_graph(self, rep, t, dim_in, dim_out, do_out, FLAGS):
        ''' Construct output/regression layers '''
        
        ht_input = tf.concat([rep, t],1)
        yt, weights_outt, weights_predt = self._build_output(ht_input, dim_in+1, dim_out, do_out, FLAGS)

        h0_input = tf.concat([rep, t-t],1)
        y0, weights_out0, weights_pred0 = self._build_output(h0_input, dim_in+1, dim_out, do_out, FLAGS)

        return yt, y0, weights_outt, weights_predt, weights_out0, weights_pred0

def trainCBIV(CBIV, sess, train_step_RY, train_step_B, train_data, val_data, test_data, FLAGS, logfile, _logfile, exp):
    n_train = len(train_data['x'])
    p_treated = np.mean(train_data['t'])

    dict_factual = {CBIV.x: train_data['x'], CBIV.t: train_data['t'], CBIV.y_: train_data['yf'], \
            CBIV.do_in: 1.0, CBIV.do_out: 1.0, CBIV.r_alpha: FLAGS.p_alpha, \
            CBIV.r_lambda: FLAGS.p_lambda, CBIV.p_t: p_treated}

    dict_valid = {CBIV.x: val_data['x'], CBIV.t: val_data['t'], CBIV.y_: val_data['yf'], \
            CBIV.do_in: 1.0, CBIV.do_out: 1.0, CBIV.r_alpha: FLAGS.p_alpha, \
            CBIV.r_lambda: FLAGS.p_lambda, CBIV.p_t: p_treated}

    ''' Initialize TensorFlow variables '''
    sess.run(tf.global_variables_initializer())
    objnan = False

    mse_val_best = 99999
    obj_val_best = 99999
    mse_val = [[0,0,0],[0,0,0],[0,0,0]] # [itr, loss, error], [yt, t0, ate]_train, [yt, t0, ate]_test
    obj_val = [[0,0,0],[0,0,0],[0,0,0]]
    final   = [[0,0,0],[0,0,0],[0,0,0]]

    
    if FLAGS.batch_size > n_train:
        n_flag = n_train
    else:
        n_flag = FLAGS.batch_size
    ''' Train for multiple iterations '''
    for i in range(FLAGS.iterations):
        ''' Fetch sample '''
         
        I = random.sample(range(0, n_train), n_flag)
        x_batch = train_data['x'][I,:]
        t_batch = train_data['t'][I]
        y_batch = train_data['yf'][I]

        if not objnan:
            for _ in range(FLAGS.itr_balance):
                sess.run(train_step_B, feed_dict={CBIV.x: x_batch, CBIV.t: t_batch, \
                    CBIV.y_: y_batch, CBIV.do_in: FLAGS.dropout_in, CBIV.do_out: FLAGS.dropout_out, \
                    CBIV.r_alpha: FLAGS.p_alpha, CBIV.r_lambda: FLAGS.p_lambda, CBIV.p_t: p_treated})
            sess.run(train_step_RY, feed_dict={CBIV.x: x_batch, CBIV.t: t_batch, \
                CBIV.y_: y_batch, CBIV.do_in: FLAGS.dropout_in, CBIV.do_out: FLAGS.dropout_out, \
                CBIV.r_alpha: FLAGS.p_alpha, CBIV.r_lambda: FLAGS.p_lambda, CBIV.p_t: p_treated})
            

        ''' Project variable selection weights '''
        if FLAGS.varsel:
            wip = simplex_project(sess.run(CBIV.weights_in[0]), 1)
            sess.run(CBIV.projection, feed_dict={CBIV.w_proj: wip})

        if i % FLAGS.output_delay == 0 or i==FLAGS.iterations-1:
            obj_loss,f_error,imb_err = sess.run([CBIV.tot_loss, CBIV.pred_loss, CBIV.imb_dist],feed_dict=dict_factual)
            valid_obj, valid_f_error, valid_imb = sess.run([CBIV.tot_loss, CBIV.pred_loss, CBIV.imb_dist], feed_dict=dict_valid)

            if np.isnan(obj_loss):
                log(logfile,'Experiment %d: Objective is NaN. Skipping.' % exp)
                log(_logfile,'Experiment %d: Objective is NaN. Skipping.' % exp,False)
                objnan = True

            y_pred_t = sess.run(CBIV.output, feed_dict={CBIV.x: train_data['x'], CBIV.t: train_data['t'], CBIV.do_in: 1.0, CBIV.do_out: 1.0})
            y_pred_0 = sess.run(CBIV.output, feed_dict={CBIV.x: train_data['x'], CBIV.t: 1-train_data['t'], CBIV.do_in: 1.0, CBIV.do_out: 1.0})
            y_pred_t_test = sess.run(CBIV.output, feed_dict={CBIV.x: test_data['x'], CBIV.t: test_data['t'], CBIV.do_in: 1.0, CBIV.do_out: 1.0})
            y_pred_0_test = sess.run(CBIV.output, feed_dict={CBIV.x: test_data['x'], CBIV.t: 1-test_data['t'], CBIV.do_in: 1.0, CBIV.do_out: 1.0})

            final = [[i, valid_obj, valid_f_error], [y_pred_t, y_pred_0, y_pred_t-y_pred_0], [y_pred_t_test, y_pred_0_test, y_pred_t_test-y_pred_0_test]]
                
            if valid_f_error < mse_val_best:
                mse_val_best = valid_f_error
                mse_val = [[i, valid_obj, valid_f_error], [y_pred_t, y_pred_0, y_pred_t-y_pred_0], [y_pred_t_test, y_pred_0_test, y_pred_t_test-y_pred_0_test]]

            if valid_obj < obj_val_best:
                obj_val_best = valid_obj
                obj_val = [[i, valid_obj, valid_f_error], [y_pred_t, y_pred_0, y_pred_t-y_pred_0], [y_pred_t_test, y_pred_0_test, y_pred_t_test-y_pred_0_test]]

            train_mse =  ((y_pred_t - train_data['g']) ** 2).mean()
            test__mse =  ((y_pred_t_test - test_data['g']) ** 2).mean()
            loss_str = str(i) + '\tObj: %.3f,\tF: %.3f,\tImb: %.2g,\tValObj: %.2f,\tVal: %.3f,\tValImb: %.2g,\tTrainMSE: %.2g,\tTestMSE: %.2f' \
                    % (obj_loss, f_error, imb_err, valid_obj, valid_f_error, valid_imb, train_mse, test__mse)
            log(logfile, loss_str)
            log(_logfile, loss_str, False)

    return mse_val, obj_val, final

class CBIV(object):
    def __init__(self) -> None:
        self.config = {
                    'methodName': 'CBIV',
                    'epochs': 6000,
                    'batch_size':200,
                    'learning_rate': 5e-3,
                    'seed': 2022,   
                    'alpha': 1.0,
                    'lambda': 0.0001,
                    'loss': 'l2',
                    'save_path': './results/',
                    'reweight_sample': 0,
                    'output_delay': 20,
                    'itr_balance': 2,
                    'twoStage': True,
                    }

    def set_Configuration(self, config):
        self.config = config

    def fit(self, data, exp=-1, config=None):
        if config is None:
            config = self.config

        seed = config['seed']
        resultDir = config['save_path']
        alpha, lamda = config['alpha'], config['lambda']
        twoStage = config['twoStage']

        tf.reset_default_graph()
        random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)

        logfile = f'{resultDir}/log.txt'
        _logfile = f'{resultDir}/DRCFR.txt'

        try:
            FLAGS = get_FLAGS()
        except:
            FLAGS = tf.app.flags.FLAGS
        FLAGS.reweight_sample = config['reweight_sample']
        FLAGS.p_alpha = alpha
        FLAGS.p_lambda = lamda
        FLAGS.iterations = config['epochs']
        FLAGS.output_delay = config['output_delay']
        FLAGS.lrate= config['learning_rate']
        FLAGS.loss= config['loss']
        FLAGS.batch_size = config['batch_size']
        FLAGS.itr_balance = config['itr_balance']

        if twoStage:
            FLAGS.twoStage = 1
        else:
            FLAGS.twoStage = 0

        data.numpy()

        train = {'x':data.train.x,
                't':data.train.t,  
                'g':data.train.m[:,1:2],    
                'yf':data.train.y}
        val = {'x':data.valid.x,
                't':data.valid.t,
                'g':data.valid.m[:,1:2],    
                'yf':data.valid.y}
        test = {'x':data.test.x,
                't':data.test.t,
                'g':data.test.m[:,1:2],    
                'yf':data.test.y}

        log(logfile, f'exp:{exp}; lrate:{FLAGS.lrate}; alpha: {FLAGS.p_alpha}; lambda: {FLAGS.p_lambda}; iterations: {FLAGS.iterations}; reweight: {FLAGS.reweight_sample}')
        log(_logfile, f'exp:{exp}; lrate:{FLAGS.lrate}; alpha: {FLAGS.p_alpha}; lambda: {FLAGS.p_lambda}; iterations: {FLAGS.iterations}; reweight: {FLAGS.reweight_sample}', False)

        ''' Initialize input placeholders '''
        x  = tf.placeholder("float", shape=[None, train['x'].shape[1]], name='x') # Features
        t  = tf.placeholder("float", shape=[None, 1], name='t')   # Treatent
        y_ = tf.placeholder("float", shape=[None, 1], name='y_')  # Outcome
        ''' Parameter placeholders '''
        r_alpha = tf.placeholder("float", name='r_alpha')
        r_lambda = tf.placeholder("float", name='r_lambda')
        do_in = tf.placeholder("float", name='dropout_in')
        do_out = tf.placeholder("float", name='dropout_out')
        p = tf.placeholder("float", name='p_treated')
        dims = [train['x'].shape[1], FLAGS.dim_in, FLAGS.dim_out]
        CBIV = Net(x, t, y_, p, FLAGS, r_alpha, r_lambda, do_in, do_out, dims)

        ''' Start Session '''
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.compat.v1.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 
        
        ''' Set up optimizer '''
        R_vars = get_tf_var(['representation'])
        Y_vars = get_tf_var(['outcome'])
        B_vars = get_tf_var(['balance'])
        
        global_step_RY = tf.Variable(0, trainable=False)
        lr_RY = tf.train.exponential_decay(FLAGS.lrate, global_step_RY, FLAGS.lrate_decay_num, FLAGS.lrate_decay, staircase=True)
        opt_RY = tf.train.AdamOptimizer(lr_RY)
        train_step_RY = opt_RY.minimize(CBIV.tot_loss,global_step=global_step_RY,var_list=R_vars + Y_vars)
        
        global_step_B = tf.Variable(0, trainable=False)
        lr_B = tf.train.exponential_decay(FLAGS.lrate, global_step_B, FLAGS.lrate_decay_num, FLAGS.lrate_decay, staircase=True)
        opt_B = tf.train.AdamOptimizer(lr_B)
        train_step_B = opt_B.minimize(CBIV.lld,global_step=global_step_B,var_list=B_vars)

        mse_val, obj_val, final = trainCBIV(CBIV, sess, train_step_RY, train_step_B, train, val, test, FLAGS, logfile, _logfile, exp)
        
        self.mse_val = mse_val
        self.obj_val = obj_val
        self.final   = final
        self.CBIV   = CBIV
        self.sess  = sess

        return mse_val, obj_val, final

    def predict(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x

        if t is None:
            t = data.t

        y_pred_t = self.sess.run(self.CBIV.output, feed_dict={self.CBIV.x: x, 
        self.CBIV.t: t, self.CBIV.do_in: 1.0, self.CBIV.do_out: 1.0})
            
        return y_pred_t


    def ITE(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x

        if t is None:
            t = data.t

        y_pred_0 = self.sess.run(self.CBIV.output, feed_dict={self.CBIV.x: x, 
        self.CBIV.t: t-t, self.CBIV.do_in: 1.0, self.CBIV.do_out: 1.0})
        y_pred_1 = self.sess.run(self.CBIV.output, feed_dict={self.CBIV.x: x, 
        self.CBIV.t: t-t+1, self.CBIV.do_in: 1.0, self.CBIV.do_out: 1.0})
        y_pred_t = self.sess.run(self.CBIV.output, feed_dict={self.CBIV.x: x, 
        self.CBIV.t: t, self.CBIV.do_in: 1.0, self.CBIV.do_out: 1.0})

        
        return y_pred_0, y_pred_1, y_pred_t

    def ATE(self, data=None, t=None, x=None):
        ITE_0,ITE_1,ITE_t = self.ITE(data,t,x)

        return np.mean(ITE_1-ITE_0), np.mean(ITE_t-ITE_0)


def get_tf_var(names):
    _vars = []
    for na_i in range(len(names)):
        _vars = _vars + tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=names[na_i])
    return _vars

def get_opt(lrate, NUM_ITER_PER_DECAY, lrate_decay, loss, _vars):
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(lrate, global_step, NUM_ITER_PER_DECAY, lrate_decay, staircase=True)
    opt = tf.compat.v1.train.AdamOptimizer(lr)
    train_opt = opt.minimize(loss, global_step=global_step, var_list=_vars)
    return train_opt