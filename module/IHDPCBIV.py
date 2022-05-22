try:
    import tensorflow as tf
except:
    pass
import numpy as np
import sys, os
import getopt
import random
import datetime
import traceback

from utils.imbFun import *
from utils import log

def get_FLAGS():
    ''' Define parameter flags '''
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer('seed', 2021, """Seed. """)
    tf.app.flags.DEFINE_integer('debug', 0, """Debug mode. """)
    tf.app.flags.DEFINE_integer('save_rep', 0, """Save representations after training. """)
    tf.app.flags.DEFINE_integer('output_csv',0,"""Whether to save a CSV file with the results""")
    tf.app.flags.DEFINE_integer('output_delay', 100, """Number of iterations between log/loss outputs. """)
    tf.app.flags.DEFINE_string('loss', 'l2', """Which loss function to use (l1/l2/log)""")
    tf.app.flags.DEFINE_integer('n_in', 3, """Number of representation layers. """)
    tf.app.flags.DEFINE_integer('n_out', 3, """Number of regression layers. """)
    tf.app.flags.DEFINE_float('p_alpha', 1, """Imbalance regularization param. """)
    tf.app.flags.DEFINE_float('p_lambda', 1e-4, """Weight decay regularization parameter. """)
    tf.app.flags.DEFINE_integer('rep_weight_decay', 0, """Whether to penalize representation layers with weight decay""")
    tf.app.flags.DEFINE_float('dropout_in', 1.0, """Input layers dropout keep rate. """)
    tf.app.flags.DEFINE_float('dropout_out', 1.0, """Output layers dropout keep rate. """)
    tf.app.flags.DEFINE_string('nonlin', 'elu', """Kind of non-linearity. Default relu. """)
    tf.app.flags.DEFINE_float('lrate', 1e-3, """Learning rate. """)
    tf.app.flags.DEFINE_float('decay', 0.3, """RMSProp decay. """)
    tf.app.flags.DEFINE_integer('batch_size', 100, """Batch size. """)
    tf.app.flags.DEFINE_integer('dim_in', 200, """Pre-representation layer dimensions. """)
    tf.app.flags.DEFINE_integer('dim_out', 100, """Post-representation layer dimensions. """)
    tf.app.flags.DEFINE_integer('batch_norm', 0, """Whether to use batch normalization. """)
    tf.app.flags.DEFINE_string('normalization', 'divide', """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
    tf.app.flags.DEFINE_float('rbf_sigma', 0.1, """RBF MMD sigma """)
    tf.app.flags.DEFINE_integer('experiments', 2, """Number of experiments. """)
    tf.app.flags.DEFINE_integer('iterations', 3000, """Number of iterations. """)
    tf.app.flags.DEFINE_float('weight_init', 0.1, """Weight initialization scale. """)
    tf.app.flags.DEFINE_float('lrate_decay', 0.97, """Decay of learning rate every 100 iterations """)
    tf.app.flags.DEFINE_integer('wass_iterations', 10, """Number of iterations in Wasserstein computation. """)
    tf.app.flags.DEFINE_float('wass_lambda', 10.0, """Wasserstein lambda. """)
    tf.app.flags.DEFINE_integer('wass_bpt', 1, """Backprop through T matrix? """)
    tf.app.flags.DEFINE_integer('varsel', 0, """Whether the first layer performs variable selection. """)
    tf.app.flags.DEFINE_string('outdir', './Data/results/', """Output directory. """)
    tf.app.flags.DEFINE_string('datadir', '../Data/Causal/', """Data directory. """)
    tf.app.flags.DEFINE_string('dataform', 'ihdp_npci_1-100.train.npz', """Training data filename form. """)
    tf.app.flags.DEFINE_string('data_test', 'ihdp_npci_1-100.test.npz', """Test data filename form. """)
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
    tf.app.flags.DEFINE_string('f', '', 'kernel')

    if FLAGS.sparse:
        import scipy.sparse as sparse

    return FLAGS
    
NUM_ITERATIONS_PER_DECAY = 100

class CBIV(object):

    def __init__(self, x, s, t, y_ , p_t, FLAGS, r_alpha, r_lambda, do_in, do_out, dims):
        self.variables = {}
        self.wd_loss = 0

        if FLAGS.nonlin.lower() == 'elu':
            self.nonlin = tf.nn.elu
        else:
            self.nonlin = tf.nn.relu

        self._build_graph(x, s, t, y_ , p_t, FLAGS, r_alpha, r_lambda, do_in, do_out, dims)

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

    def _build_graph(self, x, s, t, y_ , p_t, FLAGS, r_alpha, r_lambda, do_in, do_out, dims):
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
        self.s = s
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
        y, y0, y1, weights_out, weights_pred, _, _, _, _ = self._build_output_graph(h_rep_norm, t, dim_in, dim_out, do_out, FLAGS)
        
        self.mu0 = y0
        self.mu1 = y1
        if FLAGS.twoStage:
            y = s * y1 + (1-s) * y0

        ''' Compute sample reweighting '''
        if FLAGS.reweight_sample:
            w_t = t/(2*p_t)
            w_c = (1-t)/(2*1-p_t)
            sample_weight = w_t + w_c
        else:
            sample_weight = 1.0

        self.sample_weight = sample_weight

        ''' Construct factual loss function '''
        if FLAGS.loss == 'l1':
            risk = tf.reduce_mean(sample_weight*tf.abs(y_-y))
            pred_error = -tf.reduce_mean(res)
        elif FLAGS.loss == 'log':
            y = 0.995/(1.0+tf.exp(-y)) + 0.0025
            res = y_*tf.log(y) + (1.0-y_)*tf.log(1.0-y)

            risk = -tf.reduce_mean(sample_weight*res)
            pred_error = -tf.reduce_mean(res)
        else:
            risk = tf.reduce_mean(sample_weight*tf.square(y_ - y))
            pred_error = tf.sqrt(tf.reduce_mean(tf.square(y_ - y)))

        ''' Regularization '''
        if FLAGS.p_lambda>0 and FLAGS.rep_weight_decay:
            for i in range(0, FLAGS.n_in):
                if not (FLAGS.varsel and i==0): # No penalty on W in variable selection
                    self.wd_loss += tf.nn.l2_loss(weights_in[i])

        ''' Imbalance error '''
        if FLAGS.use_p_correction:
            p_ipm = self.p_t
        else:
            p_ipm = 0.5

        if FLAGS.imb_fun == 'mmd2_rbf':
            imb_dist = mmd2_rbf(h_rep_norm,t,p_ipm,FLAGS.rbf_sigma)
            imb_error = r_alpha*imb_dist
        elif FLAGS.imb_fun == 'mmd2_lin':
            imb_dist = mmd2_lin(h_rep_norm,t,p_ipm)
            imb_error = r_alpha*mmd2_lin(h_rep_norm,t,p_ipm)
        elif FLAGS.imb_fun == 'mmd_rbf':
            imb_dist = tf.abs(mmd2_rbf(h_rep_norm,t,p_ipm,FLAGS.rbf_sigma))
            imb_error = safe_sqrt(tf.square(r_alpha)*imb_dist)
        elif FLAGS.imb_fun == 'mmd_lin':
            imb_dist = mmd2_lin(h_rep_norm,t,p_ipm)
            imb_error = safe_sqrt(tf.square(r_alpha)*imb_dist)
        elif FLAGS.imb_fun == 'wass':
            imb_dist, imb_mat = wasserstein(h_rep_norm,t,p_ipm,lam=FLAGS.wass_lambda,its=FLAGS.wass_iterations,sq=False,backpropT=FLAGS.wass_bpt)
            imb_error = r_alpha * imb_dist
            self.imb_mat = imb_mat # FOR DEBUG
        elif FLAGS.imb_fun == 'wass2':
            imb_dist, imb_mat = wasserstein(h_rep_norm,t,p_ipm,lam=FLAGS.wass_lambda,its=FLAGS.wass_iterations,sq=True,backpropT=FLAGS.wass_bpt)
            imb_error = r_alpha * imb_dist
            self.imb_mat = imb_mat # FOR DEBUG
        else:
            imb_dist = lindisc(h_rep_norm,p_ipm,t)
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

        if FLAGS.split_output:

            i0 = tf.to_int32(tf.where(t < 1)[:,0])
            i1 = tf.to_int32(tf.where(t > 0)[:,0])

            rep0 = tf.gather(rep, i0)
            rep1 = tf.gather(rep, i1)

            y0, weights_out0, weights_pred0 = self._build_output(rep, dim_in, dim_out, do_out, FLAGS)
            y1, weights_out1, weights_pred1 = self._build_output(rep, dim_in, dim_out, do_out, FLAGS)

            y = tf.dynamic_stitch([i0, i1], [tf.gather(y0, i0), tf.gather(y1, i1)])
            weights_out = weights_out0 + weights_out1
            weights_pred = weights_pred0 + weights_pred1
        else:
            h_input = tf.concat(1,[rep, t])
            y, weights_out, weights_pred = self._build_output(h_input, dim_in+1, dim_out, do_out, FLAGS)
            
            h0_input = tf.concat(1,[rep, t-t])
            y0, weights_out0, weights_pred0 = self._build_output(h0_input, dim_in+1, dim_out, do_out, FLAGS)
            
            h1_input = tf.concat(1,[rep, t-t+1])
            y1, weights_out1, weights_pred1 = self._build_output(h1_input, dim_in+1, dim_out, do_out, FLAGS)

        return y, y0, y1, weights_out, weights_pred, weights_out0, weights_pred0, weights_out1, weights_pred1

def trainNet(Net, sess, train_step, train_data, val_data, test_data, FLAGS, logfile, _logfile, exp):
    n_train = len(train_data['x'])
    p_treated = np.mean(train_data['t'])

    dict_factual = {Net.x: train_data['x'], Net.s: train_data['s'], Net.t: train_data['t'], Net.y_: train_data['yf'], \
            Net.do_in: 1.0, Net.do_out: 1.0, Net.r_alpha: FLAGS.p_alpha, \
            Net.r_lambda: FLAGS.p_lambda, Net.p_t: p_treated}

    dict_valid = {Net.x: val_data['x'], Net.s: val_data['t'], Net.t: val_data['t'], Net.y_: val_data['yf'], \
            Net.do_in: 1.0, Net.do_out: 1.0, Net.r_alpha: FLAGS.p_alpha, \
            Net.r_lambda: FLAGS.p_lambda, Net.p_t: p_treated}

    dict_cfactual = {Net.x: train_data['x'], Net.s: 1-train_data['s'], Net.t: 1-train_data['t'], Net.y_: train_data['ycf'], \
            Net.do_in: 1.0, Net.do_out: 1.0}

    ''' Initialize TensorFlow variables '''
    sess.run(tf.global_variables_initializer())
    objnan = False

    mse_val_best = 99999
    mse_val = {'best':99999, 'ate_train': None, 'ate_test': None, 'itr': 0,
            'hat_yf_train': None, 'hat_ycf_train': None, 'hat_mu0_train': None, 'hat_mu1_train': None , 
            'hat_yf_test': None, 'hat_ycf_test': None, 'hat_mu0_test': None, 'hat_mu1_test': None }

    obj_val_best = 99999
    obj_val = {'best':99999, 'ate_train': None, 'ate_test': None, 'itr': 0,
            'hat_yf_train': None, 'hat_ycf_train': None, 'hat_mu0_train': None, 'hat_mu1_train': None , 
            'hat_yf_test': None, 'hat_ycf_test': None, 'hat_mu0_test': None, 'hat_mu1_test': None }

    final   = {'best':99999, 'ate_train': None, 'ate_test': None, 'itr': 0,
            'hat_yf_train': None, 'hat_ycf_train': None, 'hat_mu0_train': None, 'hat_mu1_train': None , 
            'hat_yf_test': None, 'hat_ycf_test': None, 'hat_mu0_test': None, 'hat_mu1_test': None }

    ''' Train for multiple iterations '''
    for i in range(FLAGS.iterations):
        ''' Fetch sample '''
        I = random.sample(range(0, n_train), FLAGS.batch_size)
        x_batch = train_data['x'][I,:]
        s_batch = train_data['s'][I]
        t_batch = train_data['t'][I]
        y_batch = train_data['yf'][I]

        if not objnan:
            sess.run(train_step, feed_dict={Net.x: x_batch, Net.s: s_batch, Net.t: t_batch, \
                Net.y_: y_batch, Net.do_in: FLAGS.dropout_in, Net.do_out: FLAGS.dropout_out, \
                Net.r_alpha: FLAGS.p_alpha, Net.r_lambda: FLAGS.p_lambda, Net.p_t: p_treated})

        ''' Project variable selection weights '''
        if FLAGS.varsel:
            wip = simplex_project(sess.run(Net.weights_in[0]), 1)
            sess.run(Net.projection, feed_dict={Net.w_proj: wip})

        if i % FLAGS.output_delay == 0 or i==FLAGS.iterations-1:
            obj_loss,f_error,imb_err = sess.run([Net.tot_loss, Net.pred_loss, Net.imb_dist],
                            feed_dict=dict_factual)

            rep = sess.run(Net.h_rep_norm, feed_dict={Net.x: train_data['x'], Net.do_in: 1.0})
            rep_norm = np.mean(np.sqrt(np.sum(np.square(rep), 1)))

            cf_error = sess.run(Net.pred_loss, feed_dict=dict_cfactual)

            valid_obj, valid_f_error, valid_imb = sess.run([Net.tot_loss, Net.pred_loss, Net.imb_dist], feed_dict=dict_valid)

            if np.isnan(obj_loss):
                log(logfile,'Experiment %d: Objective is NaN. Skipping.' % exp)
                log(_logfile,'Experiment %d: Objective is NaN. Skipping.' % exp, False)
                objnan = True

            y_pred_f = sess.run(Net.output, feed_dict={Net.x: train_data['x'], Net.s: train_data['s'], Net.t: train_data['t'], Net.do_in: 1.0, Net.do_out: 1.0})
            y_pred_cf = sess.run(Net.output, feed_dict={Net.x: train_data['x'], Net.s: 1-train_data['s'], Net.t: 1-train_data['t'], Net.do_in: 1.0, Net.do_out: 1.0})
            y_pred_mu0 = sess.run(Net.output, feed_dict={Net.x: train_data['x'], Net.s: train_data['s']-train_data['s'], Net.t: train_data['t']-train_data['t'], Net.do_in: 1.0, Net.do_out: 1.0})
            y_pred_mu1 = sess.run(Net.output, feed_dict={Net.x: train_data['x'], Net.s: 1-train_data['s']+train_data['s'], Net.t: 1-train_data['t']+train_data['t'], Net.do_in: 1.0, Net.do_out: 1.0})

            y_pred_f_test = sess.run(Net.output, feed_dict={Net.x: test_data['x'], Net.s: test_data['s'], Net.t: test_data['t'], Net.do_in: 1.0, Net.do_out: 1.0})
            y_pred_cf_test = sess.run(Net.output, feed_dict={Net.x: test_data['x'], Net.s: 1-test_data['s'], Net.t: 1-test_data['t'], Net.do_in: 1.0, Net.do_out: 1.0})
            y_pred_mu0_test = sess.run(Net.output, feed_dict={Net.x: test_data['x'], Net.s: test_data['s']-test_data['s'], Net.t: test_data['t']-test_data['t'], Net.do_in: 1.0, Net.do_out: 1.0})
            y_pred_mu1_test = sess.run(Net.output, feed_dict={Net.x: test_data['x'], Net.s: 1-test_data['s']+test_data['s'], Net.t: 1-test_data['t']+test_data['t'], Net.do_in: 1.0, Net.do_out: 1.0})

            final = {'best':valid_f_error, 'ate_train': np.mean(y_pred_mu1) - np.mean(y_pred_mu0), 'ate_test': np.mean(y_pred_mu1_test) - np.mean(y_pred_mu0_test), 'itr': i,
                'hat_yf_train': y_pred_f, 'hat_ycf_train': y_pred_cf, 'hat_mu0_train': y_pred_mu0, 'hat_mu1_train': y_pred_mu1, 
                'hat_yf_test': y_pred_f_test, 'hat_ycf_test': y_pred_cf_test, 'hat_mu0_test': y_pred_mu0_test, 'hat_mu1_test': y_pred_mu1_test }

            if valid_f_error < mse_val_best:
                mse_val_best = valid_f_error
                mse_val = {'best':valid_f_error, 'ate_train': np.mean(y_pred_mu1) - np.mean(y_pred_mu0), 'ate_test': np.mean(y_pred_mu1_test) - np.mean(y_pred_mu0_test), 'itr': i,
                    'hat_yf_train': y_pred_f, 'hat_ycf_train': y_pred_cf, 'hat_mu0_train': y_pred_mu0, 'hat_mu1_train': y_pred_mu1, 
                    'hat_yf_test': y_pred_f_test, 'hat_ycf_test': y_pred_cf_test, 'hat_mu0_test': y_pred_mu0_test, 'hat_mu1_test': y_pred_mu1_test }

            if valid_obj < obj_val_best:
                obj_val_best = valid_obj
                obj_val = {'best':valid_obj, 'ate_train': np.mean(y_pred_mu1) - np.mean(y_pred_mu0), 'ate_test': np.mean(y_pred_mu1_test) - np.mean(y_pred_mu0_test), 'itr': i,
                    'hat_yf_train': y_pred_f, 'hat_ycf_train': y_pred_cf, 'hat_mu0_train': y_pred_mu0, 'hat_mu1_train': y_pred_mu1, 
                    'hat_yf_test': y_pred_f_test, 'hat_ycf_test': y_pred_cf_test, 'hat_mu0_test': y_pred_mu0_test, 'hat_mu1_test': y_pred_mu1_test }

            loss_str = str(i) + '\tObj: %.3f,\tF: %.3f,\tCf: %.3f,\tImb: %.2g,\tVal: %.3f,\tValImb: %.2g,\tValObj: %.2f,\tate_train: %.2g,\tate_test: %.2f' \
                    % (obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj, final['ate_train'], final['ate_test'])
            log(logfile, loss_str)
            log(_logfile, loss_str, False)

    return mse_val, obj_val, final

def run(exp, args, dataDir, resultDir, train, val, test, device, input_ihdp=False):
    tf.reset_default_graph()
    random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)
    np.random.seed(args.seed)

    logfile = f'{resultDir}/log.txt'
    _logfile = f'{resultDir}/CBIV.txt'
    alpha, lamda = args.ihdp_alpha, args.ihdp_lambda

    try:
        FLAGS = get_FLAGS()
    except:
        FLAGS = tf.app.flags.FLAGS

    FLAGS.reweight_sample = 0
    FLAGS.p_alpha = alpha
    FLAGS.p_lambda = lamda
    FLAGS.iterations = 200
    FLAGS.output_delay = 20
    FLAGS.lrate= 5e-4

    if args.ihdp_twoStage:
        FLAGS.twoStage = 1
        FLAGS.iterations = 100
    else:
        FLAGS.twoStage = 0

    try:
        train.to_numpy()
        val.to_numpy()
        test.to_numpy()
    except:
        pass

    if args.mode == 'xx':
        x_list = [np.concatenate((train.v, train.x), 1), 
                np.concatenate((val.v, val.x), 1), 
                np.concatenate((test.v, test.x), 1)]
    else:
        x_list = [train.x, val.x, test.x]

    train = {'x':x_list[0],
            't':train.t,    
            's':train.s,
            'g':train.g,
            'yf':train.y,
            'ycf':train.f}
    val = {'x':x_list[1],
            't':val.t,
            's':val.s,
            'g':val.g,
            'yf':val.y,
            'ycf':val.f}
    test = {'x':x_list[2],
            't':test.t,
            's':test.s,
            'g':test.g,
            'yf':test.y,
            'ycf':test.f}

    log(logfile, f'exp:{exp}; lrate:{FLAGS.lrate}; alpha: {FLAGS.p_alpha}; lambda: {FLAGS.p_lambda}; iterations: {FLAGS.iterations}; reweight: {FLAGS.reweight_sample}')
    log(_logfile, f'exp:{exp}; lrate:{FLAGS.lrate}; alpha: {FLAGS.p_alpha}; lambda: {FLAGS.p_lambda}; iterations: {FLAGS.iterations}; reweight: {FLAGS.reweight_sample}', False)

    ''' Start Session '''
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 

    ''' Initialize input placeholders '''
    x  = tf.placeholder("float", shape=[None,train['x'].shape[1]], name='x') # Features
    s  = tf.placeholder("float", shape=[None, 1], name='s')   # Saved Propensity Score
    t  = tf.placeholder("float", shape=[None, 1], name='t')   # Treatent
    y_ = tf.placeholder("float", shape=[None, 1], name='y_')  # Outcome

    ''' Parameter placeholders '''
    r_alpha = tf.placeholder("float", name='r_alpha')
    r_lambda = tf.placeholder("float", name='r_lambda')
    do_in = tf.placeholder("float", name='dropout_in')
    do_out = tf.placeholder("float", name='dropout_out')
    p = tf.placeholder("float", name='p_treated')

    ''' Define model graph '''
    log(logfile, 'Defining graph...\n')
    log(_logfile, 'Defining graph...\n', False)
    dims = [train['x'].shape[1], FLAGS.dim_in, FLAGS.dim_out]
    Net = CBIV(x, s, t, y_, p, FLAGS, r_alpha, r_lambda, do_in, do_out, dims)

    ''' Set up optimizer '''
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(FLAGS.lrate, global_step, \
        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)

    opt = None
    if FLAGS.optimizer == 'Adagrad':
        opt = tf.train.AdagradOptimizer(lr)
    elif FLAGS.optimizer == 'GradientDescent':
        opt = tf.train.GradientDescentOptimizer(lr)
    elif FLAGS.optimizer == 'Adam':
        opt = tf.train.AdamOptimizer(lr)
    else:
        opt = tf.train.RMSPropOptimizer(lr, FLAGS.decay)

    train_step = opt.minimize(Net.tot_loss,global_step=global_step)

    mse_val, obj_val, final = trainNet(Net, sess, train_step, train, val, test, FLAGS, logfile, _logfile, exp)

    return mse_val, obj_val, final


