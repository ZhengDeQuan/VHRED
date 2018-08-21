# -*- coding:utf-8 -*-


class DialogEncoderDecoder(Model):
    """
    Main model class, which links together all other sub-components
    and provides functions for training and sampling from the model.
    """

    def indices_to_words(self, seq, exclude_end_sym=True):
        '''
        :param seq:
        :param exclude_end_sym:
        :return:
        covert a list of word ids to a list of words
        if unk_sym then it will not be convert
        '''

        def convert():
            for word_index in seq:
                if word_index > len(self.idx_to_str):
                    raise ValueError('Word index is too large for the model vocabulary!')
                if not exclude_end_sym or (word_index != self.eos_sym):
                    yield self.idx_to_str[word_index]

        return list(convert())

    def words_to_indices(self, seq):
        """
        Converts a list of words to a list
        of word ids. Use unk_sym if a word is not
        known.
        """
        return [self.str_to_idx.get(word, self.unk_sym) for word in seq]

    def reverse_utterances(self, seq):
        """
        Reverses the words in each utterance inside a sequence of utterance (e.g. a dialogue)
        This is used for the bidirectional encoder RNN.
        """
        # 输入的seq是一个utterance的list，其中的每一个utterance🈶又需要被reverse
        reversed_seq = numpy.copy(seq)
        for idx in range(seq.shape[1]):
            #
            eos_indices = numpy.where(seq[:, idx] == self.eos_sym)[0]
            prev_eos_index = -1
            for eos_index in eos_indices:
                reversed_seq[(prev_eos_index + 1):eos_index, idx] = (reversed_seq[(prev_eos_index + 1):eos_index, idx])[
                                                                    ::-1]
                prev_eos_index = eos_index

        return reversed_seq

    def compute_updates(self, training_cost, params):
        updates = []

        grads = T.grad(training_cost, params)
        grads = OrderedDict(zip(params, grads))

        # Gradient clipping
        c = numpy.float32(self.cutoff)
        clip_grads = []

        norm_gs = T.sqrt(sum(T.sum(g ** 2) for p, g in grads.items()))
        normalization = T.switch(T.ge(norm_gs, c), c / norm_gs, np.float32(1.))
        notfinite = T.or_(T.isnan(norm_gs), T.isinf(norm_gs))

        for p, g in grads.items():
            clip_grads.append((p, T.switch(notfinite, numpy.float32(.1) * p, g * normalization)))

        grads = OrderedDict(clip_grads)

        if self.initialize_from_pretrained_word_embeddings and self.fix_pretrained_word_embeddings:
            assert not self.fix_encoder_parameters
            # Keep pretrained word embeddings fixed
            logger.debug("Will use mask to fix pretrained word embeddings")
            grads[self.W_emb] = grads[self.W_emb] * self.W_emb_pretrained_mask
        elif self.fix_encoder_parameters:
            # If 'fix_encoder_parameters' is on, the word embeddings will be excluded from parameter training set
            logger.debug("Will fix word embeddings to initial embeddings or embeddings from resumed model")
        else:
            logger.debug("Will train all word embeddings")

        if self.updater == 'adagrad':
            updates = Adagrad(grads, self.lr)
        elif self.updater == 'sgd':
            raise Exception("Sgd not implemented!")
        elif self.updater == 'adadelta':
            updates = Adadelta(grads)
        elif self.updater == 'rmsprop':
            updates = RMSProp(grads, self.lr)
        elif self.updater == 'adam':
            updates = Adam(grads, self.lr)
        else:
            raise Exception("Updater not understood!")

        return updates

    # Batch training function.
    def build_train_function(self):
        if not hasattr(self, 'train_fn'):
            # Compile functions
            logger.debug("Building train function")

            self.train_fn = theano.function(inputs=[self.x_data, self.x_data_reversed,
                                                    self.x_max_length, self.x_cost_mask,
                                                    self.x_reset_mask,
                                                    self.ran_cost_utterance, self.x_dropmask],
                                            outputs=[self.training_cost, self.kl_divergence_cost_acc,
                                                     self.latent_utterance_variable_approx_posterior_mean_var],
                                            updates=self.updates + self.state_updates,
                                            on_unused_input='warn',
                                            name="train_fn")

        return self.train_fn

    # Helper function used for computing the initial decoder hidden states before sampling starts.
    def build_decoder_encoding(self):
        if not hasattr(self, 'decoder_encoding_fn'):
            # Compile functions
            logger.debug("Building decoder encoding function")

            self.decoder_encoding_fn = theano.function(inputs=[self.x_data, self.x_data_reversed,
                                                               self.x_max_length, self.x_cost_mask,
                                                               self.x_reset_mask,
                                                               self.ran_cost_utterance, self.x_dropmask],
                                                       outputs=[self.hd],
                                                       on_unused_input='warn',
                                                       name="decoder_encoding_fn")

        return self.decoder_encoding_fn

    # Helper function used for the training with noise contrastive estimation (NCE).
    # This function is currently not supported.
    def build_nce_function(self):
        if not hasattr(self, 'train_fn'):
            # Compile functions
            logger.debug("Building NCE train function")

            self.nce_fn = theano.function(inputs=[self.x_data, self.x_data_reversed,
                                                  self.y_neg, self.x_max_length,
                                                  self.x_cost_mask,
                                                  self.x_reset_mask, self.ran_cost_utterance,
                                                  self.x_dropmask],
                                          outputs=[self.training_cost, self.kl_divergence_cost_acc,
                                                   self.latent_utterance_variable_approx_posterior_mean_var],
                                          updates=self.updates + self.state_updates,
                                          on_unused_input='warn',
                                          name="train_fn")

        return self.nce_fn

    # Batch evaluation function.
    def build_eval_function(self):
        if not hasattr(self, 'eval_fn'):
            # Compile functions
            logger.debug("Building evaluation function")
            self.eval_fn = theano.function(
                inputs=[self.x_data, self.x_data_reversed, self.x_max_length, self.x_cost_mask, self.x_reset_mask,
                        self.ran_cost_utterance, self.x_dropmask],
                outputs=[self.evaluation_cost, self.kl_divergence_cost_acc, self.softmax_cost, self.kl_divergence_cost,
                         self.latent_utterance_variable_approx_posterior_mean_var],
                updates=self.state_updates,
                on_unused_input='warn', name="eval_fn")
        return self.eval_fn

    # Helper function used to compare gradients given by reconstruction cost (softmax cost) and KL divergence between prior and approximate posterior for the (forward) utterance encoder.
    def build_eval_grads(self):
        if not hasattr(self, 'grads_eval_fn'):
            # Compile functions
            logger.debug("Building grad eval function")
            self.grads_eval_fn = theano.function(
                inputs=[self.x_data, self.x_data_reversed, self.x_max_length, self.x_cost_mask, self.x_reset_mask,
                        self.ran_cost_utterance, self.x_dropmask],
                outputs=[self.softmax_cost_acc, self.kl_divergence_cost_acc, self.grads_wrt_softmax_cost,
                         self.grads_wrt_kl_divergence_cost],
                on_unused_input='warn', name="eval_fn")
        return self.grads_eval_fn

    # Helper function used to compute encoder, context and decoder hidden states.
    def build_get_states_function(self):
        if not hasattr(self, 'get_states_fn'):
            # Compile functions
            logger.debug("Building selective function")

            outputs = [self.h, self.hs, self.hd] + [x for x in self.utterance_decoder_states]
            self.get_states_fn = theano.function(
                inputs=[self.x_data, self.x_data_reversed, self.x_max_length, self.x_reset_mask],
                outputs=outputs, updates=self.state_updates, on_unused_input='warn',
                name="get_states_fn")
        return self.get_states_fn

    # Helper function used to compute decoder hidden states and token probabilities.
    # Currently this function does not supported truncated computations.
    def build_next_probs_function(self):
        if not hasattr(self, 'next_probs_fn'):

            if self.add_latent_gaussian_per_utterance:

                if self.condition_latent_variable_on_dialogue_encoder:
                    self.hs_to_condition_latent_variable_on = self.beam_hs.dimshuffle((0, 'x', 1))[:, :, 0:self.sdim]
                else:
                    self.hs_to_condition_latent_variable_on = T.alloc(np.float32(0), self.beam_hs.shape[0], 1,
                                                                      self.beam_hs.shape[1])[:, :, 0:self.sdim]

                _prior_out = self.latent_utterance_variable_prior_encoder.build_encoder(
                    self.hs_to_condition_latent_variable_on, self.beam_x_data)
                latent_utterance_variable_prior_mean = _prior_out[1][-1]
                latent_utterance_variable_prior_var = _prior_out[2][-1]

                prior_sample = self.beam_ran_cost_utterance * T.sqrt(
                    latent_utterance_variable_prior_var) + latent_utterance_variable_prior_mean

                if self.condition_decoder_only_on_latent_variable:
                    decoder_inp = prior_sample
                else:
                    decoder_inp = T.concatenate([self.beam_hs, prior_sample], axis=1)
            else:
                decoder_inp = self.beam_hs

            outputs, hd = self.utterance_decoder.build_next_probs_predictor(decoder_inp, self.beam_source,
                                                                            prev_state=self.beam_hd)
            self.next_probs_fn = theano.function(
                inputs=[self.beam_hs, self.beam_hd, self.beam_source, self.beam_x_data, self.beam_ran_cost_utterance],
                outputs=[outputs, hd],
                on_unused_input='warn',
                name="next_probs_fn")
        return self.next_probs_fn

    # Currently this function does not supported truncated computations.
    # NOTE: If batch is given as input with padded endings,
    # e.g. last 'n' tokens are all zero and not part of the real sequence,
    # then the encoding must be extracted at index of the last non-padded (non-zero) token.
    def build_encoder_function(self):
        if not hasattr(self, 'encoder_fn'):

            if self.bidirectional_utterance_encoder:
                res_forward = self.utterance_encoder_forward.build_encoder(self.x_data)
                res_backward = self.utterance_encoder_backward.build_encoder(self.x_data_reversed)

                # Each encoder gives a single output vector
                h = T.concatenate([res_forward, res_backward], axis=2)

                hs = self.dialog_encoder.build_encoder(h, self.x_data)

                if self.direct_connection_between_encoders_and_decoder:
                    hs_dummy = self.dialog_dummy_encoder.build_encoder(h, self.x_data)
                    hs_complete = T.concatenate([hs, hs_dummy], axis=2)

                else:
                    hs_complete = hs
            else:
                h = self.utterance_encoder.build_encoder(self.x_data)

                hs = self.dialog_encoder.build_encoder(h, self.x_data)

                if self.direct_connection_between_encoders_and_decoder:
                    hs_dummy = self.dialog_dummy_encoder.build_encoder(h, self.x_data)
                    hs_complete = T.concatenate([hs, hs_dummy], axis=2)
                else:
                    hs_complete = hs

            if self.add_latent_gaussian_per_utterance:

                # Initialize hidden states to zero
                platent_utterance_variable_approx_posterior = theano.shared(
                    value=numpy.zeros((self.bs, self.latent_gaussian_per_utterance_dim), dtype='float32'),
                    name='encoder_fn_platent_utterance_variable_approx_posterior')

                if self.condition_latent_variable_on_dcgm_encoder:
                    platent_dcgm_avg = theano.shared(value=numpy.zeros((self.bs, self.rankdim), dtype='float32'),
                                                     name='encoder_fn_platent_dcgm_avg')
                    platent_dcgm_n = theano.shared(value=numpy.zeros((1, self.bs), dtype='float32'),
                                                   name='encoder_fn_platent_dcgm_n')

                # Create computational graph for latent variable
                latent_variable_mask = T.eq(self.x_data, self.eos_sym)

                if self.condition_latent_variable_on_dialogue_encoder:
                    hs_to_condition_latent_variable_on = hs
                else:
                    hs_to_condition_latent_variable_on = T.alloc(np.float32(0), hs.shape[0], hs.shape[1], hs.shape[2])

                logger.debug("Initializing approximate posterior encoder for utterance-level latent variable")
                if self.bidirectional_utterance_encoder and not self.condition_latent_variable_on_dcgm_encoder:
                    posterior_input_size = self.sdim + self.qdim_encoder * 2
                else:
                    posterior_input_size = self.sdim + self.qdim_encoder

                if self.condition_latent_variable_on_dcgm_encoder:
                    logger.debug("Build dcgm encoder")
                    latent_dcgm_res, latent_dcgm_avg, latent_dcgm_n = self.dcgm_encoder.build_encoder(self.x_data,
                                                                                                      prev_state=[
                                                                                                          platent_dcgm_avg,
                                                                                                          platent_dcgm_n])
                    h_future = self.utterance_encoder_rolledleft.build_encoder( \
                        latent_dcgm_res, \
                        self.x_data)

                else:
                    h_future = self.utterance_encoder_rolledleft.build_encoder( \
                        h, \
                        self.x_data)

                hs_and_h_future = T.concatenate([hs_to_condition_latent_variable_on, h_future], axis=2)

                logger.debug("Build approximate posterior encoder for utterance-level latent variable")
                _posterior_out = self.latent_utterance_variable_approx_posterior_encoder.build_encoder( \
                    hs_and_h_future, \
                    self.x_data, \
                    latent_variable_mask=latent_variable_mask)

                latent_utterance_variable_approx_posterior = _posterior_out[0]
                latent_utterance_variable_approx_posterior_mean = _posterior_out[1]
                latent_utterance_variable_approx_posterior_var = _posterior_out[2]

            training_y = self.x_data[1:self.x_max_length]
            if self.direct_connection_between_encoders_and_decoder:
                logger.debug("Build dialog dummy encoder")
                hs_dummy = self.dialog_dummy_encoder.build_encoder(h, self.x_data,
                                                                   xmask=T.neq(self.x_data, self.eos_sym))

                logger.debug("Build decoder (NCE) with direct connection from encoder(s)")
                if self.add_latent_gaussian_per_utterance:
                    if self.condition_decoder_only_on_latent_variable:
                        hd_input = latent_utterance_variable_approx_posterior_mean
                    else:
                        hd_input = T.concatenate([hs, hs_dummy, latent_utterance_variable_approx_posterior_mean],
                                                 axis=2)
                else:
                    hd_input = T.concatenate([hs, hs_dummy], axis=2)

                _, hd, _, _ = self.utterance_decoder.build_decoder(hd_input, self.x_data, y=training_y,
                                                                   mode=UtteranceDecoder.EVALUATION)

            else:
                if self.add_latent_gaussian_per_utterance:
                    if self.condition_decoder_only_on_latent_variable:
                        hd_input = latent_utterance_variable_approx_posterior_mean
                    else:
                        hd_input = T.concatenate([hs, latent_utterance_variable_approx_posterior_mean], axis=2)
                else:
                    hd_input = hs

                logger.debug("Build decoder (EVAL)")
                _, hd, _, _ = self.utterance_decoder.build_decoder(hd_input, self.x_data, y=training_y,
                                                                   mode=UtteranceDecoder.EVALUATION)

            if self.add_latent_gaussian_per_utterance:
                self.encoder_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, \
                                                          self.x_max_length], \
                                                  outputs=[h, hs_complete, hd], on_unused_input='warn',
                                                  name="encoder_fn")
                # self.encoder_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, \
                #             self.x_max_length], \
                #             outputs=[h, hs_complete, hs_and_h_future, latent_utterance_variable_approx_posterior_mean], on_unused_input='warn', name="encoder_fn")
            else:
                self.encoder_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, \
                                                          self.x_max_length], \
                                                  outputs=[h, hs_complete, hd], on_unused_input='warn',
                                                  name="encoder_fn")

        return self.encoder_fn

    def __init__(self, state):
        Model.__init__(self)

        # Make sure eos_sym is never zero, otherwise generate_encodings script would fail
        assert state['eos_sym'] > 0

        if not 'bidirectional_utterance_encoder' in state:
            state['bidirectional_utterance_encoder'] = False

        if 'encode_with_l2_pooling' in state:
            assert state['encode_with_l2_pooling'] == False  # We don't support L2 pooling right now...

        if not 'direct_connection_between_encoders_and_decoder' in state:
            state['direct_connection_between_encoders_and_decoder'] = False

        if not 'deep_direct_connection' in state:
            state['deep_direct_connection'] = False

        if not state['direct_connection_between_encoders_and_decoder']:
            assert (state['deep_direct_connection'] == False)

        if not 'collaps_to_standard_rnn' in state:
            state['collaps_to_standard_rnn'] = False

        if not 'reset_utterance_decoder_at_end_of_utterance' in state:
            state['reset_utterance_decoder_at_end_of_utterance'] = True

        if not 'reset_utterance_encoder_at_end_of_utterance' in state:
            state['reset_utterance_encoder_at_end_of_utterance'] = True

        if not 'deep_dialogue_input' in state:
            state['deep_dialogue_input'] = True

        if not 'reset_hidden_states_between_subsequences' in state:
            state['reset_hidden_states_between_subsequences'] = False

        if not 'fix_encoder_parameters' in state:
            state['fix_encoder_parameters'] = False

        if not 'decoder_drop_previous_input_tokens' in state:
            state['decoder_drop_previous_input_tokens'] = False
        else:
            if state['decoder_drop_previous_input_tokens']:
                assert state['decoder_drop_previous_input_tokens_rate']

        if not 'add_latent_gaussian_per_utterance' in state:
            state['add_latent_gaussian_per_utterance'] = False
        if not 'latent_gaussian_per_utterance_dim' in state:
            state['latent_gaussian_per_utterance_dim'] = 1
        if not 'condition_latent_variable_on_dialogue_encoder' in state:
            state['condition_latent_variable_on_dialogue_encoder'] = True
        if not 'condition_latent_variable_on_dcgm_encoder' in state:
            state['condition_latent_variable_on_dcgm_encoder'] = False
        if not 'scale_latent_variable_variances' in state:
            state['scale_latent_variable_variances'] = 0.01
        if not 'condition_decoder_only_on_latent_variable' in state:
            state['condition_decoder_only_on_latent_variable'] = False
        if not 'latent_gaussian_linear_dynamics' in state:
            state['latent_gaussian_linear_dynamics'] = False
        if not 'train_latent_gaussians_with_kl_divergence_annealing' in state:
            state['train_latent_gaussians_with_kl_divergence_annealing'] = False

        if state['train_latent_gaussians_with_kl_divergence_annealing']:
            assert state['kl_divergence_annealing_rate']

        if state['collaps_to_standard_rnn']:
            # If we collapse to standard RNN (e.g. LSTM language model) then we should not reset.
            # If we did reset, we'd have a language model over individual utterances, which is what we want!
            assert not state['reset_utterance_decoder_at_end_of_utterance']

        self.state = state
        self.global_params = []

        self.__dict__.update(state)
        self.rng = numpy.random.RandomState(state['seed'])

        # Load dictionary
        raw_dict = cPickle.load(open(self.dictionary, 'r'))

        # Probabilities for each term in the corpus used for noise contrastive estimation (NCE)
        self.noise_probs = [x[2] for x in sorted(raw_dict, key=operator.itemgetter(1))]
        self.noise_probs = numpy.array(self.noise_probs, dtype='float64')
        self.noise_probs /= numpy.sum(self.noise_probs)
        self.noise_probs = self.noise_probs ** 0.75
        self.noise_probs /= numpy.sum(self.noise_probs)

        self.t_noise_probs = theano.shared(self.noise_probs.astype('float32'), 't_noise_probs')

        # Dictionaries to convert str to idx and vice-versa
        self.str_to_idx = dict([(tok, tok_id) for tok, tok_id, _, _ in raw_dict])
        self.idx_to_str = dict([(tok_id, tok) for tok, tok_id, freq, _ in raw_dict])

        # Extract document (dialogue) frequency for each word
        self.word_freq = dict([(tok_id, freq) for _, tok_id, freq, _ in raw_dict])
        self.document_freq = dict([(tok_id, df) for _, tok_id, _, df in raw_dict])

        if self.end_sym_utterance not in self.str_to_idx:
            raise Exception("Error, malformed dictionary!")

        # Number of words in the dictionary
        self.idim = len(self.str_to_idx)
        self.state['idim'] = self.idim
        logger.debug("idim: " + str(self.idim))

        logger.debug("Initializing Theano variables")
        self.y_neg = T.itensor3('y_neg')
        self.x_data = T.imatrix('x_data')
        self.x_data_reversed = T.imatrix('x_data_reversed')
        self.x_cost_mask = T.matrix('cost_mask')
        self.x_reset_mask = T.vector('reset_mask')
        self.x_max_length = T.iscalar('x_max_length')
        self.ran_cost_utterance = T.tensor3('ran_cost_utterance')
        self.x_dropmask = T.matrix('x_dropmask')

        # The 'x' data (input) is defined as all symbols except the last, and
        # the 'y' data (output) is defined as all symbols except the first.
        training_x = self.x_data[:(self.x_max_length - 1)]
        training_x_reversed = self.x_data_reversed[:(self.x_max_length - 1)]
        training_y = self.x_data[1:self.x_max_length]
        training_x_dropmask = self.x_dropmask[:(self.x_max_length - 1)]

        # Here we find the end-of-utterance tokens in the minibatch.
        training_hs_mask = T.neq(training_x, self.eos_sym)
        training_x_cost_mask = self.x_cost_mask[1:self.x_max_length]
        training_x_cost_mask_flat = training_x_cost_mask.flatten()

        # Backward compatibility
        if 'decoder_bias_type' in self.state:
            logger.debug("Decoder bias type {}".format(self.decoder_bias_type))

        # Build word embeddings, which are shared throughout the model
        if self.initialize_from_pretrained_word_embeddings == True:
            # Load pretrained word embeddings from pickled file
            logger.debug("Loading pretrained word embeddings")
            pretrained_embeddings = cPickle.load(open(self.pretrained_word_embeddings_file, 'r'))

            # Check all dimensions match from the pretrained embeddings
            assert (self.idim == pretrained_embeddings[0].shape[0])
            assert (self.rankdim == pretrained_embeddings[0].shape[1])
            assert (self.idim == pretrained_embeddings[1].shape[0])
            assert (self.rankdim == pretrained_embeddings[1].shape[1])

            self.W_emb_pretrained_mask = theano.shared(pretrained_embeddings[1].astype(numpy.float32),
                                                       name='W_emb_mask')
            self.W_emb = add_to_params(self.global_params,
                                       theano.shared(value=pretrained_embeddings[0].astype(numpy.float32),
                                                     name='W_emb'))
        else:
            # Initialize word embeddings randomly
            self.W_emb = add_to_params(self.global_params,
                                       theano.shared(value=NormalInit(self.rng, self.idim, self.rankdim), name='W_emb'))

        # Variables to store encoder and decoder states
        if self.bidirectional_utterance_encoder:
            # Previous states variables
            self.ph_fwd = theano.shared(value=numpy.zeros((self.bs, self.qdim_encoder), dtype='float32'), name='ph_fwd')
            self.ph_bck = theano.shared(value=numpy.zeros((self.bs, self.qdim_encoder), dtype='float32'), name='ph_bck')
            self.phs = theano.shared(value=numpy.zeros((self.bs, self.sdim), dtype='float32'), name='phs')

            if self.direct_connection_between_encoders_and_decoder:
                self.phs_dummy = theano.shared(value=numpy.zeros((self.bs, self.qdim_encoder * 2), dtype='float32'),
                                               name='phs_dummy')

        else:
            # Previous states variables
            self.ph = theano.shared(value=numpy.zeros((self.bs, self.qdim_encoder), dtype='float32'), name='ph')
            self.phs = theano.shared(value=numpy.zeros((self.bs, self.sdim), dtype='float32'), name='phs')

            if self.direct_connection_between_encoders_and_decoder:
                self.phs_dummy = theano.shared(value=numpy.zeros((self.bs, self.qdim_encoder), dtype='float32'),
                                               name='phs_dummy')

        if self.utterance_decoder_gating == 'LSTM':
            self.phd = theano.shared(value=numpy.zeros((self.bs, self.qdim_decoder * 2), dtype='float32'), name='phd')
        else:
            self.phd = theano.shared(value=numpy.zeros((self.bs, self.qdim_decoder), dtype='float32'), name='phd')

        if self.add_latent_gaussian_per_utterance:
            self.platent_utterance_variable_prior = theano.shared(
                value=numpy.zeros((self.bs, self.latent_gaussian_per_utterance_dim), dtype='float32'),
                name='platent_utterance_variable_prior')
            self.platent_utterance_variable_approx_posterior = theano.shared(
                value=numpy.zeros((self.bs, self.latent_gaussian_per_utterance_dim), dtype='float32'),
                name='platent_utterance_variable_approx_posterior')

            if self.condition_latent_variable_on_dcgm_encoder:
                self.platent_dcgm_avg = theano.shared(value=numpy.zeros((self.bs, self.rankdim), dtype='float32'),
                                                      name='platent_dcgm_avg')
                self.platent_dcgm_n = theano.shared(value=numpy.zeros((1, self.bs), dtype='float32'),
                                                    name='platent_dcgm_n')

        # 于此之上是变量定义，于此之下是功能定义
        # Build utterance encoders
        if self.bidirectional_utterance_encoder:
            logger.debug("Initializing forward utterance encoder")
            self.utterance_encoder_forward = UtteranceEncoder(self.state, self.rng, self.W_emb, self, 'fwd')
            logger.debug("Build forward utterance encoder")
            res_forward = self.utterance_encoder_forward.build_encoder(training_x, xmask=training_hs_mask,
                                                                       prev_state=self.ph_fwd)

            logger.debug("Initializing backward utterance encoder")
            self.utterance_encoder_backward = UtteranceEncoder(self.state, self.rng, self.W_emb, self, 'bck')
            logger.debug("Build backward utterance encoder")
            res_backward = self.utterance_encoder_backward.build_encoder(training_x_reversed, xmask=training_hs_mask,
                                                                         prev_state=self.ph_bck)

            # The encoder h embedding is a concatenation of final states of the forward and backward encoder RNNs
            self.h = T.concatenate([res_forward, res_backward], axis=2)

        else:
            logger.debug("Initializing utterance encoder")
            self.utterance_encoder = UtteranceEncoder(self.state, self.rng, self.W_emb, self, 'fwd')

            logger.debug("Build utterance encoder")

            # The encoder h embedding is the final hidden state of the forward encoder RNN
            self.h = self.utterance_encoder.build_encoder(training_x, xmask=training_hs_mask, prev_state=self.ph)

        logger.debug("Initializing dialog encoder")
        self.dialog_encoder = DialogEncoder(self.state, self.rng, self, '')

        logger.debug("Build dialog encoder")
        self.hs = self.dialog_encoder.build_encoder(self.h, training_x, xmask=training_hs_mask, prev_state=self.phs)

        # We initialize the stochastic "latent" variables
        # platent_utterance_variable_prior
        if self.add_latent_gaussian_per_utterance:
            logger.debug("Initializing prior encoder for utterance-level latent variable")

            # First ,compute mask over latent Gaussian variables.
            # One means that a variable is part of the computational graph and zero that it's not.
            latent_variable_mask = T.eq(training_x, self.eos_sym) * training_x_cost_mask

            # We consider two kinds of prior: one case where the latent variable is
            # conditioned on the dialogue encoder, and one case where it is not conditioned on anything.
            if self.condition_latent_variable_on_dialogue_encoder:
                self.hs_to_condition_latent_variable_on = self.hs
            else:
                self.hs_to_condition_latent_variable_on = T.alloc(np.float32(0), self.hs.shape[0], self.hs.shape[1],
                                                                  self.hs.shape[2])

            self.latent_utterance_variable_prior_encoder = DialogLevelLatentEncoder(self.state, self.sdim,
                                                                                    self.latent_gaussian_per_utterance_dim,
                                                                                    self.rng, self,
                                                                                    'latent_utterance_prior')

            logger.debug("Build prior encoder for utterance-level latent variable")
            _prior_out = self.latent_utterance_variable_prior_encoder.build_encoder(
                self.hs_to_condition_latent_variable_on, training_x, xmask=training_hs_mask,
                latent_variable_mask=latent_variable_mask, prev_state=self.platent_utterance_variable_prior)

            self.latent_utterance_variable_prior = _prior_out[0]
            self.latent_utterance_variable_prior_mean = _prior_out[1]
            self.latent_utterance_variable_prior_var = _prior_out[2]

            logger.debug("Initializing approximate posterior encoder for utterance-level latent variable")
            if self.bidirectional_utterance_encoder and not self.condition_latent_variable_on_dcgm_encoder:
                posterior_input_size = self.sdim + self.qdim_encoder * 2
                # sdim should be the size of dialogue_hidden_state , and qdim should be the size of one_single_direction_uttrance_hidden_state
            else:
                posterior_input_size = self.sdim + self.qdim_encoder

            # Retrieve hidden state at the end of next utterance from the utterance encoders
            # (or at the end of the batch, if there are no end-of-token symbols at the end of the batch)
            if self.bidirectional_utterance_encoder:
                self.utterance_encoder_rolledleft = DialogLevelRollLeft(self.state, self.qdim_encoder, self.rng, self)
            else:
                self.utterance_encoder_rolledleft = DialogLevelRollLeft(self.state, self.qdim_encoder * 2, self.rng,
                                                                        self)

            if self.condition_latent_variable_on_dcgm_encoder:
                logger.debug("Initializing dcgm encoder for conditioning input to the utterance-level latent variable")

                self.dcgm_encoder = DCGMEncoder(self.state, self.rng, self.W_emb, self.qdim_encoder, self,
                                                'latent_dcgm_encoder')
                logger.debug("Build dcgm encoder")
                latent_dcgm_res, self.latent_dcgm_avg, self.latent_dcgm_n = self.dcgm_encoder.build_encoder(training_x,
                                                                                                            xmask=training_hs_mask,
                                                                                                            prev_state=[
                                                                                                                self.platent_dcgm_avg,
                                                                                                                self.platent_dcgm_n])

                self.h_future = self.utterance_encoder_rolledleft.build_encoder( \
                    latent_dcgm_res, \
                    training_x, \
                    xmask=training_hs_mask)

            else:
                self.h_future = self.utterance_encoder_rolledleft.build_encoder( \
                    self.h, \
                    training_x, \
                    xmask=training_hs_mask)

            self.latent_utterance_variable_approx_posterior_encoder = DialogLevelLatentEncoder(self.state,
                                                                                               posterior_input_size,
                                                                                               self.latent_gaussian_per_utterance_dim,
                                                                                               self.rng, self,
                                                                                               'latent_utterance_approx_posterior')

            self.hs_and_h_future = T.concatenate([self.hs_to_condition_latent_variable_on, self.h_future], axis=2)

            logger.debug("Build approximate posterior encoder for utterance-level latent variable")
            _posterior_out = self.latent_utterance_variable_approx_posterior_encoder.build_encoder( \
                self.hs_and_h_future, \
                training_x, \
                xmask=training_hs_mask, \
                latent_variable_mask=latent_variable_mask, \
                prev_state=self.platent_utterance_variable_approx_posterior)
            self.latent_utterance_variable_approx_posterior = _posterior_out[0]
            self.latent_utterance_variable_approx_posterior_mean = _posterior_out[1]
            self.latent_utterance_variable_approx_posterior_var = _posterior_out[2]

            self.latent_utterance_variable_approx_posterior_mean_var = T.sum(
                T.mean(self.latent_utterance_variable_approx_posterior_var, axis=2) * latent_variable_mask) / (T.sum(
                latent_variable_mask) + 0.0000001)
            # * self.x_cost_mask[1:self.x_max_length]) * (T.sum(T.eq(training_x, self.eos_sym)) / (T.sum(training_x_cost_mask_flat)))

            # Sample utterance latent variable from posterior
            self.posterior_sample = self.ran_cost_utterance[:(self.x_max_length - 1)] * T.sqrt(
                self.latent_utterance_variable_approx_posterior_var) + self.latent_utterance_variable_approx_posterior_mean

            # Compute KL divergence cost
            mean_diff_squared = (self.latent_utterance_variable_prior_mean \
                                 - self.latent_utterance_variable_approx_posterior_mean) ** 2

            logger.debug("Build KL divergence cost")
            kl_divergences_between_prior_and_posterior \
                = (T.sum(self.latent_utterance_variable_approx_posterior_var / self.latent_utterance_variable_prior_var,
                         axis=2) \
                   + T.sum(mean_diff_squared / self.latent_utterance_variable_prior_var, axis=2) \
                   - state['latent_gaussian_per_utterance_dim'] \
                   + T.sum(T.log(self.latent_utterance_variable_prior_var), axis=2) \
                   - T.sum(T.log(self.latent_utterance_variable_approx_posterior_var), axis=2) \
                   ) / 2

            self.kl_divergence_cost = kl_divergences_between_prior_and_posterior * latent_variable_mask
            self.kl_divergence_cost_acc = T.sum(self.kl_divergence_cost)

        else:
            # Set KL divergence cost to zero
            self.kl_divergence_cost = training_x_cost_mask * 0
            self.kl_divergence_cost_acc = theano.shared(value=numpy.float(0))
            self.latent_utterance_variable_approx_posterior_mean_var = theano.shared(value=numpy.float(0))

        # We initialize the decoder, and fix its word embeddings to that of the encoder(s)
        logger.debug("Initializing decoder")
        self.utterance_decoder = UtteranceDecoder(self.state, self.rng, self, self.dialog_encoder, self.W_emb)

        if self.direct_connection_between_encoders_and_decoder:
            logger.debug("Initializing dialog dummy encoder")
            if self.bidirectional_utterance_encoder:
                self.dialog_dummy_encoder = DialogDummyEncoder(self.state, self.rng, self, self.qdim_encoder * 2)
            else:
                self.dialog_dummy_encoder = DialogDummyEncoder(self.state, self.rng, self, self.qdim_encoder)

            logger.debug("Build dialog dummy encoder")
            self.hs_dummy = self.dialog_dummy_encoder.build_encoder(self.h, training_x, xmask=training_hs_mask,
                                                                    prev_state=self.phs_dummy)

            logger.debug("Build decoder (NCE) with direct connection from encoder(s)")
            if self.add_latent_gaussian_per_utterance:
                if self.condition_decoder_only_on_latent_variable:
                    self.hd_input = self.posterior_sample
                else:
                    self.hd_input = T.concatenate([self.hs, self.hs_dummy, self.posterior_sample], axis=2)
            else:
                self.hd_input = T.concatenate([self.hs, self.hs_dummy], axis=2)

            contrastive_cost, self.hd_nce = self.utterance_decoder.build_decoder(self.hd_input, training_x,
                                                                                 y_neg=self.y_neg, y=training_y,
                                                                                 xmask=training_hs_mask,
                                                                                 xdropmask=training_x_dropmask,
                                                                                 mode=UtteranceDecoder.NCE,
                                                                                 prev_state=self.phd)

            logger.debug("Build decoder (EVAL) with direct connection from encoder(s)")
            target_probs, self.hd, self.utterance_decoder_states, target_probs_full_matrix = self.utterance_decoder.build_decoder(
                self.hd_input, training_x, xmask=training_hs_mask, xdropmask=training_x_dropmask, y=training_y,
                mode=UtteranceDecoder.EVALUATION, prev_state=self.phd)

        else:
            if self.add_latent_gaussian_per_utterance:
                if self.condition_decoder_only_on_latent_variable:
                    self.hd_input = self.posterior_sample
                else:
                    self.hd_input = T.concatenate([self.hs, self.posterior_sample], axis=2)
            else:
                self.hd_input = self.hs

            logger.debug("Build decoder (NCE)")
            contrastive_cost, self.hd_nce = self.utterance_decoder.build_decoder(self.hd_input, training_x,
                                                                                 y_neg=self.y_neg, y=training_y,
                                                                                 xmask=training_hs_mask,
                                                                                 xdropmask=training_x_dropmask,
                                                                                 mode=UtteranceDecoder.NCE,
                                                                                 prev_state=self.phd)

            logger.debug("Build decoder (EVAL)")
            target_probs, self.hd, self.utterance_decoder_states, target_probs_full_matrix = self.utterance_decoder.build_decoder(
                self.hd_input, training_x, xmask=training_hs_mask, xdropmask=training_x_dropmask, y=training_y,
                mode=UtteranceDecoder.EVALUATION, prev_state=self.phd)

        # Prediction cost and rank cost
        self.contrastive_cost = T.sum(contrastive_cost.flatten() * training_x_cost_mask_flat)
        self.softmax_cost = -T.log(target_probs) * training_x_cost_mask_flat
        self.softmax_cost_acc = T.sum(self.softmax_cost)

        # Prediction accuracy
        self.training_misclassification = T.neq(T.argmax(target_probs_full_matrix, axis=2),
                                                training_y).flatten() * training_x_cost_mask_flat

        self.training_misclassification_acc = T.sum(self.training_misclassification)

        # Compute training cost, which equals standard cross-entropy error
        self.training_cost = self.softmax_cost_acc
        if self.use_nce:
            self.training_cost = self.contrastive_cost

        # Compute training cost as variational lower bound with possible annealing of KL-divergence term
        if self.add_latent_gaussian_per_utterance:
            if self.train_latent_gaussians_with_kl_divergence_annealing:
                self.evaluation_cost = self.training_cost + self.kl_divergence_cost_acc

                self.kl_divergence_cost_weight = add_to_params(self.global_params, theano.shared(value=numpy.float32(0),
                                                                                                 name='kl_divergence_cost_weight'))
                self.training_cost = self.training_cost + self.kl_divergence_cost_weight * self.kl_divergence_cost_acc
            else:
                self.training_cost += self.kl_divergence_cost_acc
                self.evaluation_cost = self.training_cost

            # Compute gradient of utterance decoder Wd_hh for debugging purposes
            self.grads_wrt_softmax_cost = T.grad(self.softmax_cost_acc, self.utterance_decoder.Wd_hh)
            if self.bidirectional_utterance_encoder:
                self.grads_wrt_kl_divergence_cost = T.grad(self.kl_divergence_cost_acc,
                                                           self.utterance_encoder_forward.W_in)
            else:
                self.grads_wrt_kl_divergence_cost = T.grad(self.kl_divergence_cost_acc, self.utterance_encoder.W_in)
        else:
            self.evaluation_cost = self.training_cost

        # Init params
        if self.collaps_to_standard_rnn:
            self.params = self.global_params + self.utterance_decoder.params
            assert len(set(self.params)) == (len(self.global_params) + len(self.utterance_decoder.params))
        else:
            if self.bidirectional_utterance_encoder:
                self.params = self.global_params + self.utterance_encoder_forward.params + self.utterance_encoder_backward.params + self.dialog_encoder.params + self.utterance_decoder.params
                assert len(set(self.params)) == (
                            len(self.global_params) + len(self.utterance_encoder_forward.params) + len(
                        self.utterance_encoder_backward.params) + len(self.dialog_encoder.params) + len(
                        self.utterance_decoder.params))
            else:
                self.params = self.global_params + self.utterance_encoder.params + self.dialog_encoder.params + self.utterance_decoder.params
                assert len(set(self.params)) == (len(self.global_params) + len(self.utterance_encoder.params) + len(
                    self.dialog_encoder.params) + len(self.utterance_decoder.params))

        if self.add_latent_gaussian_per_utterance:
            assert len(set(self.params)) + len(set(self.latent_utterance_variable_prior_encoder.params)) \
                   == len(set(self.params + self.latent_utterance_variable_prior_encoder.params))
            self.params += self.latent_utterance_variable_prior_encoder.params
            assert len(set(self.params)) + len(set(self.latent_utterance_variable_approx_posterior_encoder.params)) \
                   == len(set(self.params + self.latent_utterance_variable_approx_posterior_encoder.params))
            self.params += self.latent_utterance_variable_approx_posterior_encoder.params

            if self.condition_latent_variable_on_dcgm_encoder:
                assert len(set(self.params)) + len(set(self.dcgm_encoder.params)) \
                       == len(set(self.params + self.dcgm_encoder.params))
                self.params += self.dcgm_encoder.params

        # Create set of parameters to train
        self.params_to_train = []
        self.params_to_exclude = []
        if self.fix_encoder_parameters:
            # If the option fix_encoder_parameters is on, then we exclude all parameters
            # related to the utterance encoder(s) and dialogue encoder, including the word embeddings,
            # from the parameter training set.
            if self.bidirectional_utterance_encoder:
                self.params_to_exclude = self.global_params + self.utterance_encoder_forward.params + self.utterance_encoder_backward.params + self.dialog_encoder.params
            else:
                self.params_to_exclude = self.global_params + self.utterance_encoder.params + self.dialog_encoder.params

        if self.add_latent_gaussian_per_utterance:
            # We always need to exclude the KL-divergence term weight from training,
            # since this value is being annealed (and should therefore not be optimized with SGD).
            if self.train_latent_gaussians_with_kl_divergence_annealing:
                self.params_to_exclude += [self.kl_divergence_cost_weight]

        for param in self.params:
            if not param in self.params_to_exclude:
                self.params_to_train += [param]

        self.updates = self.compute_updates(self.training_cost / training_x.shape[1], self.params_to_train)

        # Truncate gradients properly by bringing forward previous states
        # First, create reset mask
        x_reset = self.x_reset_mask.dimshuffle(0, 'x')
        # if flag 'reset_hidden_states_between_subsequences' is on, then always reset
        if self.reset_hidden_states_between_subsequences:
            x_reset = 0

        # Next, compute updates using reset mask (this depends on the number of RNNs in the model)
        self.state_updates = []
        if self.bidirectional_utterance_encoder:
            self.state_updates.append((self.ph_fwd, x_reset * res_forward[-1]))
            self.state_updates.append((self.ph_bck, x_reset * res_backward[-1]))
            self.state_updates.append((self.phs, x_reset * self.hs[-1]))
            self.state_updates.append((self.phd, x_reset * self.hd[-1]))
        else:
            self.state_updates.append((self.ph, x_reset * self.h[-1]))
            self.state_updates.append((self.phs, x_reset * self.hs[-1]))
            self.state_updates.append((self.phd, x_reset * self.hd[-1]))

        if self.direct_connection_between_encoders_and_decoder:
            self.state_updates.append((self.phs_dummy, x_reset * self.hs_dummy[-1]))

        if self.add_latent_gaussian_per_utterance:
            self.state_updates.append(
                (self.platent_utterance_variable_prior, x_reset * self.latent_utterance_variable_prior[-1]))
            self.state_updates.append((self.platent_utterance_variable_approx_posterior,
                                       x_reset * self.latent_utterance_variable_approx_posterior[-1]))

            if self.condition_latent_variable_on_dcgm_encoder:
                self.state_updates.append((self.platent_dcgm_avg, x_reset * self.latent_dcgm_avg[-1]))
                self.state_updates.append((self.platent_dcgm_n, x_reset.T * self.latent_dcgm_n[-1]))

            if self.train_latent_gaussians_with_kl_divergence_annealing:
                self.state_updates.append((self.kl_divergence_cost_weight, T.minimum(1.0,
                                                                                     self.kl_divergence_cost_weight + self.kl_divergence_annealing_rate)))

        # Beam-search variables
        self.beam_x_data = T.imatrix('beam_x_data')
        self.beam_source = T.lvector("beam_source")
        # self.beam_source = T.imatrix("beam_source")
        #         self.x_data = T.imatrix('x_data')
        self.beam_hs = T.matrix("beam_hs")
        self.beam_step_num = T.lscalar("beam_step_num")
        self.beam_hd = T.matrix("beam_hd")
        self.beam_ran_cost_utterance = T.matrix('beam_ran_cost_utterance')
