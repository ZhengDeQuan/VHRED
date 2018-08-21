# -*- coding:utf-8 -*-

class UtteranceEncoder(EncoderDecoderBase):
    """
    This is the GRU-gated RNN encoder class, which operates on hidden states at the word level (intra-utterance level).
    It encodes utterances into real-valued fixed-sized vectors.
    """

    def init_params(self, word_embedding_param):
        # Initialzie W_emb to given word embeddings
        assert (word_embedding_param != None)
        self.W_emb = word_embedding_param

        """ sent weights """
        self.W_in = add_to_params(self.params,
                                  theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim_encoder),
                                                name='W_in' + self.name))
        self.W_hh = add_to_params(self.params,
                                  theano.shared(value=OrthogonalInit(self.rng, self.qdim_encoder, self.qdim_encoder),
                                                name='W_hh' + self.name))
        self.b_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_encoder,), dtype='float32'),
                                                             name='b_hh' + self.name))

        if self.utterance_encoder_gating == "GRU":
            self.W_in_r = add_to_params(self.params,
                                        theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim_encoder),
                                                      name='W_in_r' + self.name))
            self.W_in_z = add_to_params(self.params,
                                        theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim_encoder),
                                                      name='W_in_z' + self.name))
            self.W_hh_r = add_to_params(self.params, theano.shared(
                value=OrthogonalInit(self.rng, self.qdim_encoder, self.qdim_encoder), name='W_hh_r' + self.name))
            self.W_hh_z = add_to_params(self.params, theano.shared(
                value=OrthogonalInit(self.rng, self.qdim_encoder, self.qdim_encoder), name='W_hh_z' + self.name))
            self.b_z = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_encoder,), dtype='float32'),
                                                                name='b_z' + self.name))
            self.b_r = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_encoder,), dtype='float32'),
                                                                name='b_r' + self.name))

    # This function takes as input word indices and extracts their corresponding word embeddings
    def approx_embedder(self, x):
        return self.W_emb[x]

    def plain_sent_step(self, x_t, m_t, *args):
        args = iter(args)
        h_tm1 = next(args)

        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')  # 从 N 变成 N x 1

        # If 'reset_utterance_encoder_at_end_of_utterance' flag is on,
        # then reset the hidden state if this is an end-of-utterance token
        # as given by m_t
        if self.reset_utterance_encoder_at_end_of_utterance:
            hr_tm1 = m_t * h_tm1
        else:
            hr_tm1 = h_tm1

        h_t = self.sent_rec_activation(T.dot(x_t, self.W_in) + T.dot(hr_tm1, self.W_hh) + self.b_hh)

        # Return hidden state only
        return [h_t]

    def GRU_sent_step(self, x_t, m_t, *args):
        args = iter(args)
        h_tm1 = next(args)

        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')

            # If 'reset_utterance_encoder_at_end_of_utterance' flag is on,
        # then reset the hidden state if this is an end-of-utterance token
        # as given by m_t
        if self.reset_utterance_encoder_at_end_of_utterance:
            hr_tm1 = m_t * h_tm1
        else:
            hr_tm1 = h_tm1

        r_t = T.nnet.sigmoid(T.dot(x_t, self.W_in_r) + T.dot(hr_tm1, self.W_hh_r) + self.b_r)
        z_t = T.nnet.sigmoid(T.dot(x_t, self.W_in_z) + T.dot(hr_tm1, self.W_hh_z) + self.b_z)
        h_tilde = self.sent_rec_activation(T.dot(x_t, self.W_in) + T.dot(r_t * hr_tm1, self.W_hh) + self.b_hh)
        h_t = (np.float32(1.0) - z_t) * hr_tm1 + z_t * h_tilde

        # return both reset state and non-reset state
        return [h_t, r_t, z_t, h_tilde]

    def build_encoder(self, x, xmask=None, prev_state=None, **kwargs):
        #外部调用
        #res_forward = self.utterance_encoder_forward.build_encoder(training_x, xmask=training_hs_mask, prev_state=self.ph_fwd)
        one_step = False
        if len(kwargs):
            one_step = True

        # if x.ndim == 2 then
        # x = (n_steps, batch_size)
        if x.ndim == 2:
            batch_size = x.shape[1]
        # else x = (word_1, word_2, word_3, ...)
        # or x = (last_word_1, last_word_2, last_word_3, ..)
        # in this case batch_size is
        else:
            batch_size = 1

        # if it is not one_step then we initialize everything to previous state or zero
        if not one_step:
            if prev_state:
                h_0 = prev_state
            else:
                h_0 = T.alloc(np.float32(0), batch_size, self.qdim_encoder)

        # in sampling mode (i.e. one step) we require
        else:
            # in this case x.ndim != 2
            assert x.ndim != 2
            assert 'prev_h' in kwargs
            h_0 = kwargs['prev_h']

        # We extract the word embeddings from the word indices
        xe = self.approx_embedder(x)
        if xmask == None:
            xmask = T.neq(x, self.eos_sym)

        # We add ones at the the beginning of the reset vector to align the resets with y_training:
        # for example for
        # training_x =        </s> a b c </s> d
        # xmask =               0  1 1 1  0   1
        # rolled_xmask =        1  0 1 1  1   0 1
        # Thus, we ensure that the no information in the encoder is carried from input "</s>" to "a",
        # or from "</s>" to "d".
        # Now, the state at exactly </s> always reflects the previous utterance encoding.
        # Since the dialogue encoder uses xmask, and inputs it when xmask=0, it will input the utterance encoding
        # exactly on the </s> state.

        if xmask.ndim == 2:
            # ones_vector = theano.shared(value=numpy.ones((1, self.bs), dtype='float32'), name='ones_vector')
            ones_vector = T.ones_like(xmask[0, :]).dimshuffle('x', 0)
            rolled_xmask = T.concatenate([ones_vector, xmask], axis=0)
        else:
            ones_scalar = theano.shared(value=numpy.ones((1), dtype='float32'), name='ones_scalar')
            rolled_xmask = T.concatenate([ones_scalar, xmask])

        # GRU Encoder
        if self.utterance_encoder_gating == "GRU":
            f_enc = self.GRU_sent_step
            o_enc_info = [h_0, None, None, None]

        else:
            f_enc = self.plain_sent_step
            o_enc_info = [h_0]

        # Run through all the utterances (encode everything)
        if not one_step:
            _res, _ = theano.scan(f_enc,
                                  sequences=[xe, rolled_xmask], \
                                  outputs_info=o_enc_info)
        else:  # Make just one step further
            _res = f_enc(xe, rolled_xmask, [h_0])[0]

        # Get the hidden state sequence
        if self.utterance_encoder_gating != 'GRU':
            h = _res
        else:
            h = _res[0]

        return h

    def __init__(self, state, rng, word_embedding_param, parent, name):
        #外部调用
        #self.utterance_encoder_forward = UtteranceEncoder(self.state, self.rng, self.W_emb, self, 'fwd')
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.name = name
        self.init_params(word_embedding_param)
