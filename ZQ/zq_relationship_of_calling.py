# -*- coding:utf-8 -*-
总纲：
在DialogEncoderDecoder这个类的init函数中，其余的类的实例生成顺序，以及生成的实例的初次的调用顺序，（也可能在初次的调用后就没有下一次的调用了，）按照下面所列举的顺序进行。
以下这些类在调用了各自的init函数后，显示调用的第一个函数都是，build_encoder()，最后的那个decoder类除外，它调用的是build_decoder()

1.UtteranceEncoder
  UtteranceEncoder.build_encoder


2.
logger.debug("Initializing dialog encoder")
self.dialog_encoder = DialogEncoder(self.state, self.rng, self, '')

logger.debug("Build dialog encoder")
self.hs = self.dialog_encoder.build_encoder(self.h, training_x, xmask=training_hs_mask, prev_state=self.phs)

3.DialogLevelLatentEncoder
  DialogLevelLatentEncoder.build_encoder
self.latent_utterance_variable_prior_encoder = DialogLevelLatentEncoder(self.state,
                                                                        self.sdim,
                                                                        self.latent_gaussian_per_utterance_dim,
                                                                        self.rng,
                                                                        self,
                                                                        'latent_utterance_prior')

logger.debug("Build prior encoder for utterance-level latent variable")
_prior_out = self.latent_utterance_variable_prior_encoder.build_encoder(
                        self.hs_to_condition_latent_variable_on,
                        training_x,
                        xmask=training_hs_mask,
                        latent_variable_mask=latent_variable_mask,
                        prev_state=self.platent_utterance_variable_prior)

self.latent_utterance_variable_prior = _prior_out[0]
self.latent_utterance_variable_prior_mean = _prior_out[1]
self.latent_utterance_variable_prior_var = _prior_out[2]

# Retrieve hidden state at the end of next utterance from the utterance encoders
# (or at the end of the batch, if there are no end-of-token symbols at the end of the batch)
4.DialogLevelRollLeft
  DialogLevelRollLeft.build_encoder



5.DialogLevelLatentEncoder
  DialogLevelLatentEncoder.build_encoder
self.latent_utterance_variable_approx_posterior_encoder = DialogLevelLatentEncoder(self.state,
                                                                                   posterior_input_size,
                                                                                   self.latent_gaussian_per_utterance_dim,
                                                                                   self.rng,
                                                                                   self,
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


6.UtteranceDecoder
  UtteranceDecoder.build_decoder()
logger.debug("Initializing decoder")

logger.debug("Build decoder (NCE)")
self.utterance_decoder = UtteranceDecoder(self.state, self.rng, self, self.dialog_encoder, self.W_emb)
contrastive_cost, self.hd_nce = self.utterance_decoder.build_decoder(
                                        self.hd_input,
                                        training_x,
                                        y_neg=self.y_neg,
                                        y=training_y,
                                        xmask=training_hs_mask,
                                        xdropmask=training_x_dropmask,
                                        mode=UtteranceDecoder.NCE,
                                        prev_state=self.phd)

logger.debug("Build decoder (EVAL)")
target_probs, self.hd, self.utterance_decoder_states, target_probs_full_matrix =
self.utterance_decoder.build_decoder(self.hd_input, training_x, xmask=training_hs_mask,
                                         xdropmask=training_x_dropmask, y=training_y,
                                         mode=UtteranceDecoder.EVALUATION, prev_state=self.phd)
