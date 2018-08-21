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



#调用
train_batch = model.build_train_function(
                                                x_data, x_data_reversed,
                                                max_length, x_cost_mask,
                                                x_reset,
                                                ran_cost_utterance, ran_decoder_drop_mask

 )