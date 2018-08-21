# -*- coding:utf-8 -*-


# DialogEncoderDecoder 是一个总管的类，
# 所有其他的类，在这个类的内部进行交互和动作
model = DialogEncoderDecoder(state)
eval_batch = model.build_eval_function()
eval_grads = model.build_eval_grads()
train_batch = model.build_train_function()
eval_grads = model.build_eval_grads()
c, kl_divergence_cost, posterior_mean_variance = train_batch(x_data, x_data_reversed, max_length, x_cost_mask, x_reset,
                                                             ran_cost_utterance, ran_decoder_drop_mask)



