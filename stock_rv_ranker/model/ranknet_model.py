import numpy as np
import tensorflow as tf




class RankNet_Model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_step(self, data):
        return super().train_step(data)




class RankNet_Lambda_Model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def train_step(self,data):
        return super().train_step(data)


    def apply_gradient_lambdarank(optimizer, model, x, score, mask, doc_cnt, eval_ndcg):
        with tf.GradientTape() as tape:
            oi = model(x)
        
        S_ij = tf.maximum(tf.minimum(tf.subtract(tf.expand_dims(score,1), score),1.),-1.)
        P_ij = tf.multiply(mask, tf.multiply(0.5, tf.add(1., S_ij)))
        P_ij_pred = tf.multiply(mask,tf.nn.sigmoid(tf.subtract(oi, tf.transpose(oi))))
        lambda_ij = tf.add(tf.negative(P_ij), P_ij_pred)
        
        ndcg, ndcg_delta_ij = eval_ndcg(score, tf.squeeze(oi, 1), return_ndcg_delta=True)
        lambda_ij = tf.multiply(lambda_ij, ndcg_delta_ij)
        
        lambda_i = tf.reduce_sum(lambda_ij,1) - tf.reduce_sum(lambda_ij,0)
        
        doi_dwk = tape.jacobian(oi, model.trainable_weights)
        
        # 1. reshape lambda_i to match the rank of the corresponding doi_dwk
        # 2. multiple reshaped lambda_i with the corresponding doi_dwk
        # 3. compute the sum across first 2 dimensions
        gradients = list(map(lambda k: 
                            tf.reduce_sum(tf.multiply(tf.reshape(lambda_i,  tf.concat([tf.shape(lambda_i),tf.ones(tf.rank(k) - 1, dtype=tf.int32)], axis=-1)), k), [0,1]),
                            doi_dwk))
        
        # model could still be trained without calculating the loss below
        loss_value = tf.reduce_sum(tf.keras.losses.binary_crossentropy(P_ij, P_ij_pred))
        loss_value = tf.multiply(loss_value, doc_cnt)
        
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        
        return oi, loss_value, ndcg