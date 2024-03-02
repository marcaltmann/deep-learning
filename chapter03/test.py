import tensorflow as tf

input_var = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:
    result = input_var * 2 + 10
gradient = tape.gradient(result, input_var)
