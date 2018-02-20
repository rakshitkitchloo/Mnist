
import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot =True)

x=tf.placeholder(tf.float32,shape=[None, 784])
y=tf.placeholder(tf.float32,shape=[None,10])

w1=tf.Variable(tf.truncated_normal([784,10]))
b1=tf.Variable(tf.truncated_normal([10]))

result=tf.add(tf.matmul(x,w1),b1)



sess=tf.Session()
init=tf.global_variables_initializer()

sess.run(init)


ab=tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=result)
cross_entropy=tf.reduce_mean(ab)

trainer=tf.train.GradientDescentOptimizer(0.3)
train_step=trainer.minimize(cross_entropy)

for i in range(5000):
	batch=mnist.train.next_batch(100)
	sess.run(train_step, {x: batch[0],y: batch[1]})
	cost=sess.run(cross_entropy,{x:batch[0],y:batch[1]})
	correct_prediction=tf.equal(tf.argmax(result,1),tf.argmax(y,1))
	accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

	precise=sess.run(accuracy,{x:mnist.train.images,y:mnist.train.labels})
	precise2=sess.run(accuracy,{x:mnist.test.images,y:mnist.test.labels})

	print "epoch: ",i," cost: ",cost," Train accuracy: ",precise," Test Accuracy: ",precise2

correct_prediction=tf.equal(tf.argmax(result,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#print sess.run(correct_prediction,{x:mnist.test.images, y:mnist.test.labels})
print sess.run(accuracy, {x:mnist.test.images, y:mnist.test.labels})

