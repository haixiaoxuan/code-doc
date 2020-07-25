## tensorflow 2.0



#### 常量

常量被赋值之后就不能被改变，重新赋值相当于创建新的内存空间

```python
# 字符串常量
# dtype -> tf.int64 tf.float32 tf.double tf.string tf.bool
a = tf.constant("hello")
b = tf.constant("tensorflow2")
c = tf.strings.join([a, b], " ")
```







#### 变量

```python
v = tf.Variable([1.0,2.0],name = "v")
```







#### tensor op

##### 基本操作

```python
# 求tensor的维度
tf.rank(xx)

# 将tensor转变为numpy array
xx.numpy()

# 改变tensor数据类型
tf.cast(xx, tf.float32)

# 查看shape
xx.shape
tf.reshape(xx, ())

# 对变量执行加法操作
v.assign_add([1.0,1.0])
v.assign(xx)	# 对v值进行替换

# 打印操作
tf.print()

# random
tf.random.set_seed(1.0)
tf.random.uniform([400,2],minval=-10,maxval=10)	# 均匀分布
tf.random.normal([400,1],mean = 0.0,stddev= 2.0)	# 正态分布
# 正态分布随机，剔除两倍方差以外的数据重新生成
tf.random.truncated_normal((5,5), mean=0.0, stddev=1.0, dtype=tf.float32)

# 矩阵乘法
@
tf.transpose	# 转置

# 删掉tensor中所有维度为1的维度
tf.squeeze

# sample
tf.range(1,10, delta = 2)
tf.linspace(0.0, 2*3.14, 100)
tf.zeros([3,3])
tf.ones([3,3])
tf.fill([3,2],5)
tf.eye(3,3) 			#单位矩阵
tf.linalg.diag([1,2,3]) #对⻆阵
```

##### 索引切片

```python
t.shape = (5, 5)
t[0]	# 第0行
t[-1]	# 倒数第一行
t[1,3]	# 第一行第三列, 等价于t[1][3]
t[1:4,:]		# 第一行至第三行
t[1:4,:4:2]		# 第1⾏⾄第三行，第0列到第三列每两列取⼀列
x[1,:].assign(tf.constant([0.0,0.0]))	# 可以对变量进行赋值
a[...,1]		# 省略号可以表示多个冒号



# 不规则切片 ⽤tf.gather,tf.gather_nd,tf.boolean_mask
tf.gather(scores,[0,5,9],axis=1)	# 抽取第二个维度中的0，5，9， 后面维度的全部
# 抽取第二个维度0，5，9 第三个维度1，3，6
tf.gather(tf.gather(scores,[0,5,9],axis=1),[1,3,6],axis=2)
# 抽取第一第二维度分别为 (0,0),(2,4),(3,6)
tf.gather_nd(scores,indices = [(0,0),(2,4),(3,6)])
tf.boolean_mask(c,c<0) 	# 等价于c[c<0]，找出比c小的



# 不规则切片只能获取tensor的部分元素，并不能修改，如果要修改tensor的部分元素得到新的tensor，
# 使用 tf.where tf.scatter_nd
tf.where(c<0, tf.fill(c.shape,np.nan), c)
tf.where(c<0)	# 返回所有满足条件的位置坐标
# 将指定位置替换为指定元素，其余全为0
tf.scatter_nd([[0,0],[2,1]],[c[0,0],c[2,1]],c.shape)

indices = tf.where(c<0)
tf.scatter_nd(indices,tf.gather_nd(c,indices),c.shape)
```

##### 维度转变

```python
"""
	tf.reshape 可以改变张量的形状。
	tf.squeeze 可以减少维度。如果张量在某个维度为1，利⽤tf.squeeze可以消除这个维度.
	tf.expand_dims 可以增加维度。
	tf.transpose 可以交换维度。
"""

tf.reshape(b,[1,3,3,2])
tf.expand_dims(s,axis=0) #在第0维插⼊⻓度为1的⼀个维度
# 交换维度顺序，分别为 第3维，第1维，第2维，第0维
tf.transpose(a,perm=[3,1,2,0])
```

##### 合并分割

```python
"""
	合并：tf.concat，tf.stack
		 tf.concat是连接，不会增加维度，⽽tf.stack是堆叠，会增加维度。
	分割：tf.split
		 split是concat的逆运算
"""
a = tf.constant([[1.0,2.0],[3.0,4.0]])
b = tf.constant([[5.0,6.0],[7.0,8.0]])
tf.concat([a,b],axis = 0)		# 对第0维进行合并 -> shape = [4, 2]
tf.stack([a,b,c], axis = 0)				# shape = [2,2,2]

tf.split(c,3,axis = 0) 	# 指定分割份数，平均分割
tf.split(c,[2,2,2],axis = 0) 	# 指定每份的记录数量
```

##### 数据运算

```python
# 标量运算符
a+b
a-b
a*b
a/b
a**2
a**(0.5)
a%3 	# mod的运算符᯿载，等价于m = tf.math.mod(a,3)
a//3 	# 整除，向下取整
(a>=2)	# 返回 bool 类型的tensor
(a>=2)&(a<=3)
(a>=2)|(a<=3)
a==5 	# tf.equal(a,5)
tf.sqrt(a)	# 开根号
tf.add_n([a,b,c])		# 对a,b,c执行加法操作
tf.maximum(a,b)		# 对每一个维度求出最大值
tf.minimum(a,b)



# 向量运算符
tf.reduce_sum(a)
tf.reduce_mean(a)
tf.reduce_max(a)
tf.reduce_min(a)
tf.reduce_prod(a)	# 累乘
tf.reduce_sum(b, axis=1, keepdims=True)	# 指定维度进行sum

tf.reduce_all(p)	# 对bool类型进行reduce，
tf.reduce_any(q)

# 使用tf.foldr实现reduce_sum
tf.foldr(lambda a,b:a+b,tf.range(10))
# cum扫描累积
tf.math.cumsum(a)		# [1 3 6 ... 28 36 45]
tf.math.cumprod(a)
tf.argmax(a)
#tf.math.top_k可以⽤于对张量排序
values,indices = tf.math.top_k(a,3,sorted=True)



# 矩阵运算
a@b 	# 等价于tf.matmul(a,b)
tf.transpose(a)		# 转置
tf.linalg.inv(a)	# 矩阵求逆，必须为tf.float32或tf.double类型
tf.linalg.trace(a)	# 矩阵求trace，对角元素的和
tf.linalg.norm(a)	# 求矩阵的二范式
tf.linalg.det(a)	# 矩阵行列式
tf.linalg.eigvalsh(a)	# 矩阵特征值
q,r = tf.linalg.qr(a)	# 矩阵qr分解
tf.print(q)
tf.print(r)
v,s,d = tf.linalg.svd(a)	# 矩阵svd分解
tf.matmul(tf.matmul(s,tf.linalg.diag(v)),d)	# v为奇异值
```

##### 广播机制

```python
"""
	1、如果张量的维度不同，将维度较⼩的张量进⾏扩展，直到两个张量的维度都⼀样。
	2、如果两个张量在某个维度上的⻓度是相同的，或者其中⼀个张量在该维度上的⻓度为1，那么
		我们就说这两个张量在该维度上是相容的。
	3、如果两个张量在所有维度上都是相容的，它们就能使⽤⼴播。
	4、⼴播之后，每个维度的⻓度将取两个张量在该维度⻓度的较⼤值。
	5、在任何⼀个维度上，如果⼀个张量的⻓度为1，另⼀个张量⻓度⼤于1，那么在该维度上，就好
		像是对第⼀个张量进⾏了复制。
"""

# 扩展tensor维度
a = tf.constant([1,2,3])
b = tf.constant([[0,0,0],[1,1,1],[2,2,2]])
b + a #等价于 b + tf.broadcast_to(a,b.shape)

# tf.broadcast_to(a,b.shape)
# array([[1, 2, 3],
#  		 [1, 2, 3],
#  		 [1, 2, 3]], dtype=int32)>

#计算⼴播后计算结果的形状，静态形状，TensorShape类型参数
tf.broadcast_static_shape(a.shape,b.shape)

#计算⼴播后计算结果的形状，动态形状，Tensor类型参数
c = tf.constant([1,2,3])
d = tf.constant([[1],[2],[3]])
tf.broadcast_dynamic_shape(tf.shape(c),tf.shape(d))
```







#### Autograph

​	实践中，⼀般会先⽤动态计算图调试代码，然后在需要提⾼性能的的地⽅利⽤@tf.function切换成
Autograph获得更⾼的效率。

​	```编码规范```

​	①，被@tf.function修饰的函数应尽可能使⽤TensorFlow中的函数⽽不是Python中的其他函数。例
如使⽤tf.print⽽不是print，使⽤tf.range⽽不是range，使⽤tf.constant(True)⽽不是True.
​	②，避免在@tf.function修饰的函数内部定义tf.Variable.
​	③，被@tf.function修饰的函数不可修改该函数外部的Python列表或字典等数据结构变量。

**note:** 如果调⽤被@tf.function装饰的函数时输⼊的参数不是Tensor类型，则每次都会重新
创建计算图。

![1593658538606](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1593658538606.png)

![1593658562596](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1593658562596.png)

```python
@tf.function
def strjoin(x,y):
 	z = tf.strings.join([x,y],separator = " ")
 	tf.print(z)
 	return z


# log的使用
logdir = './data/autograph'
writer = tf.summary.create_file_writer(logdir)

#开启autograph跟踪
tf.summary.trace_on(graph=True, profiler=True)

#执⾏autograph
result = strjoin("hello","world")

#将计算图信息写⼊⽇志
with writer.as_default():
 	tf.summary.trace_export(
 	name="autograph",
 	step=0,
 	profiler_outdir=logdir)
```

##### tf.Model

![1593659384085](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1593659384085.png)

```python
x = tf.Variable(1.0,dtype=tf.float32)
#在tf.function中⽤input_signature限定输⼊张量的签名类型：shape和dtype
@tf.function(input_signature=[tf.TensorSpec(shape = [], dtype = tf.float32)])
def add_print(a):
 	x.assign_add(a)
 	tf.print(x)
 	return(x)
# 如果输入类型不正确的tensor会报错




# 使用tf.Model 进行封装
class DemoModule(tf.Module):
 	def __init__(self,init_value = tf.constant(0.0),name=None):
 		super(DemoModule, self).__init__(name=name)
 		with self.name_scope: #相当于with tf.name_scope("demo_module")
 			self.x = tf.Variable(init_value,dtype = tf.float32,trainable=True)

 @tf.function(input_signature=[tf.TensorSpec(shape = [], dtype =tf.float32)])
 def addprint(self,a):
 	with self.name_scope:
 		self.x.assign_add(a)
 		tf.print(self.x)
 		return(self.x)
demo = DemoModule(init_value = tf.constant(1.0))
result = demo.addprint(tf.constant(5.0))

#查看模块中的全部变量和全部可训练变量
print(demo.variables)
print(demo.trainable_variables)

#查看模块中的全部⼦模块
demo.submodules

#使⽤tf.saved_model 保存模型，并指定需要跨平台部署的⽅法
tf.saved_model.save(demo,"./data/demo/1",signatures ={"serving_default":demo.addprint})

#加载模型
demo2 = tf.saved_model.load("./data/demo/1")
demo2.addprint(tf.constant(5.0))




# 除了利⽤tf.Module的⼦类化实现封装，我们也可以通过给tf.Module添加属性的⽅法进⾏封装。
mymodule = tf.Module()
mymodule.x = tf.Variable(0.0)
@tf.function(input_signature=[tf.TensorSpec(shape = [], dtype = tf.float32)])
def addprint(a):
 	mymodule.x.assign_add(a)
 	tf.print(mymodule.x)
 	return (mymodule.x)
mymodule.addprint = addprint

```

![1593660081739](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1593660081739.png)







#### 自动微分

```python
# 对变量求导
x = tf.Variable(0.0, name = "x",dtype = tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)
with tf.GradientTape() as tape:
 	y = a*tf.pow(x,2) + b*x + c

dy_dx = tape.gradient(y,x)
print(dy_dx)


# 对常量tensor求导， 需要增加watch
with tf.GradientTape() as tape:
 	tape.watch([a,b,c])
 	y = a*tf.pow(x,2) + b*x + c

dy_dx,dy_da,dy_db,dy_dc = tape.gradient(y,[x,a,b,c])


# 求⼆阶导数
with tf.GradientTape() as tape2:
 	with tf.GradientTape() as tape1:
 		y = a*tf.pow(x,2) + b*x + c
 	dy_dx = tape1.gradient(y,x)
dy2_dx2 = tape2.gradient(dy_dx,x)
print(dy2_dx2)


# 利用梯度计算
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for _ in range(1000):
 	with tf.GradientTape() as tape:
 	y = a*tf.pow(x,2) + b*x + c
 	dy_dx = tape.gradient(y,x)
 	optimizer.apply_gradients(grads_and_vars=[(dy_dx,x)])
# 等价于 optimizer.minimize， 相当于先求梯度再apply_gradients
@tf.function
def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a * tf.pow(x, 2) + b * x + c
    return y
for _ in range(1000):
 	optimizer.minimize(f,[x]) 
```







#### 模型定义

三种方式：

​	① 使⽤Sequential按层顺序构建模型

​	② 使⽤函数式API构建任意结构模型

​	③ 继承Model基类构建⾃定义模型



①

```python
model = models.Sequential()
model.add(layers.Dense(20,activation = 'relu',input_shape=(15,)))
model.add(layers.Dense(10,activation = 'relu' ))
model.add(layers.Dense(1,activation = 'sigmoid' ))

model.summary()
```

②

```python
inputs = layers.Input(shape=(32,32,3))
x = layers.Conv2D(32,kernel_size=(3,3))(inputs)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(64,kernel_size=(5,5))(x)
x = layers.MaxPool2D()(x)
x = layers.Dropout(rate=0.1)(x)
x = layers.Flatten()(x)
x = layers.Dense(32,activation='relu')(x)
outputs = layers.Dense(1,activation = 'sigmoid')(x)
model = models.Model(inputs = inputs,outputs = outputs)
```

③

```python
class CnnModel(models.Model):
    def __init__(self):
        super(CnnModel, self).__init__()

    def build(self, input_shape):
        self.embedding = layers.Embedding(MAX_WORDS, 7, input_length=MAX_LEN)
        self.conv_1 = layers.Conv1D(16, kernel_size=5, name="conv_1", activation="relu")
        self.pool = layers.MaxPool1D()
        self.conv_2 = layers.Conv1D(128, kernel_size=2, name="conv_2", activation="relu")
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1, activation="sigmoid")
        super(CnnModel, self).build(input_shape)

    def call(self, x):
        x = self.embedding(x)
        x = self.conv_1(x)
        x = self.pool(x)
        x = self.conv_2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return (x)


model = CnnModel()
model.build(input_shape=(None, MAX_LEN))
model.summary()

```



##### 定义不可训练的层

```python
model.layers[0].trainable = False #冻结第0层的变量,使其不可训练
```





#### 训练模型

三种方式：

​	①内置fit⽅法

​	②内置train_on_batch⽅法

​	③以及⾃定义训练循环

```python
# ⼆分类问题选择⼆元交叉熵损失函数
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
history = model.fit(x_train,y_train,
 					batch_size= 64,
 					epochs= 30,
 					validation_split=0.2 #分割⼀部分训练数据⽤于验证
 					)


"""
	compile参数：
	optimizer: 
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
		optimizer=optimizers.Nadam()
		optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
		
	loss:
		loss=tf.keras.losses.binary_crossentropy
		loss = losses.mean_squared_error(Y, y_hat)	# mse

	metrics:
		metrics=["accuracy", "AUC", "mae"]
		valid_loss = metrics.Mean(name='valid_loss')
		valid_metric = metrics.BinaryAccuracy(name='valid_accuracy')
		
"""

"""
	fit参数：
		epochs
		validation_data
		callbacks=[]
		workers=4
"""
```

train_on_batch

```python
# 该内置⽅法相⽐较fit⽅法更加灵活，可以不通过回调函数⽽直接在批次层次上更加精细地控制训练的过程。

def train_model(model,ds_train,ds_valid,epoches):
 	for epoch in tf.range(1,epoches+1):
 		model.reset_metrics()

 		# 在后期降低学习率
 		if epoch == 5:
 			model.optimizer.lr.assign(model.optimizer.lr/2.0)
 			tf.print("Lowering optimizer Learning Rate...\n\n")

 		for x, y in ds_train:
 			train_result = model.train_on_batch(x, y)
 		for x, y in ds_valid:
 			valid_result = model.test_on_batch(x, y,reset_metrics=False)

 		if epoch%1 ==0:
 			tf.print("epoch = ",epoch)
 			print("train:",dict(zip(model.metrics_names,train_result)))
 			print("valid:",dict(zip(model.metrics_names,valid_result)))
 			print("")
```

自定义循环训练

```python
optimizer = optimizers.Nadam()
loss_func = losses.SparseCategoricalCrossentropy()

train_loss = metrics.Mean(name='train_loss')
train_metric = metrics.SparseCategoricalAccuracy(name='train_accuracy')

valid_loss = metrics.Mean(name='valid_loss')
valid_metric = metrics.SparseCategoricalAccuracy(name='valid_accuracy')

@tf.function
def train_step(model, features, labels):
	with tf.GradientTape() as tape:
 		predictions = model(features,training = True)
 		loss = loss_func(labels, predictions)
 	gradients = tape.gradient(loss, model.trainable_variables)
 	optimizer.apply_gradients(zip(gradients, model.trainable_variables))
 	
    train_loss.update_state(loss)
 	train_metric.update_state(labels, predictions)

@tf.function
def valid_step(model, features, labels):
 	predictions = model(features)
 	batch_loss = loss_func(labels, predictions)
 	valid_loss.update_state(batch_loss)
 	valid_metric.update_state(labels, predictions)

def train_model(model,ds_train,ds_valid,epochs):
 	for epoch in tf.range(1,epochs+1):

 		for features, labels in ds_train:
 			train_step(model,features,labels)
 		for features, labels in ds_valid:
 			valid_step(model,features,labels)

 		logs = 'Epoch={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}'

 		if epoch%1 ==0:
 			tf.print(tf.strings.format(logs,
									(epoch,train_loss.result(),
                                     train_metric.result(),
                                     valid_loss.result(),valid_metric.result())))
 			tf.print("")

 		train_loss.reset_states()
 		valid_loss.reset_states()
 		train_metric.reset_states()
 		valid_metric.reset_states()
        
train_model(model,ds_train,ds_test,10)
```











#### callback

##### tensorboard

```python
logdir = "./data/keras_model/logs")
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

# 运行tensorboard
from tensorboard import notebook
notebook.list()
notebook.start("--logdir ./data/keras_model")
```

##### earlystopping

```python
# 当loss在200个epoch后没有提升，则提前终⽌训练。
stop_callback = tf.keras.callbacks.EarlyStopping(monitor = "loss", patience=200)
```

##### 动态改变学习率

```python
# 如果loss在100个epoch后没有提升，学习率减半。
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss",
factor =0.5, patience = 100)
```









#### 模型评估

```python
import matplotlib.pyplot as plt
def plot_metric(history, metric):
	train_metrics = history.history[metric]
	val_metrics = history.history['val_'+metric]
	epochs = range(1, len(train_metrics) + 1)
	plt.plot(epochs, train_metrics, 'bo--')
	plt.plot(epochs, val_metrics, 'ro-')
	plt.title('Training and validation '+ metric)
	plt.xlabel("Epochs")
	plt.ylabel(metric)
	plt.legend(["train_"+metric, 'val_'+metric])
	plt.show()
	
plot_metric(history,"loss")
plot_metric(history,"AUC")

val_loss,val_accuracy = model.evaluate(x = x_test,y = y_test)

"""
	evaluate参数：
		workers=4
"""
```







#### 使用模型

```python
# 预测概率
model.predict(x_test[0:10])
# model(tf.constant(x_test[0:10].values,dtype = tf.float32)) #等价写法

# 预测类别
model.predict_classes(x_test[0:10])

# 批量预测
model.predict_on_batch(x[0:20])
```









#### 保存模型

​	① keras方式保存

​	② tensorflow原生方式 ( 支持跨平台 )



①

```python
# 保存模型结构及权重
model.save('./data/keras_model.h5') 
# 重新加载模型
model = models.load_model('./data/keras_model.h5')
model.evaluate(x_test,y_test)


# 保存模型结构
json_str = model.to_json()
# 恢复模型结构
model_json = models.model_from_json(json_str)


#保存模型权重
model.save_weights('./data/keras_model_weight.h5')
# 恢复模型结构
model_json = models.model_from_json(json_str)
model_json.compile(
    optimizer='adam',
 	loss='binary_crossentropy',
	metrics=['AUC']
 )
# 加载权重
model_json.load_weights('./data/keras_model_weight.h5')
model_json.evaluate(x_test,y_test)
```



②

```python
# 保存权重，该⽅式仅仅保存权重张量
model.save_weights('./data/tf_model_weights.ckpt', save_format = "tf")

# 保存模型结构与模型参数到⽂件,该⽅式保存的模型具有跨平台性便于部署
model.save('./data/tf_model_savedmodel', save_format="tf")
model_loaded = tf.keras.models.load_model('./data/tf_model_savedmodel')
model_loaded.evaluate(x_test,y_test)
```





#### 数据处理

##### 图片数据处理

​	① 使用tf.keras.ImageGenerator工具构建图片数据生成器

​	② 使用tf.keras.Dataset搭配tf.image中的一些处理方法构建数据管道



②

```python
def load_image(img_path, size=(32, 32)):
    label = tf.constant(1, tf.int8) if tf.strings.regex_full_match(img_path, ".*/automobile/.*") \
        else tf.constant(0, tf.int8)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)  # 注意此处为jpeg格式
    img = tf.image.resize(img, size) / 255.0
    return (img, label)

# 使⽤并⾏化预处理num_parallel_calls 和预存数据prefetch来提升性能
ds_train = tf.data.Dataset.list_files("./data/cifar2/train/*/*.jpg"). \
    map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
    shuffle(buffer_size=1000).batch(BATCH_SIZE). \
    prefetch(tf.data.experimental.AUTOTUNE)
```



##### 文本数据处理

​	两种方式

​	①利用tf.keras.preprocessing中的Tokenizer词典构建工具和tf.keras.utils.Squence构建文本数据生成器管道

​	②使用tf.data.Dataset搭配keras.layers.experimental.preprocessing.TextVectorization预处理层



②

```python
import re, string

import tensorflow as tf
from tensorflow.keras import models, layers, preprocessing, optimizers, losses, metrics
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

train_data_path = "./data/imdb/train.csv"

MAX_WORDS = 10000  # 仅考虑最⾼频的10000个词
MAX_LEN = 200  # 每个样本保留200个词的⻓度
BATCH_SIZE = 20


# 构建管道
def split_line(line):
    arr = tf.strings.split(line, "\t")
    label = tf.expand_dims(tf.cast(tf.strings.to_number(arr[0]), tf.int32), axis=0)
    text = tf.expand_dims(arr[1], axis=0)
    return (text, label)


ds_train_raw = tf.data.TextLineDataset(filenames=[train_data_path]) \
    .map(split_line, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE)


def clean_text(text):
    lowercase = tf.strings.lower(text)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    cleaned_punctuation = tf.strings.regex_replace(stripped_html,
              '[%s]' % re.escape(string.punctuation), '')
    return cleaned_punctuation


# 构建词典
vectorize_layer = TextVectorization(
    standardize=clean_text,
    split='whitespace',
    max_tokens=MAX_WORDS - 1,  # 有⼀个留给占位符
    output_mode='int',
    output_sequence_length=MAX_LEN)
ds_text = ds_train_raw.map(lambda text, label: text)
vectorize_layer.adapt(ds_text)
print(vectorize_layer.get_vocabulary()[0:100])
# 单词编码
ds_train = ds_train_raw.map(lambda text, label: (vectorize_layer(text), label)) \
    .prefetch(tf.data.experimental.AUTOTUNE)
```





#### dataset

```python
tf.data.Dataset.from_tensor_slices((X,Y)) \
 	.shuffle(buffer_size = 1000).batch(100) \
 	.prefetch(tf.data.experimental.AUTOTUNE) 


# window 使用
WINDOW_SIZE = 3
def batch_dataset(dataset):
    dataset_batched = dataset.batch(WINDOW_SIZE, drop_remainder=True)
    return dataset_batched

ds_data = tf.data.Dataset.from_tensor_slices(
    tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)).window(WINDOW_SIZE, shift=1).\
    flat_map(batch_dataset)
    
    
# zip 合并两个 dataset
tf.data.Dataset.zip((ds_data,ds_label))
```



##### 构建dataset

###### 从numpy构建

```python
# iris["data"] 和 iris["target"] 都是 numpy array
ds1 = tf.data.Dataset.from_tensor_slices((iris["data"],iris["target"]))
for features,label in ds1.take(5):
 	print(features,label)
```



###### 从 pandas df构建

```python
dfiris = pd.DataFrame(iris["data"],columns = iris.feature_names)
tf.data.Dataset.from_tensor_slices((dfiris.to_dict("list"),iris["target"]))
# df.to_dict("list")	将df转为{"column": [value1， value2]} 的格式
```



###### 从python generator构建

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 定义⼀个从⽂件中读取图⽚的generator
image_generator = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
 			"./data/cifar2/test/",
			target_size=(32, 32),
 			batch_size=20,
 			class_mode='binary')
classdict = image_generator.class_indices
print(classdict)

def generator():
 	for features,label in image_generator:
 		yield (features,label)

ds3 = tf.data.Dataset.from_generator(generator,output_types=(tf.float32,tf.int32))
```

![1594015850941](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594015850941.png)



###### 从csv构建

```python
ds4 = tf.data.experimental.make_csv_dataset(
 	file_pattern = ["./data/titanic/train.csv","./data/titanic/test.csv"],
 	batch_size=3,
 	label_name="Survived",
 	na_value="",
 	num_epochs=1,
 	ignore_errors=True)
```



###### 从文本数据构建

```python
ds5 = tf.data.TextLineDataset(
 	filenames = ["./data/titanic/train.csv","./data/titanic/test.csv"]
 	).skip(1) #略去第⼀⾏header
```



###### 从文件路径构建

```properties
参考 上一章 数据处理-图片数据处理
```



###### 从tfrecord构建

```python
# inpath：原始数据路径 outpath:TFRecord⽂件输出路径
def create_tfrecords(inpath,outpath):
 	writer = tf.io.TFRecordWriter(outpath)
 	dirs = os.listdir(inpath)
 	for index, name in enumerate(dirs):
 		class_path = inpath +"/"+ name+"/"
 		for img_name in os.listdir(class_path):
 			img_path = class_path + img_name
 			img = tf.io.read_file(img_path)
	 		#img = tf.image.decode_image(img)
 			#img = tf.image.encode_jpeg(img) #统⼀成jpeg格式压缩
 			example = tf.train.Example(
 				features=tf.train.Features(feature={
                'label':
				tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
 				'img_raw':
				tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.numpy()]))
 				}))
 			writer.write(example.SerializeToString())
 	writer.close()

create_tfrecords("./data/cifar2/test/","./data/cifar2_test.tfrecords/")


def parse_example(proto):
 	description ={ 'img_raw' : tf.io.FixedLenFeature([], tf.string),
 				   'label': tf.io.FixedLenFeature([], tf.int64)}
 	example = tf.io.parse_single_example(proto, description)
 	img = tf.image.decode_jpeg(example["img_raw"]) #注意此处为jpeg格式
 	img = tf.image.resize(img, (32,32))
 	label = example["label"]
 	return(img,label)

tf.data.TFRecordDataset("./data/cifar2_test.tfrecords").map(parse_example).shuffle(3000)

```



##### 基本使用

![1594016835426](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594016835426.png)

```python
ds = tf.data.Dataset.from_tensor_slices(["hello world","hello China","hello Beijing"])

# map
ds.map(lambda x:tf.strings.split(x," "))

# flatmap
ds.flat_map(lambda x:tf.data.Dataset.from_tensor_slices(tf.strings.split(x," ")))

#  interleave: 效果类似flat_map,但可以将不同来源的数据夹在⼀起。
ds.interleave(lambda x:tf.data.Dataset.from_tensor_slices(tf.strings.split(x," ")))

# filter
ds_filter = ds.filter(lambda x: tf.strings.regex_full_match(x, ".*[a|B].*"))

# zip
ds1 = tf.data.Dataset.range(0,3)
ds2 = tf.data.Dataset.range(3,6)
ds3 = tf.data.Dataset.range(6,9)
ds_zip = tf.data.Dataset.zip((ds1,ds2,ds3))
for x,y,z in ds_zip:
 	print(x.numpy(),y.numpy(),z.numpy())
    
# reduce
ds = tf.data.Dataset.from_tensor_slices([1,2,3,4,5.0])
result = ds.reduce(0.0,lambda x,y:tf.add(x,y))

# batch:构建批次，每次放⼀个批次。⽐原始数据增加⼀个维度。 其逆操作为unbatch。
ds = tf.data.Dataset.range(12)
ds_batch = ds.batch(4)
for x in ds_batch:
 	print(x)
    
# padded_batch:构建批次，类似batch, 但可以填充到相同的形状。
elements = [[1, 2],[3, 4, 5],[6, 7],[8]]
ds = tf.data.Dataset.from_generator(lambda: iter(elements), tf.int32)
ds_padded_batch = ds.padded_batch(2,padded_shapes = [4,])	# batch大小为2，每次输出shape为															  #（2， 4）
for x in ds_padded_batch:
 	print(x) 
    
# window:构建滑动窗⼝，返回Dataset of Dataset.
ds = tf.data.Dataset.range(12)
#window返回的是Dataset of Dataset,可以⽤flat_map压平
ds_window = ds.window(3, shift=1).flat_map(lambda x: x.batch(3,drop_remainder=True))
for x in ds_window:
	 print(x)
        
# shuffle
ds.shuffle(buffer_size = 5)

# repeat:᯿复数据若⼲次，不带参数时，᯿复⽆数次。
ds.repeat(3)

#shard:采样，从某个位置开始隔固定距离采样⼀个元素。
ds.shard(3,index = 1)	# 从index1开始，每隔三个采一个
```



##### 性能提升

![1594018146316](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594018146316.png)

```python
# prefetch
ds.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)   # AUTOTUNE让程序自动选择合适的参数

# interleave
ds_files = tf.data.Dataset.list_files("./data/titanic/*.csv")
ds = ds_files.interleave(lambda x:tf.data.TextLineDataset(x).skip(1))

# 设置num_parallel_calls
ds.map(load_image,num_parallel_calls = tf.data.experimental.AUTOTUNE)

# cache 使⽤ cache ⽅法让数据在第⼀个epoch后缓存到内存中，仅限于数据集不⼤情形。
# 即后面的epoch直接使用cache中的数据
tf.data.Dataset.from_generator(generator,output_types = (tf.int32)).cache()

# 先map后batch 改为 先batch后map
ds.map(lambda x:x**2).batch(20)
ds.batch(20).map(lambda x:x**2)
```



#### feature_column

![1594019250302](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594019250302.png)

![1594019279928](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594019279928.png)

```python
feature_columns = []
# 数值列
for col in ['age','fare','parch','sibsp'] + [c for c in dfdata.columns if 		
                                             c.endswith('_nan')]:
 	feature_columns.append(tf.feature_column.numeric_column(col))
  
# 分桶列
age = tf.feature_column.numeric_column('age')
age_buckets = tf.feature_column.bucketized_column(age,boundaries=[18, 25, 30, 35, 40, 45, 
                                                                  50, 55, 60, 65])
feature_columns.append(age_buckets)

# 类别列
# 所有的Catogorical Column类型最终都要通过indicator_column转换成Dense Column类型才能传⼊模型！！
sex = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list(
 		key='sex',vocabulary_list=["male", "female"]))
feature_columns.append(sex)

# 嵌⼊列
cabin = tf.feature_column.embedding_column(
 	tf.feature_column.categorical_column_with_hash_bucket('cabin',32),2)
feature_columns.append(cabin)

# 交叉列
pclass_cate = tf.feature_column.categorical_column_with_vocabulary_list(
 	key='pclass',vocabulary_list=[1,2,3])
crossed_feature = tf.feature_column.indicator_column(
 	tf.feature_column.crossed_column([age_buckets,pclass_cate],hash_bucket_size=15))
feature_columns.append(crossed_feature)

# 使用
model = tf.keras.Sequential([
 layers.DenseFeatures(feature_columns), #将特征列放⼊到tf.keras.layers.DenseFeatures中!!!
 layers.Dense(64, activation='relu'),
 layers.Dense(64, activation='relu'),
 layers.Dense(1, activation='sigmoid')
])

# 训练时使用包含列名的 dataset
```









#### 激活函数

```properties
tf.nn.sigmoid：将实数压缩到0到1之间，⼀般只在⼆分类的最后输出层使⽤。主要缺陷为存在梯
			   度消失问题，计算复杂度⾼，输出不以0为中⼼。
tf.nn.softmax：sigmoid的多分类扩展，⼀般只在多分类问题的最后输出层使⽤。
tf.nn.tanh：将实数压缩到-1到1之间，输出期望为0。主要缺陷为存在梯度消失问题，计算复杂度⾼。
tf.nn.relu：修正线性单元，最流⾏的激活函数。⼀般隐藏层使⽤。主要缺陷是：输出不以0为中
			⼼，输⼊⼩于0时存在梯度消失问题(死亡relu)。
tf.nn.leaky_relu：对修正线性单元的改进，解决了死亡relu问题。
tf.nn.elu：指数线性单元。对relu的改进，能够缓解死亡relu问题。
tf.nn.selu：扩展型指数线性单元。在权重⽤tf.keras.initializers.lecun_normal初始化前提下能够
对神经⽹络进⾏⾃归⼀化。不可能出现梯度爆炸或者梯度消失问题。需要和Dropout的变种AlphaDropout⼀起使⽤。
tf.nn.swish：⾃⻔控激活函数。⾕歌出品，相关研究指出⽤swish替代relu将获得轻微效果提升。
gelu：⾼斯误差线性单元激活函数。在Transformer中表现最好。tf.nn模块尚没有实现该函数。

```

使用：

```python
# 在keras模型中使⽤激活函数⼀般有两种⽅式，⼀种是作为某些层的activation参数指定，另⼀种是显式添加layers.Activation激活层。

model = models.Sequential()
model.add(layers.Dense(32,input_shape = (None,16),activation = tf.nn.relu)) #通过activation参数指定
model.add(layers.Dense(10))
model.add(layers.Activation(tf.nn.softmax)) # 显式添加layers.Activation激活层
model.summary()
```



![1594021593462](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594021593462.png)



![1594021614995](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594021614995.png)

![1594021636745](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594021636745.png)

![1594021663273](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594021663273.png)

![1594021686637](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594021686637.png)

![1594021712993](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594021712993.png)

![1594021736189](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594021736189.png)





#### 模型层layer

```properties
tf.keras.layers内置了⾮常丰富的各种功能的模型层。例如，
	layers.Dense,
	layers.Flatten,
	layers.Input,
	layers.DenseFeature,
	layers.Dropout
	layers.Conv2D,
	layers.MaxPooling2D,
	layers.Conv1D
	layers.Embedding,
	layers.GRU,
	layers.LSTM,
	layers.Bidirectional
如果这些内置模型层不能够满⾜需求，我们也可以通过编写tf.keras.Lambda匿名模型层或继承tf.keras.layers.Layer基类构建⾃定义的模型层。
其中tf.keras.Lambda匿名模型层只适⽤于构造没有学习参数的模型层。


layers常用方法：
	layers.Concatenate()([branch1,branch2,branch3])		# 将多个输出合并
```



##### 内置模型层

![1594023756041](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594023756041.png)

![1594023777255](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594023777255.png)

![1594023798534](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594023798534.png)

​	**循环网络相关层**

![1594023856791](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594023856791.png)



##### 自定义网络层

```python
# 如果⾃定义模型层没有需要被训练的参数，⼀般推荐使⽤Lamda层实现
# 如果⾃定义模型层有需要被训练的参数，则可以通过对Layer基类⼦类化实现
# Lamda层由于没有需要被训练的参数，只需要定义正向传播逻辑即可，使⽤⽐Layer基类⼦类化更加简单。

from tensorflow.keras import layers,models,regularizers
mypower = layers.Lambda(lambda x:tf.math.pow(x,2))
mypower(tf.range(5))
```

```python
# Layer的⼦类化⼀般需要᯿新实现初始化⽅法，Build⽅法和Call⽅法。下⾯是⼀个简化的线性层的范例，类似Dense.
class Linear(layers.Layer):
 	def __init__(self, units=32, **kwargs):
 		super(Linear, self).__init__(**kwargs)
 		self.units = units
 	
    #build⽅法⼀般定义Layer需要被训练的参数。
 	def build(self, input_shape):
 		self.w = self.add_weight(shape=(input_shape[-1], self.units),
 					initializer='random_normal', trainable=True)
 		self.b = self.add_weight(shape=(self.units,), initializer='random_normal',
					trainable=True)
 		super(Linear,self).build(input_shape) # 相当于设置self.built = True
 	
    #call⽅法⼀般定义正向传播运算逻辑，__call__⽅法调⽤了它。
 	def call(self, inputs):
 		return tf.matmul(inputs, self.w) + self.b

 	#如果要让⾃定义的Layer通过Functional API 组合成模型时可以序列化，需要⾃定义get_config⽅法。
 	def get_config(self):
     	config = super(Linear, self).get_config()
 		config.update({'units': self.units})
 		return config
    
linear = Linear(units = 8)
print(linear.built)	# False
#指定input_shape，显式调⽤build⽅法，第0维代表样本数量，⽤None填充
linear.build(input_shape = (None,16))
print(linear.built)	# True

model = models.Sequential()
#注意该处的input_shape会被模型加⼯，⽆需使⽤None代表样本数量维
model.add(Linear(units = 16,input_shape = (64,))) 
```





#### 损失函数loss

```python
# 监督学习的⽬标函数由损失函数和正则化项组成。（Objective = Loss + Regularization）
# 对于keras模型，⽬标函数中的正则化项⼀般在各层中指定，例如使⽤Dense的 kernel_regularizer 和
# bias_regularizer等参数指定权᯿使⽤l1或者l2正则化项，此外还可以⽤kernel_constraint 和
# bias_constraint等参数约束权᯿的取值范围，这也是⼀种正则化⼿段。

from tensorflow.keras import layers,models,losses,regularizers,constraints

model = models.Sequential()
model.add(layers.Dense(64, input_dim=64,
 		kernel_regularizer=regularizers.l2(0.01),
 		activity_regularizer=regularizers.l1(0.01),
 		kernel_constraint = constraints.MaxNorm(max_value=2, axis=0)))
model.add(layers.Dense(10,
 		kernel_regularizer=regularizers.l1_l2(0.01,0.01),activation = "sigmoid"))
model.compile(optimizer = "rmsprop",
 			  loss = "sparse_categorical_crossentropy",metrics = ["AUC"])

```



##### 内置损失函数

```properties
内置的损失函数⼀般有类的实现和函数的实现两种形式。
如：CategoricalCrossentropy 和 categorical_crossentropy 都是类别交叉熵损失函数，前者是类的实
现形式，后者是函数的实现形式。
```

![1594027158739](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594027158739.png)



##### 自定义损失函数

```python
# ⾃定义损失函数接收两个张量y_true,y_pred作为输⼊参数，并输出⼀个标量作为损失函数值。
# 也可以对tf.keras.losses.Loss进⾏⼦类化，᯿写call⽅法实现损失的计算逻辑，从⽽得到损失函数的类的实现。

# 下⾯是⼀个Focal Loss的⾃定义实现示范。Focal Loss是⼀种对binary_crossentropy的改进损失函数形式。
# 在类别不平衡和存在难以训练样本的情形下相对于⼆元交叉熵能够取得更好的效果

def focal_loss(gamma=2., alpha=.25):
 	def focal_loss_fixed(y_true, y_pred):
 		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
 		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
 		loss = -tf.sum(alpha * tf.pow(1. - pt_1, gamma) * tf.log(1e-07+pt_1))\
		 	-tf.sum((1-alpha) * tf.pow( pt_0, gamma) * tf.log(1. - pt_0 + 1e07))
 		return loss
 	return focal_loss_fixed


class FocalLoss(losses.Loss):
 	def __init__(self,gamma=2.0,alpha=0.25):
 		self.gamma = gamma
 		self.alpha = alpha
 	def call(self,y_true,y_pred):
 		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
 		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
 		loss = -tf.sum(self.alpha * tf.pow(1. - pt_1, self.gamma) * tf.log(1e07+pt_1)) \
 		-tf.sum((1-self.alpha) * tf.pow( pt_0, self.gamma) * tf.log(1. -pt_0 + 1e-07))
 		return loss
```



#### 评估指标 matrix

```properties
通常损失函数都可以作为评估指标，如MAE,MSE,CategoricalCrossentropy等也是常⽤的评估指标。
但评估指标不⼀定可以作为损失函数，例如AUC,Accuracy,Precision。因为评估指标不要求连续可导，⽽损失函数通常要求连续可导
```

![1594027654736](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594027654736.png)

![1594027699447](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594027699447.png)



##### 自定义评估指标

```python
# ⾃定义评估指标需要接收两个张量y_true,y_pred作为输⼊参数，并输出⼀个标量作为评估值。
# 也可以对tf.keras.metrics.Metric进⾏⼦类化，᯿写初始化⽅法, update_state⽅法, result⽅法实现评估
# 指标的计算逻辑，从⽽得到评估指标的类的实现形式。

# 我们以⾦融⻛控领域常⽤的KS指标为例，示范⾃定义评估指标。
# KS指标适合⼆分类问题，其计算⽅式为 KS=max(TPR-FPR).

#函数形式的⾃定义评估指标
@tf.function
def ks(y_true,y_pred):
 	y_true = tf.reshape(y_true,(-1,))
 	y_pred = tf.reshape(y_pred,(-1,))
 	length = tf.shape(y_true)[0]
 	t = tf.math.top_k(y_pred,k = length,sorted = False)
 	y_pred_sorted = tf.gather(y_pred,t.indices)
 	y_true_sorted = tf.gather(y_true,t.indices)
 	cum_positive_ratio = tf.truediv(
 	tf.cumsum(y_true_sorted),tf.reduce_sum(y_true_sorted))
 	cum_negative_ratio = tf.truediv(
 	tf.cumsum(1 - y_true_sorted),tf.reduce_sum(1 - y_true_sorted))
 	ks_value = tf.reduce_max(tf.abs(cum_positive_ratio - cum_negative_ratio))
 	return ks_value

ks(y_true,y_pred)




#类形式的⾃定义评估指标
class KS(metrics.Metric):

 	def __init__(self, name = "ks", **kwargs):
 		super(KS,self).__init__(name=name,**kwargs)
 		self.true_positives = self.add_weight(
 			name = "tp",shape = (101,), initializer = "zeros")
 		self.false_positives = self.add_weight(
 			name = "fp",shape = (101,), initializer = "zeros")

 	@tf.function
 	def update_state(self,y_true,y_pred):
 		y_true = tf.cast(tf.reshape(y_true,(-1,)),tf.bool)
 		y_pred = tf.cast(100*tf.reshape(y_pred,(-1,)),tf.int32)

 		for i in tf.range(0,tf.shape(y_true)[0]):
 			if y_true[i]:
 				self.true_positives[y_pred[i]].assign(
 							self.true_positives[y_pred[i]]+1.0)
 			else:
 				self.false_positives[y_pred[i]].assign(
 							self.false_positives[y_pred[i]]+1.0)
 		return (self.true_positives,self.false_positives)

	@tf.function
 	def result(self):
 		cum_positive_ratio = tf.truediv(
 		tf.cumsum(self.true_positives),tf.reduce_sum(self.true_positives))
 		cum_negative_ratio = tf.truediv(
				tf.cumsum(self.false_positives),tf.reduce_sum(self.false_positives))
 		ks_value = tf.reduce_max(tf.abs(cum_positive_ratio - cum_negative_ratio))
		return ks_value

# 使用
myks = KS()
myks.update_state(y_true,y_pred)
myks.result()
```

![1594027634357](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594027634357.png)



#### 优化器optimizers

发展历程：SGD -> SGDM -> NAG ->Adagrad -> Adadelta(RMSprop) -> Adam -> Nadam 



##### 内置优化器

在keras.optimizers⼦模块中，它们基本上都有对应的类的实现。

![1594028765248](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594028765248.png)



##### 优化器的使用

```python
# 优化器主要使⽤apply_gradients⽅法传⼊变量和对应梯度从⽽来对给定变量进⾏迭代，或者直接使⽤minimize⽅法对⽬标函数进⾏迭代优化。
# 更常⻅的使⽤是在编译时将优化器传⼊keras的Model,通过调⽤model.fit实现对Loss的的迭代优化。

# 初始化优化器时会创建⼀个变量optimier.iterations⽤于记录迭代的次数。因此优化器和tf.Variable⼀样，⼀般需要在@tf.function外创建。

# 求f(x) = a*x**2 + b*x + c的最⼩值
# 使⽤optimizer.apply_gradients
x = tf.Variable(0.0,name = "x",dtype = tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
@tf.function
def minimizef():
 	a = tf.constant(1.0)
 	b = tf.constant(-2.0)
 	c = tf.constant(1.0)

 	while tf.constant(True):
 		with tf.GradientTape() as tape:
             y = a*tf.pow(x,2) + b*x + c
 		dy_dx = tape.gradient(y,x)
 		optimizer.apply_gradients(grads_and_vars=[(dy_dx,x)])

 		#迭代终⽌条件
 		if tf.abs(dy_dx)<tf.constant(0.00001):
 			break

 		if tf.math.mod(optimizer.iterations,100)==0:	# 每迭代100次
 			tf.print("step = ",optimizer.iterations)
 			tf.print("x = ", x)
 			tf.print("")

 	y = a*tf.pow(x,2) + b*x + c
 	return y

tf.print("y =",minimizef())
tf.print("x =",x)




# 使⽤optimizer.minimize
x = tf.Variable(0.0,name = "x",dtype = tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
def f():
 	a = tf.constant(1.0)
 	b = tf.constant(-2.0)
 	c = tf.constant(1.0)
 	y = a*tf.pow(x,2)+b*x+c
 	return(y)
@tf.function
def train(epoch = 1000):
 	for _ in tf.range(epoch):
 		optimizer.minimize(f,[x])
 	tf.print("epoch = ",optimizer.iterations)
 	return(f())
train(1000)
tf.print("y = ",f())
tf.print("x = ",x)




# 使⽤model.fit
class FakeModel(tf.keras.models.Model):
 	def __init__(self,a,b,c):
 		super(FakeModel,self).__init__()
 		self.a = a
 		self.b = b
 		self.c = c

 	def build(self):
 		self.x = tf.Variable(0.0,name = "x")
 		self.built = True

 	def call(self,features):
 		loss = self.a*(self.x)**2+self.b*(self.x)+self.c
 		return(tf.ones_like(features)*loss)

def myloss(y_true,y_pred):
 	return tf.reduce_mean(y_pred)

model = FakeModel(tf.constant(1.0),tf.constant(-2.0),tf.constant(1.0))
model.build()
model.summary()
model.compile(optimizer =tf.keras.optimizers.SGD(learning_rate=0.01),loss = myloss)
history = model.fit(tf.zeros((100,2)),tf.ones(100),batch_size = 1,epochs = 10) #迭代1000次
tf.print("x=",model.x)
tf.print("loss=",model(tf.constant(0.0)))

```





#### 回调函数callback

##### 内置回调函数

![1594087867239](C:\Users\xiaoxuan\AppData\Roaming\Typora\typora-user-images\1594087867239.png)



##### 自定义回调函数

```python
# 继承⾄ keras.callbacks.Callbacks基类，拥有params和model这两个属性
# 可以使⽤callbacks.LambdaCallback编写较为简单的回调函数，也可以通过对callbacks.Callback⼦类化
# 编写更加复杂的回调函数逻辑。

from tensorflow.keras import layers,models,losses,metrics,callbacks

# 示范使⽤LambdaCallback编写较为简单的回调函数
import json
json_log = open('./data/keras_log.json', mode='wt', buffering=1)
json_logging_callback = callbacks.LambdaCallback(
 	on_epoch_end=
    lambda epoch, logs: json_log.write(json.dumps(dict(epoch = epoch,**logs)) + '\n'),
 	on_train_end=lambda logs: json_log.close()
)


# 示范通过Callback⼦类化编写回调函数（LearningRateScheduler的源代码）
class LearningRateScheduler(callbacks.Callback):

 	def __init__(self, schedule, verbose=0):
 		super(LearningRateScheduler, self).__init__()
 		self.schedule = schedule
 		self.verbose = verbose
 	def on_epoch_begin(self, epoch, logs=None):
 		if not hasattr(self.model.optimizer, 'lr'):
 			raise ValueError('Optimizer must have a "lr" attribute.')
 		try:
 			lr = float(K.get_value(self.model.optimizer.lr))
 			lr = self.schedule(epoch, lr)
 		except TypeError: # Support for old API for backward compatibility
 			lr = self.schedule(epoch)
 		if not isinstance(lr, (tf.Tensor, float, np.float32, np.float64)):
 			raise ValueError('The output of the "schedule" function should be float.')
	 	if isinstance(lr, ops.Tensor) and not lr.dtype.is_floating:
 			raise ValueError('The dtype of Tensor should be float')
 		K.set_value(self.model.optimizer.lr, K.get_value(lr))
 		if self.verbose > 0:
 			print('\nEpoch %05d: LearningRateScheduler reducing learning rate to %s.' % 
                  (epoch + 1, lr))
 	def on_epoch_end(self, epoch, logs=None):
 		logs = logs or {}
 		logs['lr'] = K.get_value(self.model.optimizer.lr)
```





#### gpu设置

```python
gpus = tf.config.list_physical_devices("GPU")

if gpus:
 	gpu0 = gpus[0] #如果有多个GPU，仅使⽤第0个GPU
 	tf.config.experimental.set_memory_growth(gpu0, True) #设置GPU显存⽤量按需使⽤
 	# 或者也可以设置GPU显存为固定使⽤量(例如：4G)
 	#tf.config.experimental.set_virtual_device_configuration(gpu0,
 	#[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
 	tf.config.set_visible_devices([gpu0],"GPU")
    
    
    
#此处在colab上使⽤1个GPU模拟出两个逻辑GPU进⾏多GPU训练
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
 	# 设置两个逻辑GPU模拟多GPU训练
 	try:
 		tf.config.experimental.set_virtual_device_configuration(gpus[0],
                             [tf.config.experimental.
                               VirtualDeviceConfiguration(memory_limit=1024),
							tf.config.experimental.
                              VirtualDeviceConfiguration(memory_limit=1024)])
 		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
 		print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
 	except RuntimeError as e:
 		print(e)
```



#### 模型部署

略

参考 eat_tf2_ebook.pdf

使用spark进行分布式推理