from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Embedding, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Input, Embedding, Dense, Reshape, Concatenate, Conv2D, Conv2DTranspose, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
from PIL import Image
import numpy as np

# 配置参数
IMAGE_SIZE = 64
NUM_CHANNELS = 3
LATENT_DIM = 100
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 5000

# 存储所有的图像和标签
images = []
labels = []

# 读取所有的PNG格式图片
folder_path = 'diamond-pearl'
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        # 打开图片并转换为灰度图像
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert('L')

        # 将图像转换为NumPy数组并添加到列表中
        img_array = np.array(img)
        images.append(img_array)

        # 将标签添加到列表中
        label = filename.split('_')[0]
        labels.append(label)

# 将列表转换为NumPy数组
images = np.array(images)
labels = np.array(labels)

# 归一化图像数据
images = (images - 127.5) / 127.5

# 定义生成器模型
def build_generator():
    z = Input(shape=(LATENT_DIM,))
    c = Input(shape=(1,), dtype='int32')
    c_emb = Embedding(NUM_CLASSES, LATENT_DIM)(c)
    c_emb = Flatten()(c_emb)  # 将形状从(None, 1, 100)转换为(None, 100)
    z_emb = Concatenate()([z, c_emb])
    x = Dense(4 * 4 * 512, use_bias=False)(z_emb)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Reshape((4, 4, 512))(x)
    x = Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(NUM_CHANNELS, 4, strides=2, padding='same', use_bias=False, activation='tanh')(x)
    label_input = Input(shape=(1,), dtype='int32')
    label_emb = Embedding(NUM_CLASSES, IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS)(label_input)
    label_emb = Reshape((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))(label_emb)
    x = Concatenate()([x, label_emb])
    generator = Model([z, c, label_input], x)
    return generator

# 定义判别器模型
def build_discriminator():
    x = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS + 1))
    x = Conv2D(64, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(512, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model(x, name='discriminator')
    return discriminator


# 定义分类数目、嵌入层的维度和输入图像的形状
num_classes = 10
latent_dim = 100
img_shape = (64, 64, 3)

def build_generator():
    # 定义输入层
    input_noise = Input(shape=(latent_dim,))
    input_class = Input(shape=(1,))

    # 定义嵌入层
    embedding = Embedding(num_classes, latent_dim)(input_class)
    embedding = Reshape((latent_dim,))(embedding)

    # 将噪声和嵌入向量连接起来
    concatenated = Concatenate()([input_noise, embedding])

    # 添加三个转置卷积层和BatchNormalization层
    x = Dense(128 * 8 * 8, activation='relu')(concatenated)
    x = Reshape((8, 8, 128))(x)
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)
    x = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')(x)

    # 定义模型
    generator = Model(inputs=[input_noise, input_class], outputs=x, name='generator')

    return generator

def build_discriminator():
    # 定义输入层
    input_image = Input(shape=img_shape)

    # 添加三个卷积层和BatchNormalization层
    x = Conv2D(32, kernel_size=4, strides=2, padding='same', activation='relu')(input_image)
    x = Conv2D(64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = Flatten()(x)

    # 添加一个全连接层和输出层
    x = Dense(1, activation='sigmoid')(x)

    # 定义模型
    discriminator = Model(inputs=input_image, outputs=x, name='discriminator')

    return discriminator

def build_cgan(generator, discriminator):
    # 定义输入层和嵌入层
    input_text = Input(shape=(1,))
    embedding = Embedding(num_classes, latent_dim)(input_text)
    embedding = Reshape((latent_dim,))(embedding)

    # 定义生成器和鉴别器的输入层
    input_noise = Input(shape=(latent_dim,))
    input_class = Input(shape=(1,))
    input_image = Input(shape=img_shape)

    # 使用生成器生成图像
    generated_image = generator([input_noise, input_class])

    # 上采样生成器生成的图像
    upsampled_image = Conv2DTranspose(3, kernel_size=4, strides=4, padding='same')(generated_image)

    # 将上采样后的图像和输入的真实图像连接起来
    concatenated = Concatenate(axis=3)([upsampled_image, input_image])

    # 使用鉴别器判断图像的真假
    validity = discriminator(concatenated)

    # 定义模型
    cgan = Model(inputs=[input_noise, input_class, input_image, input_text], outputs=validity, name='cgan')

    return cgan



# 定义优化器和损失函数
optimizer = Adam(0.0002, 0.5)
loss_fn = 'binary_crossentropy'

# 构建生成器、判别器和CGAN模型
generator = build_generator()
discriminator = build_discriminator()
cgan = build_cgan(generator, discriminator)

# 编译判别器模型
discriminator.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])

# 固定判别器模型的参数，
discriminator.trainable = False

#编译CGAN模型
cgan.compile(loss=loss_fn, optimizer=optimizer)

#训练模型# 随机选择一批真实图像
for epoch in range(EPOCHS):
    idx = np.random.randint(0, images.shape[0], BATCH_SIZE)
real_images = images[idx]
labels = labels[idx]

# 生成一批噪声
noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))

# 在噪声中加入标签信息
sampled_labels = np.random.randint(0, NUM_CLASSES, BATCH_SIZE).reshape(-1, 1)

# 生成一批假的图像
fake_images = generator.predict([noise, sampled_labels, sampled_labels])

# 组合真实图像和假的图像，用于训练判别器模型
combined_images = np.concatenate([real_images, fake_images])

# 标注真实图像和假的图像的标签信息
labels = np.concatenate([labels, sampled_labels])

# 创建标签
valid_labels = np.ones((BATCH_SIZE, 1))
fake_labels = np.zeros((BATCH_SIZE, 1))

# 训练判别器模型
d_loss_real = discriminator.train_on_batch(real_images, valid_labels)
d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

# 重新生成噪声
noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))

# 随机选择一批标签信息
sampled_labels = np.random.randint(0, NUM_CLASSES, BATCH_SIZE).reshape(-1, 1)

# 标注生成的假图像的标签信息
valid_labels = np.ones((BATCH_SIZE, 1))

# 训练生成器模型和CGAN模型
g_loss = cgan.train_on_batch([noise, sampled_labels, sampled_labels], valid_labels)

# 打印损失函数的值
print(f'Epoch {epoch+1}/{EPOCHS} - D loss: {d_loss[0]:.4f} - D acc: {d_loss[1]*100:.2f}% - G loss: {g_loss:.4f}')

# 每100个epoch保存一次生成器模型的权重
if (epoch+1) % 100 == 0:
    generator.save_weights(f'generator_weights_epoch_{epoch+1}.h5')


def generate_pokemon(generator, text_input, color):
    # 将输入的文本转换为标签信息
    label = np.zeros((1, 1))
    if color.lower() == 'red':
        label[0, 0] = 0
    elif color.lower() == 'blue':
        label[0, 0] = 1
    elif color.lower() == 'yellow':
        label[0, 0] = 2
    elif color.lower() == 'green':
        label[0, 0] = 3
    elif color.lower() == 'black':
        label[0, 0] = 4
    elif color.lower() == 'white':
        label[0, 0] = 5
    elif color.lower() == 'purple':
        label[0, 0] = 6
    elif color.lower() == 'pink':
        label[0, 0] = 7
    elif color.lower() == 'brown':
        label[0, 0] = 8
    else:
        label[0, 0] = 9

    # 生成噪声
    noise = np.random.normal(0, 1, (1, LATENT_DIM))

    # 生成口袋妖怪的图像
    generated_image = generator.predict([noise, label, label])[0]

    # 将图像从[-1,1]的范围转换为[0,255]的范围
    generated_image = (generated_image + 1) * 127.5

    # 将图像转换为PIL图像
    generated_image = Image.fromarray(generated_image.astype('uint8'))

    # 保存图像
    save_dir = './generated_images'  # 保存路径
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'{text_input}_{color.lower()}.png')
    with open(save_path, 'wb') as f:
        generated_image.save(f, 'png')
    # 显示图像
    generated_image.show()
