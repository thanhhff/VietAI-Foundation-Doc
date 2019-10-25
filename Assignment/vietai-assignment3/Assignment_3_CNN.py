#!/usr/bin/env python
# coding: utf-8

# # Giới thiệu Convolution Nets
# 
# Convolutional Neural Networks (CNN) là một trong những mô hình deep learning phổ biến nhất và có ảnh hưởng nhiều nhất trong cộng đồng Computer Vision. CNN được dùng trong nhiều bài toán như nhận dạng ảnh, phân tích video, ảnh MRI, hoặc cho các bài của lĩnh vực xử lý ngôn ngữ tự nhiên, và hầu hết đều giải quyết tốt các bài toán này. 
# 
# CNN cũng có lịch sử khá lâu đời. Kiến trúc gốc của mô hình CNN được giới thiệu bởi một nhà khoa học máy tính người Nhật vào năm 1980. Sau đó, năm 1998, Yan LeCun lần đầu huấn luyện mô hình CNN với thuật toán backpropagation cho bài toán nhận dạng chữ viết tay. Tuy nhiên, mãi đến năm 2012, khi một nhà khoa học máy tính người Ukraine Alex Krizhevsky (đệ của Geoffrey Hinton) xây dựng mô hình CNN (AlexNet) và sử dụng GPU để tăng tốc quá trình huấn luyện deep nets để đạt được top 1 trong cuộc thi Computer Vision thường niên ImageNet với độ lỗi phân lớp top 5 giảm hơn 10% so với những mô hình truyền thống trước đó, đã tạo nên làn sóng mãnh mẽ sử dụng deep CNN với sự hỗ trợ của GPU để giải quyết càng nhiều các vấn đề trong Computer Vision.
# 
# # Bài Toán Phân loại Ảnh
# Phân loại ảnh là một bài toán quan trọng bậc nhất trong lĩnh vực Computer Vision. Chúng ta đã có rất nhiều nghiên cứu để giải quyết bài toán này bằng cách rút trích các đặc trưng rất phổ biến như SIFT, HOG rồi cho máy tính học nhưng những cách này tỏ ra không thực sự hiểu quả. Nhưng ngược lại, đối với con người, chúng ta lại có bản năng tuyệt vời để phân loại được những đối tượng trong khung cảnh xung quanh một cách dễ dàng.
# 
# Dữ liệu đầu vào của bài toán là một bức ảnh. Một ảnh được biểu diễn bằng ma trận các giá trị. Mô hình phân lớp sẽ phải dự đoán được lớp của ảnh từ ma trận điểm ảnh này, ví dụ như ảnh đó là con mèo, chó, hay là chim.
# 
# ![](https://pbcquoc.github.io/images/cnn_input.png)
# 
# # Nội dung 
# Trong assignment này, mình sẽ hướng dẫn các bạn xây dựng mô hình CNN (Convolution Neural Nets) cho bài toán phân loại ảnh. Các bạn sẽ sử dụng tensorflow [eager execution](https://www.tensorflow.org/guide/eager) để xây dựng model, huấn luyện mô hình trên tập train và predict ảnh trong tập test. 
# 
# Assignment này sẽ có câú trúc như sau:
# 1. Import/ Xử lý dữ liệu
# 2. Xây dựng mô hình
# 3. Huấn luyện mô hình
# 4. Đánh giá mô hình
# 5. Sử dụng mô hình đã huấn luyện để dự đoán và nộp kết quả lên Kaggle
# 
# **Lưu ý: điểm xếp hạng trên Kaggle chỉ chiếm 30% số điểm, 70% số điểm còn lại sẽ chấm ở file notebook này**.

# # Import thư viện
# 
# Chúng ta sử dụng một số hàm cơ bản trong tensorflow, sklearn và phải gọi hàm tf.enable_eager_execution. 

# In[1]:


import os
import numpy as np

np.warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from tqdm import tqdm

from google_drive_downloader import GoogleDriveDownloader as gdd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.contrib.eager.python import tfe
from PIL import Image

tf.enable_eager_execution()
tf.set_random_seed(0)
np.random.seed(0)

# # Import và inspect dữ liệu
# Trong bài này, các bạn phải xây dựng mô hình để xác định các địa danh nổi tiếng trên lãnh thổ Việt Nam được mô tả trong bức ảnh. Tập dữ liệu huấn luyện bao gồm 10 ngàn ảnh, là một phần nhỏ của bộ dữ liệu trong cuộc thi ZaloAI năm 2018. 
# 

# ## Download dữ liệu
# Bạn có thể sử dụng trực tiếp dữ liệu trên competition được host trên Kaggle: [VietAI Foundation Course - CNN Assignment](https://www.kaggle.com/c/vietai-fc-cnn-assignment/data)
# 
# Hoặc tải dữ liệu xuống từ Google Drive

# In[2]:


# gdd.download_file_from_google_drive(file_id='1ycR7Aexe8xbZ8oEDsQwGc9SIiFklRpfu', dest_path='./data.zip', unzip=True)


# Dữ liệu tải xuống sẽ chứa trong folder `data`. Cấu trúc thư mục như sau:

# In[3]:


data_dir = 'data'
os.listdir(data_dir)

# Trong đó:
# - **images**: thư mục chứa tất cả các ảnh dùng cho việc huấn luyện và đánh giá
# - **train.csv**: file CSV chứa tên các file và nhãn dùng cho việc huấn luyện
# - **sample_submission.csv**: file CSV mẫu chứa tên các file cần đánh giá và nhãn dummy.

# ## Đọc và xử lý dữ liệu

# Đọc dữ liệu từ file CSV:

# In[4]:


train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
train_df.head()

# In[5]:


train_df.info()

# In[6]:


test_df = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
test_df.head()

# In[7]:


test_df.info()

# Tổng cộng có 8234 ảnh cho việc huấn luyện và 2059 ảnh cần dự đoán nhãn, ta tiến hành thống kê phân bố các nhãn trên tập huấn luyện:

# In[8]:


train_df.label.value_counts()


# Số lượng các ảnh cho mỗi lớp từ 400 đến 2000. Trong đó lớp số 2 có số lượng ảnh nhiều nhất.

# ## TODO 1: Cài đặt hàm đọc ảnh và đưa về NumPy Array
# Để máy tính hiểu được các ảnh, chúng ta cần đọc và chuyển các ảnh về tensor. Bên cạnh đó, các tensor biểu diễn cần có kích thước cố định nên trong quá trình đọc ảnh, ta cần thay đổi về kích thước mong muốn (resize ảnh). Trong các bài toán về deep learning, ta thường biểu diễn ảnh dưới dạng tensor có kích thước `(224,224,3)` với 3 kênh màu, 224 pixels cho mỗi kênh.
# 
# Hoàn thành hàm `generate_data` bên dưới nhận vào 1 list N đường dẫn đến ảnh và kích thước `size` ảnh cần resize. Trả về numpy array có kích thước `(N,size,size,3)` với các giá trị được normalized trong khoảng \[0 ; 1\].

# In[9]:


def generate_data(image_paths, size=224):
    """
    Đọc và chuyển các ảnh về numpy array
    
    Parameters
    ----------
    image_paths: list of N strings
        List các đường dẫn ảnh
    size: int
        Kích thước ảnh cần resize
    
    Returns
    -------
    numpy array kích thước (N, size, size, 3)
    """
    image_array = np.zeros((len(image_paths), size, size, 3), dtype='float32')

    for idx, image_path in tqdm(enumerate(image_paths)):
        ### START CODE HERE
        # Đọc ảnh bằng thư viện Pillow và resize ảnh
        image = Image.open(image_path)
        image = image.resize((size, size), Image.ANTIALIAS)

        # Chuyển ảnh thành numpy array và gán lại mảng image_array
        pixels = np.asarray(image)

        # convert from integers to floats
        pixels = pixels.astype('float32')
        pixels /= 255
        image_array[idx] = pixels

        ### END CODE HERE

    return image_array


# Sử dụng hàm `generate_data` để tạo ma trận của tập dữ liệu train và test:

# In[10]:


# List các đường dẫn file cho việc huấn luyện
train_files = [os.path.join("data/images", file) for file in train_df.image]

# List các nhãn
train_y = train_df.label

# Tạo numpy array cho dữ liệu huấn luyện
train_arr = generate_data(train_files)

# Hãy kiểm tra kích thước của tensor `train_arr` vừa tạo ra. Kích thước đúng sẽ là `(8234,224,224,3)`.

# In[11]:


train_arr.shape

# Tiến hành tạo tensor dữ liệu cho tập test:

# In[12]:


test_files = [os.path.join("data/images", file) for file in test_df.image]
test_x = generate_data(test_files)
test_x.shape

# Tạo **one-hot labels** từ `train_y` để đưa vào huấn luyện với Tensorflow.

# In[13]:


num_classes = len(np.unique(train_y))
y_ohe = tf.keras.utils.to_categorical(train_y, num_classes=num_classes)

# ## Chia dữ liệu để huấn luyện và đánh giá
# 
# Ta sẽ không sử dụng 100% tập dữ liệu đã có nhãn để huấn luyện mà sẽ chỉ huấn luyện trên 75% bộ dữ liệu và sử dụng 25% còn lại dùng để đánh giá model qua các epoch.
# 
# Chúng ta sử dụng hàm `train_test_split` trong thư viện sklearn để chia tập dữ liệu thành 2 phần train/validation một cách nhanh chóng.

# In[14]:


x_train, x_valid, y_train_ohe, y_valid_ohe = train_test_split(train_arr, y_ohe, test_size=0.25)

print("Train size: {} - Validation size: {}".format(x_train.shape, x_valid.shape))


# ## Mô Hình CNN
# 
# CNN bao gồm tập hợp các lớp cơ bản sau: convolutional layer + nonlinear layer (RELU, ...), pooling layer, fully connected layer. Các lớp này liên kết với nhau theo một thứ tự nhất định. Thông thường, một ảnh sẽ được lan truyền qua tầng convolutional layer + nonlinear layer đầu tiên, sau đó các giá trị tính toán được sẽ lan truyền qua pooling layer, bộ ba convolutional layer + nonlinear layer + pooling layer có thể được lặp lại nhiều lần trong network. Và sau đó được lan truyền qua tầng fully connected layer và softmax để tính xác suất ảnh đó thuộc lớp nào.
# 
# ![](https://pbcquoc.github.io/images/cnn_model.png)
# 
# ### Convolutional Layer
# Convolutional layer thường là lớp đầu tiên và cũng là lớp quan trọng nhất của mô hình CNN. Lớp này có chức năng chính là phát hiện các đặc trưng về không gian một cách hiệu quả. Trong tầng này có 4 đối tượng chính là: ma trận đầu vào, bộ **filter**, và **receptive field**, **feature map**. Conv layer nhận đầu vào là một ma trận 3 chiều và một bộ filter cần phải học. Bộ filters này sẽ trượt qua từng vị trí trên bức ảnh để tính tích chập (convolution) giữa bộ filter và phần tương ứng trên bức ảnh. Phần tương ứng này trên bức ảnh gọi là receptive field, tức là vùng mà một neuron có thể nhìn thấy để đưa ra quyết định, và ma trận sinh ra bởi quá trình này được gọi là feature map. Để hình dung, các bạn có thể tưởng tượng, bộ filters giống như các tháp canh trong nhà tù quét lần lượt qua không gian xung quanh để tìm kiếm tên tù nhân bỏ trốn. Khi phát hiện tên tù nhân bỏ trốn, thì chuông báo động sẽ reo lên, giống như các bộ filters tìm kiếm được đặc trưng nhất định thì tích chập đó sẽ cho giá trị lớn. 
# 
# <div class="img-div" markdown="0">
#     <img src="https://media.giphy.com/media/3orif7it9f4phjv4LS/giphy.gif" />
# </div>
# 
# Với ví dụ ở bên dưới, dữ liệu đầu vào là ma trận có kích thước 8x8x1, một bộ filter có kích thước 2x2x1, feature map có kích thước 7x7x1. Mỗi giá trị ở feature map được tính bằng tổng của tích các phần tử tương ứng của bộ filter 2x2x1 với receptive field trên ảnh. Và để tính tất cả các giá trị cho feature map, các bạn cần trượt filter từ trái sang phải, từ trên xuống dưới. Do đó, các bạn có thể thấy rằng phép convolution bảo toàn thứ tự không gian của các điểm ảnh. Ví dụ điểm góc trái của dữ liệu đầu vào sẽ tương ứng với bên một điểm bên góc trái của feature map. 
# 
# <div class="img-div" markdown="0">
#     <img src="https://pbcquoc.github.io/images/cnn_covolution_layer.png" />
# </div>
# 
# #### Tầng convolution như là feature detector 
# 
# Tầng convolution có chức năng chính là phát hiện đặc trưng cụ thể của bức ảnh. Những đặc trưng này bao gồm đặc trưng cơ bản là góc, cạnh, màu sắc, hoặc đặc trưng phức tạp hơn như texture của ảnh. Vì bộ filter quét qua toàn bộ bức ảnh, nên những đặc trưng này có thể nằm ở vị trí bất kì trong bức ảnh, cho dù ảnh bị xoay trái/phải thì những đặc trưng này vẫn được phát hiện. 
# 
# Ở minh họa dưới, các bạn có một filter 5x5 dùng để phát hiện góc/cạnh, filter này chỉ có giá trị một tại các điểm tương ứng một góc cong. 
# 
# <div class="img-div" markdown="0">
#     <img src="https://pbcquoc.github.io/images/cnn_high_level_feature.png" />
# </div>
# 
# Dùng filter ở trên trượt qua ảnh của nhân vật Olaf trong trong bộ phim Frozen. Chúng ta thấy rằng, chỉ ở những vị trí trên bức ảnh có dạng góc như đặc trưng ở filter thì mới có giá trị lớn trên feature map, những vị trí còn lại sẽ cho giá trị thấp hơn. Điều này có nghĩa là, filter đã phát hiện thành công một dạng góc/cạnh trên dữ liệu đầu vào. Tập hợp nhiều bộ filters sẽ cho phép các bạn phát hiện được nhiều loại đặc trưng khác nhau, và giúp định danh được đối tượng. 
# 
# <div class="img-div" markdown="0">
#     <img src="https://pbcquoc.github.io/images/cnn_high_level_feature_ex.png" />
# </div>
# 
# #### Các tham số của tầng convolution: Kích thước bộ filters, stride và padding
# 
# Kích thước bộ filters là một trong những siêu tham số quan trọng nhất của tầng convolution. Kích thước này tỉ lệ thuận với số lượng tham số cần học tại mỗi tầng convolution và là tham số quyết định receptive field của tầng này. Kích thước phổ biến nhất của bộ filter là 3x3.
# 

# # Xây dựng mô hình
# Các bạn cần phải xây dựng mô hình CNN có kiến trúc sau đây. Bộ filter có kích thước 3x3. Đối với các tham số còn lại, các bạn có thể tự do lựa chọn để cho ra kết quả huấn luyện tốt nhất.
# 
# ![](https://github.com/pbcquoc/cnn/raw/master/images/cnn_architecture_2.png)
# 

# ## Định nghĩa block CNN
# Để hỗ trợ quá trình định nghĩa mô hình. Các bạn cần định nghĩa một block bao gồm 3 lớp sau: Conv2D, MaxPool2D, ReLU. Block này sẽ được tái sử dụng nhiều lần trong networks. Các layers cần được khai báo trong hàm init và được gọi trong hàm call. Hãy tham khảo ví dụ dưới đây.
# 
# ```python
# 
# class ConvBlock(tf.keras.Model):
#     def __init__(self):
#         super(ConvBlock, self).__init__()
#         self.cnn = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1),  padding="same")
#         
#     def call(self, inputs, training=None, mask=None):
#         x = self.cnn(inputs)
# 
#         return x
# ```
# 
# Các tài liệu tham khảo:
# - [tf.keras.layers.Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)
# - [tf.keras.layers.MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D)

# In[15]:


class ConvBlock(tf.keras.Model):
    def __init__(self, filters, kernel, strides, padding):
        '''
        Khởi tạo Convolution Block với các tham số đầu vào
        
        Parameters
        ----------
        filters: int
            số lượng filter
        kernel: int
            kích thước kernel
        strides: int
            stride của convolution layer
        padding: str
            Loại padding của convolution layer
        
        '''

        super(ConvBlock, self).__init__()
        ## TODO 2
        ### START CODE HERE

        # Tạo layer Conv2D
        self.cnn = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding=padding)

        # Tạo layer MaxPool2D
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        # Tạo các layer khác tùy ý nếu cần thiết

        ### END CODE HERE

    def call(self, inputs):
        '''
        Hàm này sẽ được gọi trong quá trình forwarding của mạng
        
        Parameters
        ----------
        inputs: tensor đầu vào
        
        Returns
        -------
        tensor
            giá trị đầu ra của mạng
        '''

        x = self.cnn(inputs)
        ## TODO 3
        ### START CODE HERE

        # Forward inputs qua từng layer và gán vào biến x để trả về

        x = tf.keras.activations.relu(x)

        ## END CODE HERE

        return x


# ## Định nghĩa toàn bộ mô hình CNN
# Các bạn sử dụng block ở trên để định nghĩa toàn bộ mô hình CNN có kiến trúc như hình dưới. Các layer cần được khởi tạo trong hàm init, và được gọi trong hàm call.

# In[ ]:


# In[ ]:


class CNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        ## TODO 4
        ### START CODE HERE

        # Khởi tạo các convolution block
        self.block1 = ConvBlock(filters=32, kernel=(3, 3), strides=(1, 1), padding="same")
        self.block2 = ConvBlock(filters=32, kernel=(3, 3), strides=(1, 1), padding="same")
        self.block3 = ConvBlock(filters=32, kernel=(3, 3), strides=(1, 1), padding="same")
        self.block4 = ConvBlock(filters=64, kernel=(3, 3), strides=(1, 1), padding="same")
        self.block5 = ConvBlock(filters=64, kernel=(3, 3), strides=(1, 1), padding="same")

        # Khởi tạo layer để flatten feature map 
        self.flatten = tf.keras.layers.Flatten()

        ### END CODE HERE

        ## TODO 5
        ### START CODE HERE

        # Khởi tạo fully connected layer
        self.dense1 = tf.keras.layers.Dense(num_classes)

        ### END CODE HERE

    def call(self, inputs):
        ## TODO 6
        x = inputs
        ### START CODE HERE

        # Forward gía trị inputs qua các tầng CNN và gán vào x
        x = ConvBlock.call(x)

        ### END CODE HERE

        ## TODO 7

        ### START CODE HERE 

        # Forward giá trị x qua Fully connected layer
        self.flatten(x)
        x = self.dense1

        ### END CODE HERE

        # Để sử dụng hàm softmax, ta phải thực thi trên CPU
        with tf.device('/cpu:0'):
            output = tf.nn.softmax(x)

        return output


# ## TODO 2: Cài Đặt Block CNN trong lớp ConvBlock
# Sử dụng `tf.keras.layers.Conv2D` và `tf.keras.layers.MaxPool2D` để cài đặt tầng convolution và tầng pooling

# ## TODO 3: Gọi các tầng trong ConvBlock của lớp ConvBlock
# Hãy gọi các tầng đã cài đặt trọng lớp ConvBlock trong hàm call

# ## TODO 4: Khai báo ConvBlock 1,2,3,4,5 trong mô hình CNN
# Gọi ConvBlock đã cài đặt ở trên

# ## TODO 5: Khai báo Tầng Fully Connected Layer cho mô hình CNN
# Gọi `tf.keras.layers.Dense` để cài đặt tầng này

# ## TODO 6: Gọi các tầng Conv đã khai báo trong mô hình CNN ở trên
# Gọi các tầng Conv đã cài đặt

# ## TODO 7: Gọi tầng Fully Connected Layer
# Hãy flatten tầng phía trước và gọi tầng fully connected layer để convert về ma trận có số chiều bằng số lớp cần phân loại

# # Huấn Luyện
# Đoạn code này thực hiện quá trình huấn luyện mô hình CNN. Mỗi lần chạy mô hình sẽ lấy `batch_size` mẫu dữ liệu, feedforward, tính loss, và cập nhật gradient cho toàn bộ trọng số. Toàn bộ quá trình này được thực hiện trong hàm `fit()` được build sẵn trong model keras.
# 
# Sau khi huấn luyện xong, chúng ta sẽ sử dụng mô hình để phân lớp các ảnh trong tập test bằng hàm `predict()`

# In[ ]:


device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'
batch_size = 32
epochs = 16

with tf.device(device):
    # Khởi tạo model
    model = CNN(num_classes)

    # Tạo callback để lưu model có accuracy trên tập validation tốt nhất
    mcp = tf.keras.callbacks.ModelCheckpoint("my_model.h5", monitor="val_acc",
                                             save_best_only=True, save_weights_only=True)

    # Compile model
    model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Huấn luyện
    model.fit(x_train, y_train_ohe, batch_size=batch_size, epochs=epochs,
              validation_data=(x_valid, y_valid_ohe), verbose=1, callbacks=[mcp])

# # Dự Đoán các ảnh trên tập test
# 
# Chúng ta sử dụng mô hình đã được huấn luyện bên trên để dự đoán cho các ảnh trong tập test, xuất ra file CSV và submit kết quả lên Kaggle:
# 
# [Link nộp kết quả](https://www.kaggle.com/c/vietai-fc-cnn-assignment/submissions)

# ## Tạo và load model đã lưu trước đó

# In[ ]:


# Load best model
model = CNN(num_classes)

# Thiết lập kích thước input cho model
dummy_x = tf.zeros((1, 224, 224, 3))
model._set_inputs(dummy_x)

# Load model đã lưu trước đó trong quá trình huấn luyện
model.load_weights('my_model.h5')
print("Model đã được load")

# ## Dự đoán nhãn của các ảnh trên tập test

# Sử dụng hàm predict để dự đoán:

# In[ ]:


pred = model.predict(test_x)

# pred là một ma trận xác suất của ảnh trên các lớp.
# Ta lấy lớp có xác suất cao nhất trên từng ảnh bằng hàm argmax
pred_labels = np.argmax(pred, axis=1)

# Hiển thị thử kết quả của tập test:

# In[ ]:


test_df['label'] = pred_labels
test_df.head(20)

# Lưu kết quả thành file CSV:

# In[ ]:


test_df.to_csv("submission.csv", index=False)

# ## Nộp kết quả lên Kaggle

# 1. Truy cập vào [Kaggle](https://www.kaggle.com), đăng ký/ đăng nhập tài khoản.
# 
# 2. Truy cập vào đường dẫn của competition [VietAI Foundation Course - CNN Assignment](https://www.kaggle.com/t/1ca504e0910d4bfc9ba0ad0ffca12e2e).
# 
# 3. Nhấn vào nút **Join Competition**.
# ![alt text](https://storage.googleapis.com/vietai/Screen%20Shot%202019-05-13%20at%2018.48.12.png)
# 
# 4. Nhấn vào nút **I Understand and Accept**.
# ![alt text](https://storage.googleapis.com/vietai/Screen%20Shot%202019-05-13%20at%2018.48.52.png)
# 
# 5. Chọn **Team**.
# ![alt text](https://storage.googleapis.com/vietai/Screen%20Shot%202019-05-13%20at%2018.49.43.png)
# 
# 6. Đặt team name theo đúng họ và tên của bạn và bấm **Save team name**.
# ![alt text](https://storage.googleapis.com/vietai/Screen%20Shot%202019-05-13%20at%2018.50.30.png)
# 
# 7. Để nộp file CSV vừa tạo, các bạn nhấp vào **Submit Predictions**.
#  ![alt text](https://storage.googleapis.com/vietai/Screen%20Shot%202019-05-13%20at%2018.51.39.png)
#  
# 8. Upload file CSV và nộp.
#  ![alt text](https://storage.googleapis.com/vietai/Screen%20Shot%202019-05-13%20at%2018.52.19.png)
# 
# 9. Sau khi nộp, màn hình sẽ hiện ra kết quả, để biết vị trí mình trên leaderboard, các bạn nhấp vào **Jump to your position on the leaderboard**.
#  ![alt text](https://storage.googleapis.com/vietai/Screen%20Shot%202019-05-13%20at%2018.55.23.png)
# 
# 10. Leaderboard sẽ như sau:
#  ![alt text](https://storage.googleapis.com/vietai/Screen%20Shot%202019-05-13%20at%2018.55.32.png)

# # Thang điểm
# 
# - Hoàn tất codes trên Notebook: 7đ
# - Kaggle:
#   
#   + Vượt qua baseline1: 3đ
#   + Vượt qua baseline2: +1đ
#   + Vượt qua baseline3: +1đ

# # Authors: Quoc Pham, Chuong Huynh
