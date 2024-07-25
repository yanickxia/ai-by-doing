input_texts = [
    "我有一个苹果",
    "你好吗",
    "见到你很高兴",
    "我简直不敢相信",
    "我知道那种感觉",
    "我真的非常后悔",
    "我也这样以为",
    "这样可以吗",
    "这事可能发生在任何人身上",
    "我想要一个手机",
]
output_texts = [
    "I have a apple",
    "How are you",
    "Nice to meet you",
    "I can not believe it",
    "I know the feeling",
    "I really regret it",
    "I thought so, too",
    "Is that OK",
    "It can happen to anyone",
    "I want a iphone",
]

def count_char(input_texts):
    input_characters = set()  # 用来存放输入集出现的中文字
    for input_text in input_texts:  # 遍历输入集的每一个句子
        for char in input_text:  # 遍历每个句子的每个字
            if char not in input_characters:
                input_characters.add(char)
    return input_characters


input_characters = count_char(input_texts)


def count_word(output_texts):
    target_characters = set()  # 用来存放输出集出现的单词
    target_texts = []  # 存放加了句子开头和结尾标记的句子
    for target_text in output_texts:  # 遍历输出集的每个句子
        target_text = "> " + target_text + " <"
        target_texts.append(target_text)
        word_list = target_text.split(" ")  # 对每个英文句子按空格划分，得到每个单词
        for word in word_list:  # 遍历每个单词
            if word not in target_characters:
                target_characters.add(word)
    return target_texts, target_characters


target_texts, target_characters = count_word(output_texts)


input_characters = sorted(list(input_characters))  # 这里排序是为了每一次
target_characters = sorted(list(target_characters))  # 构建的字典都一样
# 构建字符到数字的字典，每个字符对应一个数字
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

# 构建反向字典，每个数字对应一个字符
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


num_encoder_tokens = len(input_characters)  # 输入集不重复的字数
num_decoder_tokens = len(target_characters)  # 输出集不重复的单词数
max_encoder_seq_length = max([len(txt) for txt in input_texts])  # 输入集最长句子的长度
max_decoder_seq_length = max([len(txt) for txt in target_texts])  # 输出集最长句子的长度


import numpy as np

# 创三个全为 0 的三维矩阵，第一维为样本数，第二维为句最大句子长度，第三维为每个字符的独热编码。
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)

for i, (input_text, target_text) in enumerate(
    zip(input_texts, target_texts)
):  # 遍历输入集和输出集
    for t, char in enumerate(input_text):  # 遍历输入集每个句子
        encoder_input_data[i, t, input_token_index[char]] = 1.0  # 字符对应的位置等于 1
    for t, char in enumerate(target_text.split(" ")):  # 遍历输出集的每个单词
        # 解码器的输入序列
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # 解码器的输出序列
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0


import tensorflow as tf

latent_dim = 256  # 循环神经网络的神经单元数

# 编码器模型
encoder_inputs = tf.keras.Input(shape=(None, num_encoder_tokens))  # 编码器的输入
encoder = tf.keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)  # 编码器的输出

encoder_states = [state_h, state_c]  # 状态值


# 解码器模型
decoder_inputs = tf.keras.Input(shape=(None, num_decoder_tokens))  # 解码器输入
decoder_lstm = tf.keras.layers.LSTM(
    latent_dim, return_sequences=True, return_state=True
)

# 初始化解码模型的状态值为 encoder_states
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# 连接一层全连接层，并使用 Softmax 求出每个时刻的输出
decoder_dense = tf.keras.layers.Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)  # 解码器输出


# 定义训练模型
model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()


# 定义优化算法和损失函数
model.compile(optimizer="adam", loss="categorical_crossentropy")

# 训练模型
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=10,
    epochs=200,
)


# 重新定义编码器模型
encoder_model = tf.keras.Model(encoder_inputs, encoder_states)
encoder_model.summary()



""" 重新定义解码器模型 """
decoder_state_input_h = tf.keras.Input(shape=(latent_dim,))  # 解码器状态 H 输入
decoder_state_input_c = tf.keras.Input(shape=(latent_dim,))  # 解码器状态 C 输入
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)  # LSTM 模型输出

decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)  # 连接一层全连接层
# 定义解码器模型
decoder_model = tf.keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

decoder_model.summary()


def decode_sequence(input_seq):
    """
    decoder_dense:中文句子的向量形式。
    """
    # 使用编码器预测出状态值
    states_value = encoder_model.predict(input_seq)

    # 构建解码器的第一个时刻的输入，即句子开头符号 >
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index[">"]] = 1.0
    stop_condition = False  # 设置停止条件
    decoded_sentence = []  # 存放结果
    while not stop_condition:
        # 预测出解码器的输出
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # 求出对应的字符
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        # 如果解码的输出为句子结尾符号 < ，则停止预测
        if sampled_char == "<" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True
        if sampled_char != "<":
            decoded_sentence.append(sampled_char)
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0
        # 更新状态，用来继续送入下一个时刻
        states_value = [h, c]
    return decoded_sentence

def answer(question):
    # 将句子转化为一个数字矩阵
    inseq = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype="float32")
    for t, char in enumerate(question):
        inseq[0, t, input_token_index[char]] = 1.0
    # 输入模型得到输出结果
    decoded_sentence = decode_sequence(inseq)
    return decoded_sentence


test_sent = "我有一个苹果"
result = answer(test_sent)
print("中文句子：", test_sent)
print("翻译结果：", " ".join(result))