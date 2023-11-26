# How to create a similarity api with copilot

Copilot 以及 copilot chat可以快速提升生产力。该项目在copilot chat的提示下完成了一个向量相似度接口。
向量相似度在搜索推荐中至关重要，将它做成接口可以保证业务的稳定性和较快的响应速度。在学习了copilot chat的开发方法后，在copilot的协作下，可以快速开发相似度接口。

## 项目拆解：向copilot chat提问
![image](https://github.com/LeiShu-BNU/create_a_similarity_api_with_copilot/assets/81801688/1fe394a6-eabb-4b4f-85b7-c128f56a01d8)

GitHub Copilot：
创建一个向量相似度接口的步骤可能包括：

导入必要的库，如numpy。
定义一个函数，该函数接受两个向量作为输入。
在函数内部，计算两个向量的点积和模长。
使用点积和模长的公式计算相似度。
返回相似度。
以下是这个过程的Python代码：

'''
import numpy as np

def vector_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    
    similarity = dot_product / (norm_a * norm_b)
    
    return similarity
'''

这个函数可以计算两个向量的余弦相似度，这是一种常用的向量相似度度量。


在对第一个提问的回答中，copilot chat不仅拆分了必要的步骤，并且输出了核心步骤的代码。

## 补充细节

![image](https://github.com/LeiShu-BNU/create_a_similarity_api_with_copilot/assets/81801688/0248f11b-a10c-4e77-9347-b0ba08a2245f)

GitHub Copilot：
创建一个Flask应用的步骤可能包括：

导入Flask库。
创建一个Flask应用实例。
定义一个路由，该路由对应你的向量相似度函数。
在该路由的函数中，获取请求中的两个向量，调用向量相似度函数，然后返回结果。
以下是这个过程的Python代码：
'''
from flask import Flask, request
import numpy as np

app = Flask(__name__)

def vector_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    
    similarity = dot_product / (norm_a * norm_b)
    
    return similarity

@app.route('/similarity', methods=['POST'])
def similarity():
    data = request.get_json()
    vec1 = data['vec1']
    vec2 = data['vec2']
    result = vector_similarity(vec1, vec2)
    return {'similarity': result}

if __name__ == '__main__':
    app.run(debug=True)
'''

这个Flask应用定义了一个POST路由'/similarity'，它接受一个包含两个向量的JSON请求，计算这两个向量的相似度，然后返回结果。

## 进一步提示，完善流程

![image](https://github.com/LeiShu-BNU/create_a_similarity_api_with_copilot/assets/81801688/28b19ea9-5cc6-4f5e-abb8-73d2e4517b9d)

GitHub Copilot：
要使用sentence_transformers库加载模型并获取向量，你需要先安装这个库，然后在你的代码中导入它。你可以使用SentenceTransformer类的encode方法来获取句子的向量表示。

以下是如何修改你的代码来实现这个功能：
'''
from flask import Flask, request
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)

# 加载模型
model = SentenceTransformer('all-MiniLM-L6-v2')

def vector_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    
    similarity = dot_product / (norm_a * norm_b)
    
    return similarity

@app.route('/similarity', methods=['POST'])
def similarity():
    data = request.get_json()
    sentence1 = data['sentence1']
    sentence2 = data['sentence2']
    
    # 获取句子的向量表示
    vec1 = model.encode(sentence1)
    vec2 = model.encode(sentence2)
    
    result = vector_similarity(vec1, vec2)
    return {'similarity': result}

if __name__ == '__main__':
    app.run(debug=True)
'''

这个修改后的Flask应用现在接受两个句子作为输入，使用sentence_transformers模型将它们转换为向量，然后计算这两个向量的相似度。

## 使用inline chat功能，修改指定行
![image](https://github.com/LeiShu-BNU/create_a_similarity_api_with_copilot/assets/81801688/25deb80e-0fea-408a-a6f5-241759eb6a60)


## 编写测试脚本
![image](https://github.com/LeiShu-BNU/create_a_similarity_api_with_copilot/assets/81801688/26eba198-a0bd-4808-be43-feeee123a8ec)

GitHub Copilot
首先，我们需要确定API的完整URL，HTTP方法（GET，POST等），以及需要传递的参数。然后，我们可以使用Python的requests库来调用API。

以下是一个基本的示例，假设API的URL是http://localhost:5000/similarity，我们使用POST方法，并传递JSON格式的参数。
'''
import requests

# API的URL
url = "http://localhost:5000/similarity"

# 需要传递的参数
data = {
    "param1": "value1",
    "param2": "value2"
}

# 发送POST请求
response = requests.post(url, json=data)

# 打印响应
print(response.json())
'''

请根据实际情况修改上述代码中的URL和参数。

## 函数化
![image](https://github.com/LeiShu-BNU/create_a_similarity_api_with_copilot/assets/81801688/b11f5fbc-4d03-42e7-a851-99566b855ff6)

结果：
'''
def calculate_similarity(sentence1, sentence2):

    # API的URL
    url = "http://localhost:5000/similarity"

    # 需要传递的参数
    data = {
        "sentence1": sentence1,
        "sentence2": sentence2
    }

    # 发送POST请求
    response = requests.post(url, json=data)

    # 获取相似度结果
    similarity = response.json().get("similarity")

    return float(similarity)
'''

