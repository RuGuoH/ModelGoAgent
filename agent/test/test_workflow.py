import requests

API_URL = "http://127.0.0.1:8012/v1/chat/completions"

def main():
    # 测试输入
    # test_input = "我想要把deep sequoia和pubmed分别用bigtranslate翻译之后，将翻译后的deep sequoia和pubmed与bigtranslate打包出售，请问违反了什么许可证问题"
    # test_input = '我想把arxiv和stack exchange分别用bigtranslate翻译之后，与deep sequoia和free law打包出售，请问违反了什么许可证问题'

    # test_input = "我用wiki微调了bert，将其与baize和bloom组成MoE后，让整个模型为外界提供API服务，请问违反了什么许可证问题"
    # test_input = "我用Flickr微调了MaskFormer，将其与i2vgen和detr组成MoE后，将整个模型共享，请问违反了什么许可证问题"

    # test_input = "我将jamendo输入由whisper、baize、stable diffusion、i2vgen-xl几个模型依次组成的生成链递归调用，前一个模型的输出作为后一个模型的输入。将最后输出的数据集共享出去，请问违反了什么许可证问题"
    # test_input = "我将jamendo输入由whisper、baize、stable diffusion、i2vgen-xl几个模型依次组成的生成链，前一个模型的输出作为后一个模型的输入，将过程中用到的模型和输出的数据集组合起来，最后一起共享出去，请问违反了什么许可证问题"

    # test_input = "利用wikimedia数据集将i2vgen的知识蒸馏到xclip中，然后将此模型与maskformer和whisper进行融合后出售。"
    # test_input = "利用wiki数据集将bloom的知识蒸馏到Bert中，然后将此模型与Llama2和BigTranslater融合后出售"

    test_input = '''
            将fake_work3和baize打包出售
    '''

    # 构造请求体
    payload = {
        "messages": [
            {"role": "user", "content": test_input}
        ],
        "stream": False,
        "userId": "test_user",
        "conversationId": "test_conv_work_id"
    }

    print("====== 测试输入 ======")
    print(test_input)
    print("\n====== 调用 /v1/chat/completions... ======\n")

    # 发送请求
    response = requests.post(API_URL, json=payload)

    print("====== 原始响应 ======")
    print(response.text)

    try:
        data = response.json()
        print("\n====== assistant 输出 ======")
        print(data["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"\n(响应解析失败: {e})")

if __name__ == "__main__":
    main()
