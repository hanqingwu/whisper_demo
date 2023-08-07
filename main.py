from flask import Flask, jsonify, request
from faster_whisper import WhisperModel
from flask_cors import *

from gevent import pywsgi
from urllib.parse import parse_qs
import base64
import uuid

#from simplelog import logger


app = Flask(__name__)
CORS(app, supports_credentials=True)


# 
@app.route('/api/speechrecognition', methods=['POST'])
@cross_origin()
def get_result():
    print("recv")
    print(request.content_length)
#    print(bytes(request.get_data()))
    data=parse_qs(request.get_data())
    try:
        base_data=data[b'upfile_b64'][0]
    except:
        result={'error': '缺少upfile_b64'} 
        return jsonify(result), 400

#    print(data)
#    print(base_data)
#    print(type(base_data))
    wav=base64.b64decode(base_data)

    filename = str(uuid.uuid4())+".wav"
    with open(filename, 'wb') as f:  # 在接收方计算机上创建要写入的文件
        f.write(wav)
        f.close()

#    model_size = "large-v2"
    model_size = "medium"

# Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16") #,local_files_only=True)

# or run on GPU with INT8
#model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8

#    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    try:
    	segments, info = model.transcribe(filename, beam_size=5, language="zh", vad_filter=True, initial_prompt="以下是普通话的句子")
    except:
        print("trancibe err ",e)
        result = {'error': '错误的音频'} 
        return jsonify(result)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    ret=""
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        ret+=segment.text

    print("ret: "+ret)
    result = {'result': ret} #wav.decode('utf-8')}

    return jsonify(result)


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
    server.serve_forever()
i#    app.run()

