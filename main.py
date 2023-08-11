from flask import Flask, jsonify, request
from faster_whisper import WhisperModel
from flask_cors import *

from gevent import pywsgi
from urllib.parse import parse_qs
import base64
import uuid
import datetime

import json
import torch
import wenetruntime as wenet
#from simplelog import logger


app = Flask(__name__)
CORS(app, supports_credentials=True)


#
@app.route('/api/speechrecognition', methods=['POST'])
@cross_origin()
def get_result():
    print("recv from :"+request.remote_addr)
   # print("recv:")
    print(request.content_length)
#    print(bytes(request.get_data()))
    data = parse_qs(request.get_data())
    try:
        base_data = data[b'upfile_b64'][0]
    except:
        result={'error': '缺少upfile_b64'}
        return jsonify(result), 400

    try: 
       sr_type = data[b'sr_type'][0]
    except:
       sr_type = ''

    print(sr_type)
#    print(data)
#    print(base_data)
#    print(type(base_data))
    wav=base64.b64decode(base_data)

    i = datetime.datetime.now()
    filename = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-'))+str(uuid.uuid4())+".wav"
    with open(filename, 'wb') as f:  # 在接收方计算机上创建要写入的文件
        f.write(wav)
        f.close()

    if sr_type == b'wenet':
        decoder = wenet.Decoder(lang='chs')
        ret = decoder.decode_wav(filename)
        data = json.loads(ret)
        print(ret)
        result = {'result': data["nbest"][0]["sentence"]}
        return jsonify(result)


#    model_size = "large-v2"
    model_size = "medium"

# Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16",cpu_threads=8, num_workers=4,local_files_only=True)


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

