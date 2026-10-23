# Paradee text-to-speech in Python: onnxruntime + misaki, the phonemizer Kokoro and Paradee were trained on.
#   pip install onnxruntime "misaki[en]" soundfile
#   python say.py "Hello from Python." hello.wav [paradee_v2f_lite.onnx]
# Needs config.json (the phoneme vocabulary) and an ONNX file next to it. One call takes at most 512 ids, so long text goes
# sentence by sentence. Words outside misaki's lexicon are dropped unless you give misaki an eSpeak fallback (see its README).
import json, re, sys
import numpy as np, onnxruntime as ort, soundfile as sf
from misaki import en

text = sys.argv[1] if len(sys.argv) > 1 else "Hello from Paradee."
out = sys.argv[2] if len(sys.argv) > 2 else "out.wav"
model = sys.argv[3] if len(sys.argv) > 3 else "paradee_v2f_lite.onnx"

vocab = json.load(open("config.json"))["vocab"]
g2p = en.G2P(trf=False, british=False)
sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
speed = np.ones(1, np.float32)

audio = []
for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
    phonemes, _ = g2p(sentence)
    ids = [0] + [vocab[p] for p in phonemes if p in vocab][:510] + [0]
    audio.append(sess.run(None, {"input_ids": np.array([ids], np.int64), "speed": speed})[0][0])
sf.write(out, np.concatenate(audio), 24000)
print(f"{out}: {sum(len(a) for a in audio) / 24000:.1f} s of audio")
