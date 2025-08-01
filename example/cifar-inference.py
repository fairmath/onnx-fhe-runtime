import onnx
import numpy as np
import onnxruntime as ort
import os
from onnx import helper
from onnx import OperatorSetIdProto

from openfhe import *

def generate_keys():
    parameters = CCParamsCKKSRNS()
    parameters.SetRingDim(65536)
    parameters.SetScalingModSize(50)
    parameters.SetBatchSize(64)
    parameters.SetFirstModSize(55)
    parameters.SetScalingTechnique(FLEXIBLEAUTO)
    parameters.SetSecurityLevel(SecurityLevel.HEStd_NotSet)

    parameters.SetMultiplicativeDepth(8)
    cc = GenCryptoContext(parameters)

    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)
    cc.Enable(PKESchemeFeature.FHE)

    serverKP = cc.KeyGen()

    cc.EvalMultKeyGen(serverKP.secretKey)
    cc.EvalRotateKeyGen(serverKP.secretKey, [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64])

    return [cc, serverKP]


option = ort.SessionOptions()
option.register_custom_ops_library(os.getcwd()+'/../build/install/lib/libonnx-fhe-runtime.so')

onnx_opset = OperatorSetIdProto()
onnx_opset.domain = ''  # default domain: 'ai.onnx'
onnx_opset.version = 22

matmul = OperatorSetIdProto()
matmul.domain = 'fhe.ckks.matmul'
matmul.version = 1 

square = OperatorSetIdProto()
square.domain = 'fhe.ckks.square'
square.version = 1

loader = OperatorSetIdProto()
loader.domain = 'fhe.ckks.loader'
loader.version = 1

saver = OperatorSetIdProto()
saver.domain = 'fhe.ckks.saver'
saver.version = 1

sess = ort.InferenceSession('fhe-mlp-cifar10.1.onnx', sess_options=option)

cc, kp = generate_keys()

pict_class6 = [132,125,130,143,142,145,148,147,148,136,122,133,142,126,123,134,132,114,120,123,118,136,122,116,107,98,97,104,98,102,97,98,146,156,154,145,139,146,145,143,146,132]
pict_class1 = [11,14,18,21,19,17,23,23,17,17,18,21,13,19,24,25,32,32,30,32,33,35,30,28,27,25,26,24,26,23,22,17,15,17,21,23,22,25,31,31,23,20]
pict_class9 = [213,211,210,209,210,211,211,211,209,208,210,213,215,221,223,224,226,226,227,227,226,228,228,225,221,219,218,217,214,212,207,205,200,198,199,200,201,202,203,204,204,204]

pict = pict_class1

encoded_1 = cc.MakeCKKSPackedPlaintext(pict)
ciphertext1 = cc.Encrypt(kp.publicKey, encoded_1)
if not SerializeToFile("c1.bin", ciphertext1, BINARY):
    raise Exception("Exception writing ciphertext to ciphertext.txt")

if not SerializeToFile('cc.bin', cc, BINARY):
    raise Exception("Exception writing cryptocontext to cryptocontext.txt")

if not cc.SerializeEvalMultKey('mk.bin', BINARY):
    raise Exception("Error writing eval mult keys")

if not cc.SerializeEvalAutomorphismKey('rk.bin', BINARY):
    raise Exception("Error writing rotation keys")

rk_np = np.array(['rk.bin'], dtype=object)
mk_np = np.array([''], dtype=object)
cc_np = np.array(['cc.bin'], dtype=object)
c1_np = np.array(['c1.bin'], dtype=object)
pk_np = np.array([''], dtype=object)

results = sess.run(None, {'cc': cc_np, 'rk': rk_np, 'mk': mk_np, 'in': c1_np, 'pk': pk_np})

rc, ok = DeserializeCiphertext(results[0][0], BINARY)
if not ok:
    raise Exception("can't deserialize crypto context")

plaintext = cc.Decrypt(kp.secretKey, rc)
rv = plaintext.GetRealPackedValue()
rv = rv[:10]

print('FHE MODEL:')
print(rv)

print('ORIGINAL MODEL')
mod = ort.InferenceSession('cifar_mlp_model.onnx')
minput = np.array(pict, dtype=np.float32).reshape(1, 42)
mresults = mod.run(None, {'onnx::Reshape_0': minput})
print(mresults[0].tolist()[0])

