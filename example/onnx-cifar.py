import onnx
import numpy as np
import onnxruntime as ort
import os

from onnx import OperatorSetIdProto

option = ort.SessionOptions()
option.register_custom_ops_library(os.getcwd()+'/../.build/install/lib/libonnx-fhe-runtime.so')

w1raw = [
    [
        -0.03875972703099251,
        0.0680551528930664,
        -0.06490413844585419,
        -0.10815239697694778,
        0.07091706246137619,
        -0.028203535825014114,
        0.11830383539199829,
        -0.0897851511836052,
        0.15226858854293823,
        -0.01000060886144638,
        -0.01165691390633583,
        0.013466075994074345,
        -0.06594888865947723,
        -0.05108067765831947,
        -0.01399044506251812,
        0.06439095735549927,
        0.07006647437810898,
        -0.08487590402364731,
        0.07064574956893921,
        0.03488857299089432,
        -0.06671324372291565,
        0.05625331029295921,
        -0.0862308219075203,
        -0.07560621947050095,
        -0.04051411151885986,
        -0.04728633910417557,
        -0.1092899739742279,
        -0.1431095153093338,
        0.14111314713954926,
        -0.029851948842406273,
        -0.0035102388355880976,
        -0.07713109254837036,
        -0.12145429849624634,
        -0.13117322325706482,
        0.14669163525104523,
        0.02768847346305847,
        0.038077738136053085,
        0.11411625891923904,
        -0.09181822836399078,
        -0.0269293375313282,
        0.14557504653930664,
        0.12429235130548477
    ],
    [
        -0.050196681171655655,
        -0.006759939715266228,
        -0.0517086461186409,
        0.09696207195520401,
        -0.06006256863474846,
        -0.13121077418327332,
        -0.10363563150167465,
        -0.09623576700687408,
        0.08491531014442444,
        0.11091933399438858,
        0.1519412100315094,
        -0.039027661085128784,
        0.032067809253931046,
        -0.11543066799640656,
        0.0175127312541008,
        -0.11193965375423431,
        0.15166223049163818,
        -0.03027893789112568,
        -0.09539279341697693,
        0.11155616492033005,
        -0.1334274709224701,
        0.12746861577033997,
        -0.0884253978729248,
        0.15284407138824463,
        -0.05249090492725372,
        -0.06298898905515671,
        -0.03655409440398216,
        0.018661480396986008,
        -0.025813570246100426,
        -0.02613239921629429,
        0.07245451956987381,
        -0.13264767825603485,
        0.11231798678636551,
        0.11513183265924454,
        0.049937762320041656,
        0.13546842336654663,
        -0.01739351823925972,
        -0.06368014216423035,
        -0.0043213581666350365,
        -0.07494115084409714,
        -0.11245151609182358,
        -0.03247201070189476
    ]
]

w2raw = [
    [0.08616358041763306, -0.2307540774345398],
    [-0.49714425206184387, -0.27252262830734253],
    [-0.17419283092021942, -0.075792595744133],
    [-0.35192322731018066, 0.01008540391921997],
    [0.060600508004426956, -0.22160550951957703],
    [0.3916865885257721, 0.23882856965065002],
    [0.6621978878974915, 0.6609815359115601],
    [-0.6170975565910339, 0.2305549681186676],
    [-0.5278924107551575, 0.26486098766326904],
    [-0.1513623148202896, -0.3760269284248352]
]

wl1 = np.array(w1raw, dtype=np.double)
wl2 = np.array(w2raw, dtype=np.double)

initializers = [
    onnx.numpy_helper.from_array(wl1, name='w1'),
    onnx.numpy_helper.from_array(wl2, name='w2'),
]

inputs = [
    onnx.helper.make_tensor_value_info('cc', onnx.TensorProto.STRING, [1]),
    onnx.helper.make_tensor_value_info('mk', onnx.TensorProto.STRING, [1]),
    onnx.helper.make_tensor_value_info('rk', onnx.TensorProto.STRING, [1]),
    onnx.helper.make_tensor_value_info('in', onnx.TensorProto.STRING, [1]),
    onnx.helper.make_tensor_value_info('pk', onnx.TensorProto.STRING, [1]),
]
outputs = [onnx.helper.make_tensor_value_info('out', onnx.TensorProto.STRING, [1])]
nodes = [
    onnx.helper.make_node('fhe.ckks.loader', ['cc', 'rk', 'mk', 'in', 'pk'], ['cctx', 'icphr', 'pubkey'], domain='fhe.ckks.loader'),
    onnx.helper.make_node('fhe.ckks.matmul', ['cctx', 'icphr', 'w1'], ['layer1'], domain='fhe.ckks.matmul'),
    onnx.helper.make_node('fhe.ckks.square', ['cctx', 'layer1'], ['s1'], domain='fhe.ckks.square'),
    onnx.helper.make_node('fhe.ckks.matmul', ['cctx', 's1', 'w2'], ['layer2'], domain='fhe.ckks.matmul'),
    onnx.helper.make_node('fhe.ckks.saver', ['layer2'], ['out'], domain='fhe.ckks.saver'),
]
graph = onnx.helper.make_graph(nodes, 'cifar10', inputs=inputs, outputs=outputs, initializer=initializers)

onnx_opset = OperatorSetIdProto()
onnx_opset.domain = '' 
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

model = onnx.helper.make_model(graph, opset_imports=[onnx_opset, matmul, square, loader, saver])
model.ir_version = 10

mstr = model.SerializeToString()
with open('fhe-mlp-cifar10.onnx', 'wb') as f:
    f.write(mstr)
