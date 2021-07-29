# tfann

Tiny Fast Artificial Neural Network Library.

It is based on specially crafted SIMD Matrix library.
It can save network structure to file.
It can generate pure dart code with no dependencies from a network.

## Getting Started

typical usage:
```
import 'package:tfann/tfann.dart';

...

List<TrainData> xor_data = [
      /*  output: column  1 - XOR of 3 bits, column  2 - AND of 3 bits,
       column  3 - OR of 3 bits, column  4 - if exactly two bits ON,
      */
      TrainData.lists([-1, -1, -1], [0, 0, 0, 0]),
      TrainData.lists([1, 1, -1], [0, 0, 1, 1]),
      TrainData.lists([1, -1, -1], [1, 0, 1, 0]),
      TrainData.lists([-1, 1, -1], [1, 0, 1, 0]),
      TrainData.lists([-1, -1, 1], [1, 0, 1, 0]),
      TrainData.lists([1, 1, 1], [1, 1, 1, 0]),
      TrainData.lists([1, -1, 1], [0, 0, 1, 1]),
      TrainData.lists([-1, 1, 1], [0, 0, 1, 1]),
    ];

final xor_net =
        TfannNetwork.full([3, 5, 4], activation: ActivationFunctionType.gelu);

print("before training...");
xor_data.forEach((data) => print(
        "in: ${data.input.toList()} out: ${xor_net.feedForward(data.input).toList()} expected: ${data.output.toList()}"));

// train network
for (int i = 0; i < 7000; ++i) {
      xor_data.forEach((data) {
        xor_net.train(data, learningRate: 0.06);
      });
}

print("after training...");

    
xor_data.forEach((data) => print(
    "in: ${data.input.toList()} out: ${xor_net.feedForward(data.input).toList()} expected: ${data.output.toList()}"));

...


```

To save the network:

```
await xor_net.save("binary.net");
```

To load the network:

```
var xor_net = TfannNetwork.fromFile("binary.net")!;
```

You may also compile the network into pure dart code. It is very good for production stage.

The produced code have no dependencies at all, even not this package.

Usage:
```
print(xor_net.compile());
```

Output:
```
import 'dart:typed_data';
import 'dart:math';

double gelu(double x) {      return 0.5*x*(1+tanh(0.7978845608028653558798921198687*(x+0.044715*x*x*x)));}


final List<Float32x4List> Lweight_tfann_evaluate_0 = [Int64List.fromList([-4684549482552772993, 3205952384]).buffer.asFloat32x4List(), Int64List.fromList([-4658463640062052053, 1067246068]).buffer.asFloat32x4List(), Int64List.fromList([4559296036336047244, 1060296640]).buffer.asFloat32x4List(), Int64List.fromList([4559652263048579599, 3214348148]).buffer.asFloat32x4List(), Int64List.fromList([4584575986206599728, 1068362496]).buffer.asFloat32x4List()];
final Float32x4List Lbias_tfann_evaluate_0 = Int64List.fromList([4528448722206921976, -4714918008769206507, 3215666407, 0]).buffer.asFloat32x4List();
final List<Float32x4List> Lweight_tfann_evaluate_1 = [Int64List.fromList([-4691824521505253971, -4665807858037931927, 3218341393, 0]).buffer.asFloat32x4List(), Int64List.fromList([-4629209937054397183, -4682969759229455083, 3189351279, 0]).buffer.asFloat32x4List(), Int64List.fromList([4406611430715703517, 4434756430857720869, 3199162943, 0]).buffer.asFloat32x4List(), Int64List.fromList([4546870734602276260, 4576005591552021506, 1051710346, 0]).buffer.asFloat32x4List()];
final Float32x4List Lbias_tfann_evaluate_1 = Int64List.fromList([-4605361703785376108, -4673147291425281497]).buffer.asFloat32x4List();


List<double> tfann_evaluate(List<double> inData) 
{
  assert(inData.length == 3);
  Float32List input = Float32List(4);
  for (int i = 0; i< 3; ++i) input[i] = inData[i];
  Float32x4List currentTensor = input.buffer.asFloat32x4List();
  Float32List outputTensor;
  outputTensor = Float32List(8);
  for (int r = 0; r < 5; ++r)
  {
    Float32x4List weightRow = Lweight_tfann_evaluate_0[r];
    Float32x4 sum = currentTensor[0]*weightRow[0];
    outputTensor[r] = sum.z + sum.y + sum.x ;
  }
  currentTensor = outputTensor.buffer.asFloat32x4List();
  for (int i = 0; i < 2; ++i)
    currentTensor[i]+=Lbias_tfann_evaluate_0[i];
  for (int i = 0; i < 5; ++i)
    outputTensor[i]=gelu(outputTensor[i]);
  outputTensor = Float32List(4);
  for (int r = 0; r < 4; ++r)
  {
    Float32x4List weightRow = Lweight_tfann_evaluate_1[r];
    Float32x4 sum = Float32x4.zero();
    for (int i = 0; i < 2; ++i)
    {     sum+=currentTensor[i]*weightRow[i];   }
    outputTensor[r] = sum.z + sum.y + sum.x + sum.w;
  }
  currentTensor = outputTensor.buffer.asFloat32x4List();
    currentTensor[0]+=Lbias_tfann_evaluate_1[0];
  for (int i = 0; i < 4; ++i)
    outputTensor[i]=gelu(outputTensor[i]);
  return currentTensor.buffer.asFloat32List(0,4).toList();
}
```