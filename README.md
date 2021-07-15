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
      TrainData.lists([-1, -1, -1], [-1, -1, -1, -1]),
      TrainData.lists([1, 1, -1], [-1, -1, 1, 1]),
      TrainData.lists([1, -1, -1], [1, -1, 1, -1]),
      TrainData.lists([-1, 1, -1], [1, -1, 1, -1]),
      TrainData.lists([-1, -1, 1], [1, -1, 1, -1]),
      TrainData.lists([1, 1, 1], [1, 1, 1, -1]),
      TrainData.lists([1, -1, 1], [-1, -1, 1, 1]),
      TrainData.lists([-1, 1, 1], [-1, -1, 1, 1]),
    ];

final xor_net =
        TfannNetwork.full([3, 3, 4], activation: activationLogisticSigmoid);

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

Add it to your `pubspec.yaml`:

```
dependencies:
  flutter:
    sdk: flutter
  tfann:
    git:
      url: https://github.com/gnudles/tfann_dart.git
      ref: main
```

You can compile the network also:
```
print(xor_net.compile());
```

output:
```
import 'dart:typed_data';
import 'dart:math';

double logisticSigmoid(double x) { return 2 / (1 + exp(-x)) - 1;}
double absSigmoid(double x) { return x / (1 + x.abs());}
double tanhSigmoid(double x) {  var e2x = exp(2 * x);    return (e2x - 1) / (e2x + 1); }
List<double> tfann_evaluate(List<double> inData) 
{
  assert(inData.length == 3);
  Float32List input = Float32List(4);
  for (int i = 0; i< 3; ++i) input[i] = inData[i];
  final List<Float32x4List> Lweight0 = [Int64List.fromList([-4560500300062533076, 3233142373]).buffer.asFloat32x4List(), Int64List.fromList([-4664829934789512787, 3208695953]).buffer.asFloat32x4List()];
  final Float32x4List Lbias0 = Int64List.fromList([4471705980817066878, 0]).buffer.asFloat32x4List();
  final List<Float32x4List> Lweight1 = [Int64List.fromList([-4508054997718421971, 0]).buffer.asFloat32x4List(), Int64List.fromList([-4518814174297465773, 0]).buffer.asFloat32x4List(), Int64List.fromList([-4513229368183927154, 0]).buffer.asFloat32x4List(), Int64List.fromList([4701420670303776536, 0]).buffer.asFloat32x4List()];
  final Float32x4List Lbias1 = Int64List.fromList([-4551560134535729087, -4587602258807592186]).buffer.asFloat32x4List();
  Float32x4List currentTensor = input.buffer.asFloat32x4List();
  Float32List outputTensor;
  outputTensor = Float32List(4);
  for (int r = 0; r < 2; ++r)
  {   Float32x4 sum = Float32x4.zero();
   Float32x4List weightRow = Lweight0[r];
    for (int i = 0; i < 1; ++i)
    {     sum+=currentTensor[i]*weightRow[i];   }
    outputTensor[r] = sum.w + sum.x + sum.y + sum.z;
  }
  currentTensor = outputTensor.buffer.asFloat32x4List();
    currentTensor[0]+=Lbias0[0];
  for (int i = 0; i < 2; ++i)
    outputTensor[i]=logisticSigmoid(outputTensor[i]);
  outputTensor = Float32List(4);
  for (int r = 0; r < 4; ++r)
  {   Float32x4 sum = Float32x4.zero();
   Float32x4List weightRow = Lweight1[r];
    for (int i = 0; i < 1; ++i)
    {     sum+=currentTensor[i]*weightRow[i];   }
    outputTensor[r] = sum.w + sum.x + sum.y + sum.z;
  }
  currentTensor = outputTensor.buffer.asFloat32x4List();
    currentTensor[0]+=Lbias1[0];
  for (int i = 0; i < 4; ++i)
    outputTensor[i]=logisticSigmoid(outputTensor[i]);
  return currentTensor.buffer.asFloat32List(0,4).toList();
}
```