# tfann

Tiny Fast Artificial Neural Network Library.

It uses internal tiny SIMD Matrix library.
It can save network structure to file.
It can generate pure dart code with no dependencies from a network.

## Getting Started

typical usage:

```dart
import 'package:tfann/tfann.dart';

...

List<TrainSetInputOutput> xor_data = [
      /*  output: column  1 - XOR of 3 bits, column  2 - AND of 3 bits,
       column  3 - OR of 3 bits, column  4 - if exactly two bits ON,
      */
      TrainSetInputOutput.lists([-1, -1, -1], [0, 0, 0, 0]),
      TrainSetInputOutput.lists([1, 1, -1], [0, 0, 1, 1]),
      TrainSetInputOutput.lists([1, -1, -1], [1, 0, 1, 0]),
      TrainSetInputOutput.lists([-1, 1, -1], [1, 0, 1, 0]),
      TrainSetInputOutput.lists([-1, -1, 1], [1, 0, 1, 0]),
      TrainSetInputOutput.lists([1, 1, 1], [1, 1, 1, 0]),
      TrainSetInputOutput.lists([1, -1, 1], [0, 0, 1, 1]),
      TrainSetInputOutput.lists([-1, 1, 1], [0, 0, 1, 1]),
    ];

final xor_net =
        TfannNetwork.full([3, 5, 4], [ActivationFunctionType.uscsls, ActivationFunctionType.uscsls]);

print("before training...");
xor_data.forEach((data) => print(
        "in: ${data.input.toList()} out: ${xor_net.feedForward(data.input).toList()} expected: ${data.output.toList()}"));

// train network
// train method takes a single TrainSet and runs it only once.
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

```dart
await xor_net.save("binary.net");
```

To load the network:

```dart
var xor_net = TfannNetwork.fromFile("binary.net")!;
```

While developing, use the  --enable-asserts flag, in order to catch bugs.

You may also compile the network into pure dart code. It is very good for production stage.

The produced code have no dependencies at all, even not this package.

Usage:

```dart

print(compileNetwork(xor_net));

```

Output:

```dart

import 'dart:typed_data';
import 'dart:math';

double uscsls(double x) {  x += 0.45353;  if (x > 4) return 1 + 0.25 * x;  if (x > -2) {    var x2 = x * x;    var x3 = x2 * x;    return (-11/576)*x3+(7/96)*x2+(7/12)*x-5/18;  }  return 0.0625 * x - 0.875;}
Float32x4 uscslsX4(Float32x4 x) {
  x += Float32x4.splat(0.45353);
  Int32x4 greater4 = x.greaterThan(Float32x4.splat(4));
  Float32x4 x2 = x * x;
  Float32x4 branch1Result = x.scale(0.25) + Float32x4.splat(1);
  Float32x4 x3 = x2 * x;

  Int32x4 lessThanMinus2 = x.lessThanOrEqual(Float32x4.splat(-2));
  Float32x4 branch3Result = x.scale(0.0625) - Float32x4.splat(0.875);  
  
  return greater4.select(
      branch1Result,
      lessThanMinus2.select(
          branch3Result,
          x3.scale(-11 / 576) +
              x2.scale(7 / 96) +
              x.scale(7 / 12) -
              Float32x4.splat(5 / 18)));
}
final List<Float32x4List> Lweight_tfann_evaluate_0 = [Uint32List.fromList([3209962030, 3216137746, 3210639258, 0]).buffer.asFloat32x4List(), Uint32List.fromList([3214929675, 3215502053, 1067274155, 0]).buffer.asFloat32x4List(), Uint32List.fromList([1065607033, 3209239346, 1067661884, 0]).buffer.asFloat32x4List(), Uint32List.fromList([3216963690, 1062113278, 1067189068, 0]).buffer.asFloat32x4List(), Uint32List.fromList([3217148364, 3217230598, 3217250982, 0]).buffer.asFloat32x4List()];
final Float32x4List Lbias_tfann_evaluate_0 = Uint32List.fromList([1068705506, 1067842099, 1055350071, 1052725129, 3202069356, 0, 0, 0]).buffer.asFloat32x4List();
final List<Float32x4List> Lweight_tfann_evaluate_1 = [Uint32List.fromList([1059405630, 3216279887, 1067104365, 1066584353, 1058614923, 0, 0, 0]).buffer.asFloat32x4List(), Uint32List.fromList([3213682520, 3189869051, 1056767980, 1033034986, 1064559518, 0, 0, 0]).buffer.asFloat32x4List(), Uint32List.fromList([1065551239, 3207167254, 1051821556, 1058158382, 3208697560, 0, 0, 0]).buffer.asFloat32x4List(), Uint32List.fromList([1052874554, 1061131736, 3210522161, 3205716665, 3215786676, 0, 0, 0]).buffer.asFloat32x4List()];
final Float32x4List Lbias_tfann_evaluate_1 = Uint32List.fromList([1055442582, 1065071056, 1060193668, 1047759191]).buffer.asFloat32x4List();


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
    outputTensor[r] = sum.x + sum.y + sum.z ;
  }
  currentTensor = outputTensor.buffer.asFloat32x4List();
  for (int i = 0; i < 2; ++i)
    currentTensor[i]+=Lbias_tfann_evaluate_0[i];
  currentTensor[0]=uscslsX4(currentTensor[0]);
  outputTensor[4]=uscsls(outputTensor[4]);
  outputTensor = Float32List(4);
  for (int r = 0; r < 4; ++r)
  {
    Float32x4List weightRow = Lweight_tfann_evaluate_1[r];
    Float32x4 sum = Float32x4.zero();
    for (int i = 0; i < 2; ++i)
    {     sum+=currentTensor[i]*weightRow[i];   }
    outputTensor[r] = sum.x + sum.y + sum.z + sum.w;
  }
  currentTensor = outputTensor.buffer.asFloat32x4List();
    currentTensor[0]+=Lbias_tfann_evaluate_1[0];
  currentTensor[0]=uscslsX4(currentTensor[0]);
  return currentTensor.buffer.asFloat32List(0,4).toList();
}

```

## Tips

The train method returns both the forward error (before changing the weights) and the back-propagated error.
You can use the back-propagated error in cases of chaining networks (like RNN or LSTM which not included here).
In these case, you would need to train the network with set of Input and Error. use TrainSetInputError for this case.
You can also use the back-propagated error to create what's called "deep fake" or "deep dream".

If you get NaN in your weights, then you got the infamous exploding gradient problem. Try again and set propErrorLimit (one of train arguments) to a small value (1/number_of_layers might be a good value).
Also, if you are using unbounded activation functions, try to set few of the layers to bell shaped activation function. The bell functions helps to stabilize the network.
