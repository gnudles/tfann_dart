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
        TfannNetwork.full([3, 4, 4], [ActivationFunctionType.uscsls, ActivationFunctionType.uscsls]);

print("before training...");
xor_data.forEach((data) => print(
        "in: ${data.input.toList()} out: ${xor_net.feedForward(data.input).toList()} expected: ${data.output.toList()}"));

// train network
// train method takes a single TrainSet and runs it only once.
for (int i = 0; i < 10000; ++i) {
      xor_data.forEach((data) {
        xor_net.train(data, learningRate: 0.04);
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

final Float32x4 _SIMD0 = Float32x4.zero();
final Float32x4 _SIMD0_75 = Float32x4.splat(0.75);
final Float32x4 _SIMD0_5 = Float32x4.splat(0.5);
final Float32x4 _SIMD0_25 = Float32x4.splat(0.25);
final Float32x4 _SIMD0_125 = Float32x4.splat(0.125);
final Float32x4 _SIMD0_375 = Float32x4.splat(0.375);
final Float32x4 _SIMD0_625 = Float32x4.splat(0.625);
final Float32x4 _SIMD0_0625 = Float32x4.splat(0.0625);
final Float32x4 _SIMD0_03 = Float32x4.splat(0.03);
final Float32x4 _SIMD0_65625 = Float32x4.splat(0.65625);
final Float32x4 _SIMD0_065 = Float32x4.splat(0.065);
final Float32x4 _SIMD0_185 = Float32x4.splat(0.185);
final Float32x4 _SIMD0_104 = Float32x4.splat(0.104);
final Float32x4 _SIMD0_208 = Float32x4.splat(0.208);
final Float32x4 _SIMD0_704 = Float32x4.splat(0.704);
final Float32x4 _SIMDm0_8 = Float32x4.splat(-0.8);
final Float32x4 _SIMDm1_5 = Float32x4.splat(-1.5);
final Float32x4 _SIMD0_28125 = Float32x4.splat(0.28125);
final Float32x4 _SIMD1 = Float32x4.splat(1);
final Float32x4 _SIMD1_47 = Float32x4.splat(1.47);
final Float32x4 _SIMD1_6 = Float32x4.splat(1.6);
final Float32x4 _SIMD4 = Float32x4.splat(4);
final Float32x4 _SIMD8 = Float32x4.splat(8);
final Float32x4 _SIMDm2 = Float32x4.splat(-2);
final Float32x4 _SIMD0_875 = Float32x4.splat(0.875);
final Float32x4 _SIMD0_4 = Float32x4.splat(0.4);
final Float32x4 _SIMDm0_16 = Float32x4.splat(-0.16);
final Float32x4List _SimdSignMaskVector = Float32x4List.fromList(List.generate(
    16,
    (index) => Float32x4(
        (index & 1) != 0 ? -1.0 : 1.0,
        (index & 2) != 0 ? -1.0 : 1.0,
        (index & 4) != 0 ? -1.0 : 1.0,
        (index & 8) != 0 ? -1.0 : 1.0)));

double uscsls(double x) {
  if (x >= 1.6) return 0.065 * x + 0.704;
  if (x > -0.8) {
    var x2 = x * x;
    var x3 = x2 * x;
    return 0.125 * (x2 - x3) + 0.625 * x;
  }
  return 0.185 * x - 0.208;
}


Float32x4 uscslsX4(Float32x4 x) {
  Int32x4 greater1_6 = x.greaterThan(_SIMD1_6);
  Float32x4 x2 = x * x;

  Float32x4 branch1Result = x.scale(0.065) + _SIMD0_704;
  Float32x4 x3 = x2 * x;

  Int32x4 lessThanMinus0_8 = x.lessThanOrEqual(_SIMDm0_8);
  Float32x4 branch3Result = x.scale(0.185) - _SIMD0_208;

  return greater1_6.select(
      branch1Result,
      lessThanMinus0_8.select(
          branch3Result,
           (x2 - x3).scale(0.125) +  x.scale(0.625)));
}

final List<Float32x4List> Lweight_tfann_evaluate_0 = [Uint32List.fromList([1065924784, 3218828940, 3218824008, 0]).buffer.asFloat32x4List(), Uint32List.fromList([1074832170, 3207276024, 3207270630, 0]).buffer.asFloat32x4List(), Uint32List.fromList([3218045595, 1058827529, 1058838751, 0]).buffer.asFloat32x4List(), Uint32List.fromList([3213025879, 3213257327, 3213261317, 0]).buffer.asFloat32x4List()];
final Float32x4List Lbias_tfann_evaluate_0 = Uint32List.fromList([1051252787, 3212525348, 3213439945, 1049866728]).buffer.asFloat32x4List();
final List<Float32x4List> Lweight_tfann_evaluate_1 = [Uint32List.fromList([3232711821, 1078727539, 3223330061, 1083118854]).buffer.asFloat32x4List(), Uint32List.fromList([3220807383, 3217432562, 3229760405, 3194501247]).buffer.asFloat32x4List(), Uint32List.fromList([3223501112, 1079543989, 1069180988, 3181878151]).buffer.asFloat32x4List(), Uint32List.fromList([1078650051, 1071470358, 1085387923, 3224445642]).buffer.asFloat32x4List()];
final Float32x4List Lbias_tfann_evaluate_1 = Uint32List.fromList([1070831670, 3197145344, 1083611721, 1076128681]).buffer.asFloat32x4List();


List<double> tfann_evaluate(List<double> inData) 
{
  assert(inData.length == 3);
  Float32List input = Float32List(4);
  for (int i = 0; i< 3; ++i) input[i] = inData[i];
  Float32x4List currentTensor = input.buffer.asFloat32x4List();
  Float32List outputTensor;
  outputTensor = Float32List(4);
  for (int r = 0; r < 4; ++r)
  {
    Float32x4List weightRow = Lweight_tfann_evaluate_0[r];
    Float32x4 sum = currentTensor[0]*weightRow[0];
    outputTensor[r] = sum.x + sum.y + sum.z ;
  }
  currentTensor = outputTensor.buffer.asFloat32x4List();
    currentTensor[0]+=Lbias_tfann_evaluate_0[0];
  currentTensor[0]=uscslsX4(currentTensor[0]);
  outputTensor = Float32List(4);
  for (int r = 0; r < 4; ++r)
  {
    Float32x4List weightRow = Lweight_tfann_evaluate_1[r];
    Float32x4 sum = currentTensor[0]*weightRow[0];
    outputTensor[r] = sum.x + sum.y + sum.z + sum.w;
  }
  currentTensor = outputTensor.buffer.asFloat32x4List();
    currentTensor[0]+=Lbias_tfann_evaluate_1[0];
  currentTensor[0]=uscslsX4(currentTensor[0]);
  return currentTensor.buffer.asFloat32List(0,4).toList();
}

```

## Activation functions

Since this library tries the best to utilize your CPU, we propose some new activation functions, that are easy to calculate, and also gives good behavior.

[Abs Sigmoid](https://www.desmos.com/calculator/ukybfmgvot)

[DivLine Sigmoid](https://www.desmos.com/calculator/bvf5vfuola)

[Fast Bell](https://www.desmos.com/calculator/pvmo0rm7nt)

[Cubic Sigmoid](https://www.desmos.com/calculator/dfnyzvoucc)

[UACSLS](https://www.desmos.com/calculator/ftbmhwspfo)

[USCLS](https://www.desmos.com/calculator/mzjwraayia)

[USCSLS](https://www.desmos.com/calculator/dlfulx2isk)

[Funny Hat](https://www.desmos.com/calculator/vuuufb7g72)

[All activation functions](https://www.desmos.com/calculator/tyruwhpfth)

## Tips

The train method returns both the forward error (before changing the weights) and the back-propagated error.
You can use the back-propagated error in cases of chaining networks (like RNN or LSTM which not included here).
In these case, you would need to train the network with set of Input and Error. use TrainSetInputError for this case.
You can also use the back-propagated error to create what's called "deep fake" or "deep dream".

If you get NaN in your weights, then you got the infamous exploding gradient problem. Try again and set 'propErrorLimit' (one of train arguments- sets a limit for the maximum propagated error) to a small value, or try smaller learning rate.
Also, if you are using unbounded activation functions, try to set few of the layers to bell shaped activation function. The bell functions helps to stabilize the network.
