# Examples

## sine.dart

```dart

import 'dart:convert';
import 'dart:math';

import 'package:tfann/tfann.dart';

final Random _r = Random();
void netTrainSine(TfannNetwork net, int rounds, double learningRate) {
  for (int i = 0; i < rounds; ++i) {
    var x = _r.nextDouble() *  6.5 - 3.25;
      net.train(TrainSetInputOutput.lists([x], [sin(x)]),
        learningRate: learningRate,maxErrClipAbove: 0.0001,skipIfErrBelow: 0.00001);
  }
}
void main() {
  {
    
    final sine_net = TfannNetwork.full([1, 64, 16, 16, 1],
        [ActivationFunctionType.funnyHat, ActivationFunctionType.uscsls,
        ActivationFunctionType.uscsls,
        ActivationFunctionType.line]);
        netTrainSine(sine_net,60000,0.8);
    netTrainSine(sine_net,80000,0.6);
    netTrainSine(sine_net,160000,0.3);
    netTrainSine(sine_net,200000,0.1);
    netTrainSine(sine_net,100000,0.05);
    var x = -3.14;
    for (int i = 0; i < 1000; ++i) {
      
      x +=  6.29 / 1000;
      print(
          "x= $x ,   result: ${sine_net.feedForward(FVector.fromList([x])).toList()} , real: ${sin(x)} error= ${sine_net.calculateMeanAbsoluteError(
            [TrainSetInputOutput.lists([x], [sin(x)])]).single}");
    }
  }
}


```

## circle.dart

```dart

import 'dart:convert';
import 'dart:math';

import 'package:tfann/tfann.dart';

void main() {
  {
    Random r = Random();
    final circle_net = TfannNetwork.full([
      2,
      16,
      8,
      8,
      1
    ], [
      ActivationFunctionType.funnyHat,
      ActivationFunctionType.uscsls,
      ActivationFunctionType.uscsls,
      ActivationFunctionType.divlineSigmoid,
    ]);
    var criteria = (sr) => (sr < 1.2 && sr > 0.4) ? 1.0 : -1.0;
    for (int i = 0; i < 2000000; ++i) {
      var x = r.nextDouble() * 4 - 2;
      var y = r.nextDouble() * 4 - 2;
      var sr = sqrt(x * x + y * y);
      circle_net.train(TrainSetInputOutput.lists([x, y], [criteria(sr)]),
          learningRate: 0.01);
    }
    int false_positive = 0;
    int false_negative = 0;
    int positive = 0;
    int negative = 0;
    int testLength = 50000;
    int positive_truth = 0;
    double threshold = 0.1;
    for (int i = 0; i < testLength; ++i) {
      var x = r.nextDouble() * 4 - 2;
      var y = r.nextDouble() * 4 - 2;
      var sr = sqrt(x * x + y * y);
      var netResult = circle_net.feedForward(FVector.fromList([x, y])).single;
      if (criteria(sr) == 1) {
        positive_truth++;
      }
      if (criteria(sr) > 0 && netResult < threshold) {
        false_negative++;
      }
      if (criteria(sr) < 0 && netResult > -threshold) {
        false_positive++;
      }
      if (netResult > threshold) {
        positive++;
      }
      if (netResult < -threshold) {
        negative++;
      }
      //print("$x $y $netResult");
    }
    print("Test ground truth: Positive: ${positive_truth * 100 / testLength}%    Negative: ${100 - positive_truth * 100 / testLength}%");
    print("Result: Positive: ${positive * 100 / testLength}%  Negative: ${negative * 100 / testLength}%");
    print("False Positive: ${false_positive * 100 / testLength}%   False Negative: ${false_negative * 100 / testLength}%");
  }
}


```

## bitwise.dart

```dart

import 'dart:convert';
import 'dart:math';

import 'package:tfann/tfann.dart';

void main() {


  List<TrainSetInputOutput> bw_data = [
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

  final bwise_net =
      TfannNetwork.full([3, 4, 4], [ActivationFunctionType.uscsls, ActivationFunctionType.uscsls]);
  // train network
  // train method takes a single TrainSet and runs it only once.
  for (int i = 0; i < 10000; ++i) {
    bw_data.forEach((data) {
      bwise_net.train(data, learningRate: 0.04);
    });
  }

  print("after training...");

    
  bw_data.forEach((data) => print(
    "in: ${data.input.toList()} out: ${bwise_net.feedForward(data.input).toList()} expected: ${data.output.toList()}"));

  print("generated code:");
  print(compileNetwork(bwise_net));

}




```

## multiply.dart

```dart

import 'dart:convert';
import 'dart:math';

import 'package:tfann/tfann.dart';

final Random _r = Random();

void netTrainMultiply(TfannNetwork net, int rounds, double learningRate) {
  for (int i = 0; i < rounds; ++i) {
    var x = _r.nextDouble() * 2 - 1;
    var y = _r.nextDouble() * 2 - 1;
    net.train(TrainSetInputOutput.lists([x, y], [x * y]),
        learningRate: learningRate);
  }
}

void main() {
  {
    //This is an infamous trick. Prepare to be amazed.
    final multiply_net = TfannNetwork.full([2, 2, 1],
        [ActivationFunctionType.squartered, ActivationFunctionType.line]);
    netTrainMultiply(multiply_net, 40000, 0.1);
    netTrainMultiply(multiply_net, 300000, 0.001);
    netTrainMultiply(multiply_net, 1000000, 0.0001);
    netTrainMultiply(multiply_net, 10000, 0.0000001);

    for (int i = 0; i < 200; ++i) {
      var x = _r.nextDouble() * 2 - 1;
      var y = _r.nextDouble() * 2 - 1;
      if (x.abs() < 0.1)
        x = x.sign *
            (_r.nextDouble() * 0.9 +
                0.1); //skip really small values, cause nn's doesn't like 'em
      if (y.abs() < 0.1) y = y.sign * (_r.nextDouble() * 0.9 + 0.1);
      var result = multiply_net.feedForward(FVector.fromList([x, y])).single;
      var expected = x * y;
      print(
          "x= $x, y= $y, got: $result, expected: $expected : error= ${((expected - result) * 100 / (expected)).abs().truncate()}%");
    }
    print(jsonEncode(multiply_net.toJson()));
  }
}


```
