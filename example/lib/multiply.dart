import 'dart:convert';
import 'dart:math';

import 'package:tfann/tfann.dart';

Random r = Random();

void netTrainMultiply(TfannNetwork net, int rounds, double learningRate) {
  for (int i = 0; i < rounds; ++i) {
    var x = r.nextDouble() * 2 - 1;
    var y = r.nextDouble() * 2 - 1;
    net.train(TrainSetInputOutput.lists([x, y], [x * y]),
        learningRate: learningRate);
  }
}

void main() {
  {
    final multiply_net = TfannNetwork.full([
      2,
      30,
      4,
      1
    ], [
      ActivationFunctionType.funnyHat,
      ActivationFunctionType.uscsls,
      ActivationFunctionType.divlineSigmoid
    ]);
    netTrainMultiply(multiply_net, 40000, 0.3);
    netTrainMultiply(multiply_net, 300000, 0.001);
    netTrainMultiply(multiply_net, 300000, 0.00001);

    for (int i = 0; i < 200; ++i) {
      var x = r.nextDouble() * 2 - 1;
      var y = r.nextDouble() * 2 - 1;
      if (x.abs() < 0.1)
        x = x.sign *
            (r.nextDouble() * 0.9 +
                0.1); //skip really small values, cause nn's doesn't like 'em
      if (y.abs() < 0.1) y = y.sign * (r.nextDouble() * 0.9 + 0.1);
      var result = multiply_net.feedForward(FVector.fromList([x, y])).single;
      var expected = x * y;
      print(
          "x= $x, y= $y, got: $result, expected: $expected : error= ${((expected - result) * 100 / (expected)).abs().truncate()}%");
    }
  }
}
