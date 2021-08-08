import 'dart:convert';
import 'dart:math';

import 'package:tfann/tfann.dart';

void main() {
  {
    Random r = Random();
    final sine_net = TfannNetwork.full([1, 32, 16, 1],
        [ActivationFunctionType.fastBell, ActivationFunctionType.uscls, ActivationFunctionType.uscsls]);
    for (int i = 0; i < 400000; ++i) {
      var x = r.nextDouble() * 6.29 - 3.145;
      sine_net.train(TrainSetInputOutput.lists([x], [sin(x)]),
          learningRate: 0.02);
    }
    var x = -3.145;
    for (int i = 0; i < 1000; ++i) {
      
      x += 6.29 / 1000;
      print(
          "x= $x : error= ${sine_net.calculateMeanAbsoluteError(
            [TrainSetInputOutput.lists([x], [sin(x)])]).single}");
    }
  }
}
