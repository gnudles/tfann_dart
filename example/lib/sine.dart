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
