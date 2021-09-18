import 'dart:convert';
import 'dart:math';

import 'package:tfann/tfann.dart';

final Random _r = Random();
void netTrainSine(TfannNetwork net, int rounds, double learningRate) {
  for (int i = 0; i < rounds; ++i) {
    var x = _r.nextDouble() *  6.29 - 3.15;
      net.train(TrainSetInputOutput.lists([x], [sin(x)]),
        learningRate: learningRate,maxErrClipAbove: 0.0001,skipIfErrBelow: 0.00001);
  }
}
void main() {
  {
    
    final sine_net = TfannNetwork.full([1, 16, 8, 8, 8, 1],
        [ActivationFunctionType.funnyHat, ActivationFunctionType.fastBell,
        ActivationFunctionType.fastBell,

        
        ActivationFunctionType.funnyHat, ActivationFunctionType.line]);
        netTrainSine(sine_net,800000,1.01);
    netTrainSine(sine_net,80000,0.8);
    netTrainSine(sine_net,40000,0.7);
    netTrainSine(sine_net,200000,0.2);
    var x = -3.14;
    for (int i = 0; i < 1000; ++i) {
      
      x +=  6.29 / 1000;
      print(
          "x= $x : error= ${sine_net.calculateMeanAbsoluteError(
            [TrainSetInputOutput.lists([x], [sin(x)])]).single}");
    }
  }
}
