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


