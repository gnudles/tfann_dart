import 'dart:convert';

import 'package:tfann/src/linalg.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:tfann/tfann.dart';


void main() {
  test('test xor', () async {
    final xor_net =
        TfannNetwork.full([3, 2, 4], activation: activationLogisticSigmoid);
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
    xor_data.forEach((data) => print(
        "in: ${data.input.toList()} out: ${xor_net.feedForward(data.input).toList()} expected: ${data.output.toList()}"));
    print("training...");
    for (int i = 0; i < 7000; ++i) {
      xor_data.forEach((data) {
        xor_net.train(data, learningRate: 0.06);
      });
    }
    xor_data.forEach((data) => print(
        "in: ${data.input.toList()} out: ${xor_net.feedForward(data.input).toList()} expected: ${data.output.toList()}"));
    await xor_net.save("binary.net");

    var new_net = TfannNetwork.fromFile("binary.net")!;
    xor_data.forEach((data) => print(
        "in: ${data.input.toList()} out: ${new_net.feedForward(data.input).toList()} expected: ${data.output.toList()}"));


    print(new_net.compile());
  });

  test('test matrix', () async {
    final FLeftMatrix leftMatrix = FLeftMatrix.fromList([
      [1, 2, 3],
      [2, 3, 4]
    ]);
    final FVector vector = FVector.fromList([1, 2, 3]);

    print(jsonEncode(leftMatrix.toJson()));
    print(jsonEncode(vector.toJson()));
    print(jsonEncode((leftMatrix.multiplyVector(vector)).toJson()));
  });
}
