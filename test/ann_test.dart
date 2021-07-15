import 'dart:convert';

import 'package:ann/src/linalg.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:ann/ann.dart';
import 'package:ml_linalg/vector.dart';

void main() {
  test('test xor', () async {
    final xor_net =
        Network.full([2, 3, 3], activation: activationLogisticSigmoid);
    List<TrainData> xor_data = [
      TrainData.lists([-1, -1], [-1, 1, -1]),
      TrainData.lists([1, 1], [-1, -1, 1]),
      TrainData.lists([1, -1], [1, -1, 1]),
      TrainData.lists([-1, 1], [1, -1, 1]),
    ];
    xor_data.forEach((data) => print(
        "in: ${data.input.toList()} out: ${xor_net.feedForward(data.input).toList()} expected: ${data.output.toList()}"));
    print("training...");
    for (int i = 0; i < 5000; ++i) {
      xor_data.forEach((data) {
        xor_net.train(data, learningRate: 0.06);
      });
    }
    xor_data.forEach((data) => print(
        "in: ${data.input.toList()} out: ${xor_net.feedForward(data.input).toList()} expected: ${data.output.toList()}"));
    await xor_net.save("binary.net");

    var new_net = Network.fromFile("binary.net").value;
    xor_data.forEach((data) => print(
        "in: ${data.input.toList()} out: ${new_net.feedForward(data.input).toList()} expected: ${data.output.toList()}"));
  });

  test('test matrix', () async {
    final FLeftMatrix leftMatrix = FLeftMatrix.fromList([
      [1, 2 ,3],
      [2 ,3, 4]
    ]);
    final FVector vector = FVector.fromList(
      [1, 2 ,3]
    );
    
    print(jsonEncode(leftMatrix.toJson()));
    print(jsonEncode(vector.toJson()));
    print(jsonEncode((leftMatrix*vector).toJson()));
  });
}
