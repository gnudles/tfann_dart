import 'dart:convert';
import 'dart:math';

import 'package:test/test.dart';
import 'package:tfann/tfann.dart';

import 'dart:typed_data';

double slu(double x) {
  x += 0.45353;
  if (x > 4) return 1 + 0.25 * x;
  if (x > -2) {
    var x2 = x * x;
    var x3 = x2 * x;
    return (-11 / 576) * x3 + (7 / 96) * x2 + (7 / 12) * x - 5 / 18;
  }
  return 0.0625 * x - 0.875;
}

final List<Float32x4List> Lweight_tfann_evaluate_0 = [
  Uint32List.fromList([3220968747, 3220961910, 3220978530, 0, 0, 0])
      .buffer
      .asFloat32x4List(),
  Uint32List.fromList([1072519158, 1072508820, 1072534122, 0, 0, 0])
      .buffer
      .asFloat32x4List(),
  Uint32List.fromList([3216374396, 3216369714, 3216380720, 0, 0, 0])
      .buffer
      .asFloat32x4List()
];
final Float32x4List Lbias_tfann_evaluate_0 =
    Uint32List.fromList([1068886605, 1082671270, 1082579327, 0, 0, 0])
        .buffer
        .asFloat32x4List();
final List<Float32x4List> Lweight_tfann_evaluate_1 = [
  Uint32List.fromList([1082116529, 1075147007, 3225379509, 0, 0, 0])
      .buffer
      .asFloat32x4List(),
  Uint32List.fromList([1067770714, 1060886464, 3223049707, 0, 0, 0])
      .buffer
      .asFloat32x4List(),
  Uint32List.fromList([1060853793, 1073984487, 1060793335, 0, 0, 0])
      .buffer
      .asFloat32x4List(),
  Uint32List.fromList([3229668114, 3217295928, 1078046015, 0, 0, 0])
      .buffer
      .asFloat32x4List()
];
final Float32x4List Lbias_tfann_evaluate_1 = Uint32List.fromList(
        [3222846824, 1048548062, 3231611048, 3213094047, 0, 0, 0])
    .buffer
    .asFloat32x4List();

List<double> tfann_evaluate(List<double> inData) {
  assert(inData.length == 3);
  Float32List input = Float32List(4);
  for (int i = 0; i < 3; ++i) input[i] = inData[i];
  Float32x4List currentTensor = input.buffer.asFloat32x4List();
  Float32List outputTensor;
  outputTensor = Float32List(4);
  for (int r = 0; r < 3; ++r) {
    Float32x4List weightRow = Lweight_tfann_evaluate_0[r];
    Float32x4 sum = currentTensor[0] * weightRow[0];
    outputTensor[r] = sum.z + sum.y + sum.x;
  }
  currentTensor = outputTensor.buffer.asFloat32x4List();
  currentTensor[0] += Lbias_tfann_evaluate_0[0];
  for (int i = 0; i < 3; ++i) outputTensor[i] = slu(outputTensor[i]);
  outputTensor = Float32List(4);
  for (int r = 0; r < 4; ++r) {
    Float32x4List weightRow = Lweight_tfann_evaluate_1[r];
    Float32x4 sum = currentTensor[0] * weightRow[0];
    outputTensor[r] = sum.z + sum.y + sum.x;
  }
  currentTensor = outputTensor.buffer.asFloat32x4List();
  currentTensor[0] += Lbias_tfann_evaluate_1[0];
  for (int i = 0; i < 4; ++i) outputTensor[i] = slu(outputTensor[i]);
  return currentTensor.buffer.asFloat32List(0, 4).toList();
}

void main() {
  test('test xor', () async {
    final xor_net =
        TfannNetwork.full([3, 5, 4], activation: ActivationFunctionType.slu);
    
    //xor_net.layers[0].activationFunc = activationBell;
    List<TrainData> xor_data_m = [
      /*  output: column  1 - XOR of 3 bits, column  2 - AND of 3 bits,
       column  3 - OR of 3 bits, column  4 - if exactly two bits ON,
      */

      TrainData.lists([1, 1, -1], [-1, -1, 1, 1]),
      TrainData.lists([1, -1, -1], [1, -1, 1, -1]),
      TrainData.lists([-1, 1, -1], [1, -1, 1, -1]),
      TrainData.lists([-1, -1, -1], [-1, -1, -1, -1]),
      TrainData.lists([-1, -1, 1], [1, -1, 1, -1]),
      TrainData.lists([1, 1, 1], [1, 1, 1, -1]),
      TrainData.lists([1, -1, 1], [-1, -1, 1, 1]),
      TrainData.lists([-1, 1, 1], [-1, -1, 1, 1]),
    ];
    List<TrainData> xor_data0 = [
      /*  output: column  1 - XOR of 3 bits, column  2 - AND of 3 bits,
       column  3 - OR of 3 bits, column  4 - if exactly two bits ON,
      */

      TrainData.lists([1, 1, -1], [0, 0, 1, 1]),
      TrainData.lists([1, -1, -1], [1, 0, 1, 0]),
      TrainData.lists([-1, 1, -1], [1, 0, 1, 0]),
      TrainData.lists([-1, -1, -1], [0, 0, 0, 0]),
      TrainData.lists([-1, -1, 1], [1, 0, 1, 0]),
      TrainData.lists([1, 1, 1], [1, 1, 1, 0]),
      TrainData.lists([1, -1, 1], [0, 0, 1, 1]),
      TrainData.lists([-1, 1, 1], [0, 0, 1, 1]),
    ];
    List<TrainData> xor_data = xor_data_m;
    xor_data.forEach((data) => print(
        "in: ${data.input.toList()} out: ${xor_net.feedForward(data.input).toList()} expected: ${data.output!.toList()}"));
    print("training...");
    for (int i = 0; i < 4000; ++i) {
      xor_data.forEach((data) {
        xor_net.train(data, learningRate: 0.03);
      });
    }
    for (int i = 0; i < 1000; ++i) {
      xor_data.forEach((data) {
        xor_net.train(data, learningRate: 0.002);
      });
    }
    for (int i = 0; i < 1000; ++i) {
      xor_data.forEach((data) {
        xor_net.train(data, learningRate: 0.0002);
      });
    }

    xor_data.forEach((data) => print(
        "in: ${data.input.toList()} out: ${xor_net.feedForward(data.input).toList()} expected: ${data.output!.toList()}"));
    print("errors:");
    print(xor_data.map((e) => xor_net.calculateMeanAbsoluteError(e)).toList());
    await xor_net.save("binary.net");

    var new_net = TfannNetwork.fromFile("binary.net")!;
    print("after saving and loading:");
    xor_data.forEach((data) => print(
        "in: ${data.input.toList()} out: ${new_net.feedForward(data.input).toList()} expected: ${data.output!.toList()}"));
    print("evaluation:");
    xor_data.forEach((data) => print(
        "in: ${data.input.toList()} out: ${tfann_evaluate(data.input.toList()).toList()} expected: ${data.output!.toList()}"));

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
