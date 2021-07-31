import 'dart:convert';
import 'dart:math';



import 'package:test/test.dart';
import 'package:tfann/tfann.dart';

import 'dart:typed_data';




double lelu(double x) {  if (x > 4) return 1 + 0.25 * x;  if (x > -2) return 0.5 * x;  return 0.0625 * x - 0.875; }
final List<Float32x4List> Lweight_tfann_evaluate_0 = [Uint32List.fromList([1074595097, 1074594525, 1074548888, 0, 0, 0]).buffer.asFloat32x4List(), Uint32List.fromList([1066487571, 1066486951, 1066408945, 0, 0, 0]).buffer.asFloat32x4List(), Uint32List.fromList([3203043203, 3203061381, 3204571980, 0, 0, 0]).buffer.asFloat32x4List(), Uint32List.fromList([1077877061, 1077877265, 1077877915, 0, 0, 0]).buffer.asFloat32x4List()];
final Float32x4List Lbias_tfann_evaluate_0 = Uint32List.fromList([3223715385, 1068675839, 3205715206, 1066062574, 0, 0, 0]).buffer.asFloat32x4List();
final List<Float32x4List> Lweight_tfann_evaluate_1 = [Uint32List.fromList([1076498796, 1081899750, 3210878056, 3229217574, 0, 0, 0]).buffer.asFloat32x4List(), Uint32List.fromList([1074367905, 3193880649, 3216376642, 3209567584, 0, 0, 0]).buffer.asFloat32x4List(), Uint32List.fromList([3211786681, 1082026308, 3206642288, 3215419462, 0, 0, 0]).buffer.asFloat32x4List(), Uint32List.fromList([3227791941, 3160304183, 1060213995, 1076696127, 0, 0, 0]).buffer.asFloat32x4List()];
final Float32x4List Lbias_tfann_evaluate_1 = Uint32List.fromList([1062699993, 3202167228, 3211185041, 3228674663, 0, 0, 0]).buffer.asFloat32x4List();


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
    outputTensor[r] = sum.z + sum.y + sum.x ;
  }
  currentTensor = outputTensor.buffer.asFloat32x4List();
    currentTensor[0]+=Lbias_tfann_evaluate_0[0];
  for (int i = 0; i < 4; ++i)
    outputTensor[i]=lelu(outputTensor[i]);
  outputTensor = Float32List(4);
  for (int r = 0; r < 4; ++r)
  {
    Float32x4List weightRow = Lweight_tfann_evaluate_1[r];
    Float32x4 sum = currentTensor[0]*weightRow[0];
    outputTensor[r] = sum.z + sum.y + sum.x + sum.w;
  }
  currentTensor = outputTensor.buffer.asFloat32x4List();
    currentTensor[0]+=Lbias_tfann_evaluate_1[0];
  for (int i = 0; i < 4; ++i)
    outputTensor[i]=lelu(outputTensor[i]);
  return currentTensor.buffer.asFloat32List(0,4).toList();
}









void main() {
  test('test xor', () async {
    final xor_net = TfannNetwork.full([3, 4, 4], activation: ActivationFunctionType.lelu);
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
        xor_net.train(data, learningRate: 0.0001);
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
