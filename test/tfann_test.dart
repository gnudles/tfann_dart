import 'dart:convert';
import 'dart:math';



import 'package:test/test.dart';
import 'package:tfann/tfann.dart';

import 'dart:typed_data';


double gelu(double x) {      return 0.5*x*(1+tanh(0.7978845608028653558798921198687*(x+0.044715*x*x*x)));}


final List<Float32x4List> Lweight_tfann_evaluate_0 = [Int64List.fromList([-4684549482552772993, 3205952384]).buffer.asFloat32x4List(), Int64List.fromList([-4658463640062052053, 1067246068]).buffer.asFloat32x4List(), Int64List.fromList([4559296036336047244, 1060296640]).buffer.asFloat32x4List(), Int64List.fromList([4559652263048579599, 3214348148]).buffer.asFloat32x4List(), Int64List.fromList([4584575986206599728, 1068362496]).buffer.asFloat32x4List()];
final Float32x4List Lbias_tfann_evaluate_0 = Int64List.fromList([4528448722206921976, -4714918008769206507, 3215666407, 0]).buffer.asFloat32x4List();
final List<Float32x4List> Lweight_tfann_evaluate_1 = [Int64List.fromList([-4691824521505253971, -4665807858037931927, 3218341393, 0]).buffer.asFloat32x4List(), Int64List.fromList([-4629209937054397183, -4682969759229455083, 3189351279, 0]).buffer.asFloat32x4List(), Int64List.fromList([4406611430715703517, 4434756430857720869, 3199162943, 0]).buffer.asFloat32x4List(), Int64List.fromList([4546870734602276260, 4576005591552021506, 1051710346, 0]).buffer.asFloat32x4List()];
final Float32x4List Lbias_tfann_evaluate_1 = Int64List.fromList([-4605361703785376108, -4673147291425281497]).buffer.asFloat32x4List();


List<double> tfann_evaluate(List<double> inData) 
{
  assert(inData.length == 3);
  Float32List input = Float32List(4);
  for (int i = 0; i< 3; ++i) input[i] = inData[i];
  Float32x4List currentTensor = input.buffer.asFloat32x4List();
  Float32List outputTensor;
  outputTensor = Float32List(8);
  for (int r = 0; r < 5; ++r)
  {
    Float32x4List weightRow = Lweight_tfann_evaluate_0[r];
    Float32x4 sum = currentTensor[0]*weightRow[0];
    outputTensor[r] = sum.z + sum.y + sum.x ;
  }
  currentTensor = outputTensor.buffer.asFloat32x4List();
  for (int i = 0; i < 2; ++i)
    currentTensor[i]+=Lbias_tfann_evaluate_0[i];
  for (int i = 0; i < 5; ++i)
    outputTensor[i]=gelu(outputTensor[i]);
  outputTensor = Float32List(4);
  for (int r = 0; r < 4; ++r)
  {
    Float32x4List weightRow = Lweight_tfann_evaluate_1[r];
    Float32x4 sum = Float32x4.zero();
    for (int i = 0; i < 2; ++i)
    {     sum+=currentTensor[i]*weightRow[i];   }
    outputTensor[r] = sum.z + sum.y + sum.x + sum.w;
  }
  currentTensor = outputTensor.buffer.asFloat32x4List();
    currentTensor[0]+=Lbias_tfann_evaluate_1[0];
  for (int i = 0; i < 4; ++i)
    outputTensor[i]=gelu(outputTensor[i]);
  return currentTensor.buffer.asFloat32List(0,4).toList();
}

void main() {
  test('test xor', () async {
    final xor_net = TfannNetwork.full([3, 5, 4], activation: ActivationFunctionType.gelu);
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
    List<TrainData> xor_data = xor_data0;
    xor_data.forEach((data) => print(
        "in: ${data.input.toList()} out: ${xor_net.feedForward(data.input).toList()} expected: ${data.output!.toList()}"));
    print("training...");
    for (int i = 0; i < 5000; ++i) {
      xor_data.forEach((data) {
        xor_net.train(data, learningRate: 0.04);
      });
    }


    xor_data.forEach((data) => print(
        "in: ${data.input.toList()} out: ${xor_net.feedForward(data.input).toList()} expected: ${data.output!.toList()}"));
    print("errors:");
    print(xor_data.map((e) => xor_net.calculateMeanAbsoluteError(e)));
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
