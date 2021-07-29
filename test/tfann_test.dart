import 'dart:convert';
import 'dart:math';

import 'package:flutter_test/flutter_test.dart';

import 'package:tfann/tfann.dart';


import 'dart:typed_data';


double logisticSigmoid(double x) { return 2 / (1 + exp(-x)) - 1;}
double absSigmoid(double x) { return x / (1 + x.abs());}
double tanh(double x) {  var e2x = exp(2 * x);    return (e2x - 1) / (e2x + 1); }
double bell(double x) {      return exp(-0.5*x*x);}
double gelu(double x) {      return 0.5*x*(1+tanh(0.7978845608028653558798921198687*(x+0.044715*x*x*x)));}

final List<Float32x4List> Lweight0 = [Int64List.fromList([-4646641295893529096, 3172877232]).buffer.asFloat32x4List(), Int64List.fromList([-4673762236250713944, 1067269284]).buffer.asFloat32x4List(), Int64List.fromList([-4641594773778789480, 3214371336]).buffer.asFloat32x4List(), Int64List.fromList([4552613333509948838, 1060709916]).buffer.asFloat32x4List(), Int64List.fromList([4552999294302083874, 3213159812]).buffer.asFloat32x4List()];
final Float32x4List Lbias0 = Int64List.fromList([-4634343050039762903, -4659082080990716510, 3212709118, 0]).buffer.asFloat32x4List();
final List<Float32x4List> Lweight1 = [Int64List.fromList([-4632948360339278083, -4650718106777998393, 3207958840, 0]).buffer.asFloat32x4List(), Int64List.fromList([4436658467828238390, -4675627400956649014, 1034688105, 0]).buffer.asFloat32x4List(), Int64List.fromList([-4679483673863445349, -4743187322439184805, 1027282343, 0]).buffer.asFloat32x4List(), Int64List.fromList([4564901376650663194, 4560493110259876338, 1061242266, 0]).buffer.asFloat32x4List()];
final Float32x4List Lbias1 = Int64List.fromList([4566627642148215372, -4783000683564538653]).buffer.asFloat32x4List();
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
    Float32x4List weightRow = Lweight0[r];
    Float32x4 sum = currentTensor[0]*weightRow[0];
    outputTensor[r] = sum.z + sum.y + sum.x ;
  }
  currentTensor = outputTensor.buffer.asFloat32x4List();
  for (int i = 0; i < 2; ++i)
    currentTensor[i]+=Lbias0[i];
  for (int i = 0; i < 5; ++i)
    outputTensor[i]=gelu(outputTensor[i]);
  outputTensor = Float32List(4);
  for (int r = 0; r < 4; ++r)
  {
    Float32x4List weightRow = Lweight1[r];
    Float32x4 sum = Float32x4.zero();
    for (int i = 0; i < 2; ++i)
    {     sum+=currentTensor[i]*weightRow[i];   }
    outputTensor[r] = sum.z + sum.y + sum.x + sum.w;
  }
  currentTensor = outputTensor.buffer.asFloat32x4List();
    currentTensor[0]+=Lbias1[0];
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
