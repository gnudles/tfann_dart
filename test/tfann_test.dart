import 'dart:convert';
import 'dart:math';

import 'package:test/test.dart';
import 'package:tfann/tfann.dart';

import 'dart:typed_data';

double slu(double x) {  x += 0.45353;  if (x > 4) return 1 + 0.25 * x;  if (x > -2) {    var x2 = x * x;    var x3 = x2 * x;    return (-11/576)*x3+(7/96)*x2+(7/12)*x-5/18;  }  return 0.0625 * x - 0.875;}
Float32x4 sluX4(Float32x4 x) {
  x += Float32x4.splat(0.45353);
  Int32x4 greater4 = x.greaterThan(Float32x4.splat(4));
  Int32x4 greater2 = x.greaterThan(Float32x4.splat(-2));
  Float32x4 x2 = x * x;
  Float32x4 x3 = x2 * x;
  return greater4.select((x.scale(0.25) + Float32x4.splat(1)), greater2.select(x3.scale(-11 / 576)  + x2.scale(7 / 96)  + x.scale(7 / 12)  - Float32x4.splat(5 / 18), x.scale(0.0625)-Float32x4.splat(0.875)));
}
final List<Float32x4List> Lweight_tfann_evaluate_0 = [Uint32List.fromList([1061734424, 1068319737, 3197279448, 0, 0, 0]).buffer.asFloat32x4List(), Uint32List.fromList([3212343254, 3214242561, 3216839713, 0, 0, 0]).buffer.asFloat32x4List(), Uint32List.fromList([3218877343, 3218807543, 3218903785, 0, 0, 0]).buffer.asFloat32x4List(), Uint32List.fromList([3223833271, 3223437569, 3224038821, 0, 0, 0]).buffer.asFloat32x4List(), Uint32List.fromList([1066640933, 3211839774, 1073016495, 0, 0, 0]).buffer.asFloat32x4List()];
final Float32x4List Lbias_tfann_evaluate_0 = Uint32List.fromList([1018821290, 1076714006, 3221889414, 1059659408, 1063215464, 0, 0, 0]).buffer.asFloat32x4List();
final List<Float32x4List> Lweight_tfann_evaluate_1 = [Uint32List.fromList([1067653208, 3219956846, 3223325029, 1078770049, 1061053443, 0, 0, 0]).buffer.asFloat32x4List(), Uint32List.fromList([1067927215, 3218923674, 1050408590, 1065383446, 1060567289, 0, 0, 0]).buffer.asFloat32x4List(), Uint32List.fromList([3182724903, 1038259357, 3222743775, 1058908675, 3172068864, 0, 0, 0]).buffer.asFloat32x4List(), Uint32List.fromList([3216437517, 1072719072, 1039901040, 3223588606, 3210176461, 0, 0, 0]).buffer.asFloat32x4List()];
final Float32x4List Lbias_tfann_evaluate_1 = Uint32List.fromList([3217132268, 3206537276, 3205234870, 3215634523, 0, 0, 0]).buffer.asFloat32x4List();


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
  currentTensor[0]=sluX4(currentTensor[0]);
  outputTensor[4]=slu(outputTensor[4]);
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
  currentTensor[0]=sluX4(currentTensor[0]);
  return currentTensor.buffer.asFloat32List(0,4).toList();
}

void main() {
  test('test xor', () async {
    final xor_net =
        TfannNetwork.full([3, 4, 4], activation: ActivationFunctionType.slq);
    
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
    for (int i = 0; i < 400; ++i) {
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
