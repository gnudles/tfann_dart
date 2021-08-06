import 'dart:convert';
import 'dart:isolate';
import 'dart:math';

import 'package:test/test.dart';
import 'package:tfann/tfann.dart';

import 'dart:typed_data';

void main() {
  group('TfannNetwork', () {
    final Random r = Random();
    final bitwiseNN =
        TfannNetwork.full([3, 3, 4], activation: ActivationFunctionType.lelq);
    List<TrainSetInputOutput> bitwiseTrainSets = [
      /*  output: column  1 - XOR of 3 bits, column  2 - AND of 3 bits,
       column  3 - OR of 3 bits, column  4 - if exactly two bits ON,
      */

      TrainSetInputOutput.lists([-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]),
      TrainSetInputOutput.lists([-1.0, -1.0, 1.0], [1.0, -1.0, 1.0, -1.0]),
      TrainSetInputOutput.lists([-1.0, 1.0, -1.0], [1.0, -1.0, 1.0, -1.0]),
      TrainSetInputOutput.lists([-1.0, 1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]),
      TrainSetInputOutput.lists([1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0]),
      TrainSetInputOutput.lists([1.0, -1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]),
      TrainSetInputOutput.lists([1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]),
      TrainSetInputOutput.lists([1.0, 1.0, 1.0], [1.0, 1.0, 1.0, -1.0]),
    ];
    for (int i = 0; i < 8000; ++i) {
      bitwiseTrainSets.forEach((data) {
        bitwiseNN.train(data, learningRate: 0.01);
      });
    }
    setUp(() {});

    test('code generation', () async {
      var sourceCode = bitwiseNN.compile(functionName: 'bitwiseEval');
      final uri = Uri.dataFromString(
        '''
import "dart:isolate";

$sourceCode

void main(_, SendPort port) {
  port.send([bitwiseEval(${bitwiseTrainSets[0].input.toList()}),
  bitwiseEval(${bitwiseTrainSets[1].input.toList()}),
  bitwiseEval(${bitwiseTrainSets[2].input.toList()}),
  bitwiseEval(${bitwiseTrainSets[3].input.toList()}),
  bitwiseEval(${bitwiseTrainSets[4].input.toList()}),
  bitwiseEval(${bitwiseTrainSets[5].input.toList()}),
  bitwiseEval(${bitwiseTrainSets[6].input.toList()}),
  bitwiseEval(${bitwiseTrainSets[7].input.toList()})]);
}
          ''',
        mimeType: 'application/dart',
      );

      final port = ReceivePort();
      await Isolate.spawnUri(uri, [], port.sendPort);

      List<List<double>> results = await port.first;
      for (int i = 0; i < 8; ++i) {
        expect(
            results[i].toString() ==
                bitwiseNN
                    .feedForward(bitwiseTrainSets[i].input)
                    .toList()
                    .toString(),
            isTrue);
      }
    });

    test('serialization', () async {
      var jsonString = jsonEncode(bitwiseNN.toJson());
      var loadedNN = TfannNetwork.fromJson(jsonDecode(jsonString));
      expect(loadedNN, isNotNull);
      for (int i = 0; i < 8; ++i) {
        expect(
            loadedNN!
                .feedForward(bitwiseTrainSets[i].input)
                .equals(bitwiseNN.feedForward(bitwiseTrainSets[i].input)),
            isTrue);
      }
    });
  });
  group('FVector', () {
    var a = FVector.fromList([-3, -2, -1]);
    var b = FVector.fromList([3, 2, 1]);
    test('smallest', () async {
      expect(a.smallestElement() == -3.0, isTrue);

      expect(b.smallestElement() == 1.0, isTrue);
    });
    test('largest', () async {
      expect(a.largestElement() == -1.0, isTrue);
      expect(b.largestElement() == 3.0, isTrue);
    });
    test('test matrix', () async {
      final FLeftMatrix leftMatrix = FLeftMatrix.fromList([
        [1, 2, 3],
        [2, 3, 4]
      ]);
      final FVector vector = FVector.fromList([1, 2, 3]);
      expect((leftMatrix.multiplyVector(vector)).equals(FVector.fromList([14,20])),isTrue);
    });
  });

  return;
  test('test xor', () async {
    final xor_net =
        TfannNetwork.full([3, 4, 4], activation: ActivationFunctionType.slq);

    //xor_net.layers[0].activationFunc = activationBell;
    List<TrainSetInputOutput> xor_data_m = [
      /*  output: column  1 - XOR of 3 bits, column  2 - AND of 3 bits,
       column  3 - OR of 3 bits, column  4 - if exactly two bits ON,
      */

      TrainSetInputOutput.lists([1, 1, -1], [-1, -1, 1, 1]),
      TrainSetInputOutput.lists([1, -1, -1], [1, -1, 1, -1]),
      TrainSetInputOutput.lists([-1, 1, -1], [1, -1, 1, -1]),
      TrainSetInputOutput.lists([-1, -1, -1], [-1, -1, -1, -1]),
      TrainSetInputOutput.lists([-1, -1, 1], [1, -1, 1, -1]),
      TrainSetInputOutput.lists([1, 1, 1], [1, 1, 1, -1]),
      TrainSetInputOutput.lists([1, -1, 1], [-1, -1, 1, 1]),
      TrainSetInputOutput.lists([-1, 1, 1], [-1, -1, 1, 1]),
    ];
    List<TrainSetInputOutput> xor_data0 = [
      /*  output: column  1 - XOR of 3 bits, column  2 - AND of 3 bits,
       column  3 - OR of 3 bits, column  4 - if exactly two bits ON,
      */

      TrainSetInputOutput.lists([1, 1, -1], [0, 0, 1, 1]),
      TrainSetInputOutput.lists([1, -1, -1], [1, 0, 1, 0]),
      TrainSetInputOutput.lists([-1, 1, -1], [1, 0, 1, 0]),
      TrainSetInputOutput.lists([-1, -1, -1], [0, 0, 0, 0]),
      TrainSetInputOutput.lists([-1, -1, 1], [1, 0, 1, 0]),
      TrainSetInputOutput.lists([1, 1, 1], [1, 1, 1, 0]),
      TrainSetInputOutput.lists([1, -1, 1], [0, 0, 1, 1]),
      TrainSetInputOutput.lists([-1, 1, 1], [0, 0, 1, 1]),
    ];
    List<TrainSetInputOutput> xor_data = xor_data_m;
    xor_data.forEach((data) => print(
        "in: ${data.input.toList()} out: ${xor_net.feedForward(data.input).toList()} expected: ${data.output.toList()}"));
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
        "in: ${data.input.toList()} out: ${xor_net.feedForward(data.input).toList()} expected: ${data.output.toList()}"));
    print("errors:");
    print(xor_data.map((e) => xor_net.calculateMeanAbsoluteError(e)).toList());
    await xor_net.save("binary.net");

    var new_net = TfannNetwork.fromFile("binary.net")!;
    print("after saving and loading:");
    xor_data.forEach((data) => print(
        "in: ${data.input.toList()} out: ${new_net.feedForward(data.input).toList()} expected: ${data.output.toList()}"));

    print(new_net.compile());
  });
}
