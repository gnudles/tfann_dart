import 'dart:convert';
import 'dart:isolate';
import 'dart:math';

import 'package:test/test.dart';
import 'package:tfann/tfann.dart';
import 'package:tfann/src/activation_function.dart';

import 'dart:typed_data';

void main() {
  group('TfannNetwork', () {
    final Random r = Random();
    final bitwiseNN = TfannNetwork.full([
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      4,
      4
    ], [
      ActivationFunctionType.logistic,
      ActivationFunctionType.uscls,
      ActivationFunctionType.uacsls,
      ActivationFunctionType.fastBell,
      ActivationFunctionType.divlineSigmoid,
      ActivationFunctionType.abs,
      ActivationFunctionType.bell,
      ActivationFunctionType.tanh,
      ActivationFunctionType.uscsls,
      ActivationFunctionType.funnyHat,
      ActivationFunctionType.cubicSigmoid,
      ActivationFunctionType.line
    ]);
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
        bitwiseNN.train(data, learningRate: 0.001);
      });
    }
    setUp(() {});

    test('code generation', () async {
      var sourceCode = compileNetwork(bitwiseNN, functionName: 'bitwiseEval');
      print(sourceCode);
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
        expect(results[i].any((element) => element.isNaN), isFalse);
        expect(results[i].every((element) => element.isFinite), isTrue);
        expect(
            results[i].toString() ==
                bitwiseNN
                    .feedForward(bitwiseTrainSets[i].input)
                    .toList()
                    .toString(),
            isTrue);
      }
    });

    test('derivatives & funcs', () {
      var xs = List.generate(256, (i) => -10 + 20 * i / 256);
      Float32x4 simdResult;
      void compareSimdAndNonSimd(
          Float32x4 Function(Float32x4) simd, double Function(double) nonSimd) {
        for (int i = 0; i < 256; i += 4) {
          simdResult = simd(Float32x4(xs[i], xs[i + 1], xs[i + 2], xs[i + 3]));
          expect((nonSimd(xs[i]) - simdResult.x).abs() < 0.00001, isTrue);
          expect((nonSimd(xs[i + 1]) - simdResult.y).abs() < 0.00001, isTrue);
          expect((nonSimd(xs[i + 2]) - simdResult.z).abs() < 0.00001, isTrue);
          expect((nonSimd(xs[i + 3]) - simdResult.w).abs() < 0.00001, isTrue);
        }
      }

      mapActivationFunction.values.forEach((actFunc) {
        if (actFunc.derivativeSIMD != null) {
          compareSimdAndNonSimd(actFunc.derivativeSIMD!, actFunc.derivative);
        }
        if (actFunc.funcSIMD != null) {
          compareSimdAndNonSimd(actFunc.funcSIMD!, actFunc.func);
        }
        for (int i = 0; i < 256; i++) {
          var compDeriv =
              (actFunc.func(xs[i] + 1 / 8192) - actFunc.func(xs[i])) * 8192;
          var actualDeriv = actFunc.derivative(xs[i] + 1 / 16384);
          expect((actualDeriv - compDeriv).abs() < 0.0001, isTrue,
              reason:
                  '${actFunc.type.toString()}: at ${xs[i]}: computed: $compDeriv, expected: $actualDeriv');
        }
      });
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
    test('slice & join', () async {
      FVector j = FVector.join([a, b]);
      expect(j[0] == -3.0, isTrue);
      expect(j[1] == -2.0, isTrue);
      expect(j[2] == -1.0, isTrue);
      expect(j[3] == 3.0, isTrue);
      expect(j[4] == 2.0, isTrue);
      expect(j[5] == 1.0, isTrue);
      FVector s = j.slice(2, 3);
      expect(s[0] == -1.0, isTrue);
      expect(s[1] == 3.0, isTrue);
      expect(s[2] == 2.0, isTrue);
      expect(j.equalJumps(3)[0][0] == -3, isTrue);
      expect(j.equalJumps(3)[0][1] == 3, isTrue);
      expect(j.equalJumps(3)[1][0] == -2, isTrue);
      expect(j.equalJumps(3)[1][1] == 2, isTrue);

      expect(j.equalJumps(2)[0][0] == -3, isTrue);
      expect(j.equalJumps(2)[0][1] == -1, isTrue);
      expect(j.equalJumps(2)[0][2] == 2, isTrue);
    });
    test('test matrix', () async {
      final FLeftMatrix leftMatrix = FLeftMatrix.fromList([
        [1, 2, 3],
        [2, 3, 4]
      ]);
      final FVector vector = FVector.fromList([1, 2, 3]);
      expect(
          (leftMatrix.multiplyVector(vector))
              .equals(FVector.fromList([14, 20])),
          isTrue);
    });
  });
}
