import 'dart:convert';
import 'dart:typed_data';

import 'package:tfann/tfann.dart';
import 'package:tfann/src/train_data.dart';

import 'activation_function.dart';
import 'dart:math';
import 'dart:io';

import 'linalg.dart';

class RandomSupply {
  static final Random rng = Random();
  static double nextBias() => rng.nextDouble() - 0.5;
  static double nextWeight() {
    var x = rng.nextDouble() * 0.8 + 0.2;
    return rng.nextBool() ? x : -x;
  }
}

class FeedArtifacts {
  final FVector activation;
  final FVector derivative;
  const FeedArtifacts(this.activation, this.derivative);
}

class TrainArtifacts {
  /// defined as Network Output minus Expected Output.
  final FVector forwardError;
  final FVector propagatedError;
  const TrainArtifacts(this.forwardError, this.propagatedError);
}

class TfannLayer {
  FLeftMatrix weights;
  FVector bias;
  ActivationFunction activationFunc;
  TfannLayer.fromJsonMap(Map json)
      : bias = FVector.fromJson(json["bias"]),
        weights = FLeftMatrix.fromJson(json["weights"]),
        activationFunc = mapActivationFunction[
            activationTypeFromString[json["activation"]]!]!;

  TfannLayer(int inputs, int outputs, ActivationFunctionType activationFuncType)
      : bias = FVector.fromList(
            List.generate(outputs, (index) => RandomSupply.nextBias())),
        weights = FLeftMatrix.fromList(List.generate(outputs,
            (o) => List.generate(inputs, (i) => RandomSupply.nextWeight()))),
        activationFunc = mapActivationFunction[activationFuncType]!;

  FVector feedForward(FVector input) {
    return ((weights.multiplyVector(input)) + bias)
      ..apply(activationFunc.func, activationFunc.funcSIMD);
  }

  FeedArtifacts createFeedArtifacts(FVector input) {
    FVector intermediateVector = (weights.multiplyVector(input)) + bias;
    return FeedArtifacts(
        intermediateVector.applied(
            activationFunc.func, activationFunc.funcSIMD),
        intermediateVector.applied(
            activationFunc.derivative, activationFunc.derivativeSIMD));
  }

  Map<String, dynamic> toJson() {
    return {
      "weights": weights.toJson(),
      "bias": bias.toJson(),
      "activation": activationFunc.type.toString()
    };
  }
}

/// A structure of a complete fully-connected network.
class TfannNetwork {
  List<TfannLayer> layers = [];
  TfannNetwork(this.layers);
  TfannNetwork.full(List<int> layersDefinition,
      {ActivationFunctionType activation = ActivationFunctionType.slq}) {
    int nextInputs = layersDefinition[0];

    layersDefinition.skip(1).forEach((outputs) {
      layers.add(TfannLayer(nextInputs, outputs, activation));
      nextInputs = outputs;
    });
  }

  /// Get the network's output vector from the input vector.
  FVector feedForward(FVector input) {
    return layers.fold(input, (vec, layer) => layer.feedForward(vec));
  }

  /// Train network with a single training pair, for a single epoch.
  ///
  /// returns the propagated error of the first layer, which is good for chained networks.
  TrainArtifacts train(TrainSet trainSet,
      {double learningRate = 0.04, double maxError = 0.0}) {
    FVector nextInputs = trainSet.input;
    List<FeedArtifacts> artifacts = [
      FeedArtifacts(nextInputs, FVector.zero(nextInputs.length))
    ];
    for (TfannLayer l in layers) {
      artifacts.add(l.createFeedArtifacts(nextInputs));
      nextInputs = artifacts.last.activation;
    }
    FVector netOutput = nextInputs;
    FVector netErrors;
    if (trainSet is TrainSetInputOutput) {
      netErrors = netOutput - (trainSet as TrainSetInputOutput).output;
    } else
      netErrors = (trainSet as TrainSetInputError).error;
    FVector normalizedErrors = netErrors;
    if (maxError > 0.0) {
      double norm = normalizedErrors.abs().largestElement();
      //double norm = normalizedErrors.squared().sumElements();
      if (norm > maxError) {
        normalizedErrors = netErrors.scaled(maxError / norm);
      }
    }

    FVector previousDelta = normalizedErrors;
    List<FVector> layerDelta = [];
    for (int i = layers.length - 1; i >= 0; --i) {
      FVector currentDelta = (artifacts[i + 1].derivative * previousDelta);
      layerDelta.add(currentDelta);

      previousDelta =
          (layers[i].weights.transposed().multiplyVector(currentDelta));
    }

    var arti = artifacts.iterator;
    for (TfannLayer l in layers) {
      l.bias -= layerDelta.last.scaled(learningRate);
      arti.moveNext();

      l.weights -= layerDelta.last.multiplyTransposed(arti.current.activation)
        ..scale(learningRate);
      layerDelta.removeLast();
    }
    return TrainArtifacts(netErrors, previousDelta);
  }

  /// Given a single training pair, calculate the network's mean-square-error.
  double calculateMeanSquareError(TrainSetInputOutput data) {
    return (feedForward(data.input) - data.output).squared().sumElements() /
        data.output.nRows;
  }

  /// Given a single training pair, calculate the network's root-mean-square-error.
  double calculateRootMeanSquareError(TrainSetInputOutput data) {
    return sqrt(calculateMeanSquareError(data));
  }

  /// Given a single training pair, calculate the network's mean-absolute-error.
  double calculateMeanAbsoluteError(TrainSetInputOutput data) {
    return (feedForward(data.input) - data.output).abs().sumElements() /
        data.output.nRows;
  }

  /// Returns Json object.
  ///
  /// To convert it to Json string, use 'jsonEncode'
  Object toJson() {
    return layers.map((l) => l.toJson()).toList();
  }

  /// Saves the network to file.
  Future<void> save(String filename) async {
    var sink = File(filename).openWrite(mode: FileMode.writeOnly);
    sink.write(jsonEncode(toJson()));

    // Close the IOSink to free system resources.
    await sink.flush();
    sink.close();
  }

  static TfannNetwork? fromFile(String filename) {
    String jsonString = File(filename).readAsStringSync();
    return fromJson(jsonDecode(jsonString));
  }

  /// Creates TfannNetwork from Json object.
  ///
  /// Use 'jsonDecode' to convert a Json into Object.
  static TfannNetwork? fromJson(dynamic jsonObject) {
    assert(jsonObject is List);
    assert((jsonObject as List).isNotEmpty);
    if (jsonObject is List) {
      assert(jsonObject.every((element) => element is Map));
      return TfannNetwork(
          jsonObject.map((e) => TfannLayer.fromJsonMap(e)).toList());
    }
    return null;
  }

  /// Returns a pure dart code that represents the function of this network.
  String compile({String functionName = 'tfann_evaluate'}) {
    StringBuffer stringBuffer = StringBuffer();
    int inputSize = layers[0].weights.nColumns;
    var activationsSet = layers.map((e) => e.activationFunc.type).toSet();

    stringBuffer.write("import 'dart:typed_data';\n");
    stringBuffer.write("import 'dart:math';\n\n");
    if (activationsSet.contains(ActivationFunctionType.logistic))
      stringBuffer.write(
          "double logisticSigmoid(double x) { return 2 / (1 + exp(-x)) - 1;}\n");
    if (activationsSet.contains(ActivationFunctionType.abs)) {
      stringBuffer
          .write("double absSigmoid(double x) { return x / (1 + x.abs());}\n");
      stringBuffer.write("final Float32x4 onesX4 = Float32x4.splat(1);\n");
      stringBuffer.write(
          "Float32x4 absSigmoidX4(Float32x4 x) =>   x / (onesX4 + x.abs());\n");
    }
    if (activationsSet.contains(ActivationFunctionType.tanh))
      stringBuffer.write(
          "double tanh(double x) {  var e2x = exp(2 * x);    return (e2x - 1) / (e2x + 1); }\n");
    if (activationsSet.contains(ActivationFunctionType.bell))
      stringBuffer
          .write("double bell(double x) {      return exp(-0.5*x*x);}\n");
    if (activationsSet.contains(ActivationFunctionType.gelu))
      stringBuffer.write(
          "double gelu(double x) {      return 0.5*x*(1+tanh(0.7978845608028653558798921198687*(x+0.044715*x*x*x)));}\n");
    if (activationsSet.contains(ActivationFunctionType.lelq))
      stringBuffer.write(
          "double lelq(double x) {  if (x > 4) return 1 + 0.25 * x;  if (x > -2) return 0.5 * x;  return 0.0625 * x - 0.875; }\n");
    if (activationsSet.contains(ActivationFunctionType.slq)) {
      stringBuffer.write(
          "double slq(double x) {  x += 0.45353;  if (x > 4) return 1 + 0.25 * x;  if (x > -2) {    var x2 = x * x;    var x3 = x2 * x;    return (-11/576)*x3+(7/96)*x2+(7/12)*x-5/18;  }  return 0.0625 * x - 0.875;}\n");
      stringBuffer.write('''Float32x4 slqX4(Float32x4 x) {
  x += Float32x4.splat(0.45353);
  Int32x4 greater4 = x.greaterThan(Float32x4.splat(4));
  Float32x4 x2 = x * x;
  Float32x4 branch1Result = x.scale(0.25) + Float32x4.splat(1);
  Float32x4 x3 = x2 * x;

  Int32x4 lessThanMinus2 = x.lessThanOrEqual(Float32x4.splat(-2));
  Float32x4 branch3Result = x.scale(0.0625) - Float32x4.splat(0.875);  
  
  return greater4.select(
      branch1Result,
      lessThanMinus2.select(
          branch3Result,
          x3.scale(-11 / 576) +
              x2.scale(7 / 96) +
              x.scale(7 / 12) -
              Float32x4.splat(5 / 18)));
}\n''');
    }
    layers.asMap().forEach((i, layer) {
      int weightsWidth = layer.weights.nColumns;
      weightsWidth = roundUp4(weightsWidth) ~/ 2;

      stringBuffer
          .write("final List<Float32x4List> Lweight_${functionName}_$i = [");

      stringBuffer.write(layer.weights.rowsData
          .map((row) =>
              "Uint32List.fromList(${row.buffer.asUint32List().toList()}).buffer.asFloat32x4List()")
          .join(", "));
      stringBuffer.write("];\n");

      stringBuffer.write(
          "final Float32x4List Lbias_${functionName}_$i = Uint32List.fromList(${layer.bias.columnData.buffer.asUint32List().toList()}).buffer.asFloat32x4List();\n");
    });
    stringBuffer
        .write("\n\nList<double> ${functionName}(List<double> inData) \n{\n");
    stringBuffer.write("  assert(inData.length == $inputSize);\n");
    stringBuffer
        .write("  Float32List input = Float32List(${roundUp4(inputSize)});\n");
    stringBuffer
        .write("  for (int i = 0; i< $inputSize; ++i) input[i] = inData[i];\n");

    stringBuffer.write(
        "  Float32x4List currentTensor = input.buffer.asFloat32x4List();\n");
    stringBuffer.write("  Float32List outputTensor;\n");
    layers.asMap().forEach((i, layer) {
      stringBuffer.write(
          "  outputTensor = Float32List(${roundUp4(layer.weights.nRows)});\n");
      stringBuffer
          .write("  for (int r = 0; r < ${layer.weights.nRows}; ++r)\n  {\n");
      stringBuffer.write(
          "    Float32x4List weightRow = Lweight_${functionName}_$i[r];\n");
      int columns4 = (layer.weights.nColumns + 3) ~/ 4;
      if (columns4 == 1) {
        stringBuffer
            .write("    Float32x4 sum = currentTensor[0]*weightRow[0];\n");
      } else {
        stringBuffer.write(
            "    Float32x4 sum = Float32x4.zero();\n    for (int i = 0; i < $columns4; ++i)\n    {     sum+=currentTensor[i]*weightRow[i];   }\n");
      }
      stringBuffer.write(
          "    outputTensor[r] = sum.z ${layer.weights.nColumns >= 2 ? '+ sum.y' : ''} ${layer.weights.nColumns >= 3 ? '+ sum.x' : ''} ${layer.weights.nColumns >= 4 ? '+ sum.w' : ''};\n  }\n");
      stringBuffer
          .write("  currentTensor = outputTensor.buffer.asFloat32x4List();\n");
      int biasDiv4 = (layer.bias.length + 3) ~/ 4;
      if (biasDiv4 > 1) {
        stringBuffer.write("  for (int i = 0; i < ${biasDiv4}; ++i)\n");
      }
      stringBuffer.write(
          "    currentTensor[${biasDiv4 == 1 ? "0" : "i"}]+=Lbias_${functionName}_$i[${biasDiv4 == 1 ? "0" : "i"}];\n");
      var currentX4Func = [
        '',
        '',
        'absSigmoidX4',
        '',
        '',
        '',
        'slqX4'
      ][layer.activationFunc.type.index];
      var currentFunc = [
        'logisticSigmoid',
        'tanh',
        'absSigmoid',
        'bell',
        'gelu',
        'lelq',
        'slq'
      ][layer.activationFunc.type.index];
      if (currentX4Func.isNotEmpty) {
        var actFull4 = layer.bias.length ~/ 4;
        var actRemain = layer.bias.length % 4;
        if (actFull4 > 0) {
          if (actFull4 > 1) {
            stringBuffer.write("  for (int i = 0; i < ${actFull4}; ++i)\n");
            stringBuffer.write(
                "    currentTensor[i]=$currentX4Func(currentTensor[i]);\n");
          } else {
            stringBuffer.write(
                "  currentTensor[0]=$currentX4Func(currentTensor[0]);\n");
          }
        }
        for (int i = 0; i < actRemain; ++i) {
          stringBuffer.write(
              "  outputTensor[${actFull4 * 4 + i}]=$currentFunc(outputTensor[${actFull4 * 4 + i}]);\n");
        }
      } else {
        stringBuffer
            .write("  for (int i = 0; i < ${layer.bias.length}; ++i)\n");
        stringBuffer
            .write("    outputTensor[i]=$currentFunc(outputTensor[i]);\n");
      }
    });
    stringBuffer.write(
        "  return currentTensor.buffer.asFloat32List(0,${layers.last.bias.length}).toList();\n}\n\n");

    return stringBuffer.toString();
  }
}
