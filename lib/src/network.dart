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
  static double get nextReal => rng.nextDouble() * 2 - 1;
}

class FeedArtifacts {
  final FVector activation;
  final FVector derivative;
  FeedArtifacts(this.activation, this.derivative);
}

class TfannLayer {
  FLeftMatrix weights;
  FVector bias;
  ActivationFunction activationFunc;
  TfannLayer.fromJsonMap(Map json)
      : bias = FVector.fromJson(json["bias"]),
        weights = FLeftMatrix.fromJson(json["weights"]),
        activationFunc = mapActivationFunction[json["activation"]]!;

  TfannLayer(int inputs, int outputs, this.activationFunc)
      : bias = FVector.fromList(
            List.generate(outputs, (index) => RandomSupply.nextReal)),
        weights = FLeftMatrix.fromList(List.generate(outputs,
            (o) => List.generate(inputs, (i) => RandomSupply.nextReal)));
  FVector feedForward(FVector input) {
    return ((weights.multiplyVector(input)) + bias)
      ..apply((x) => activationFunc.func(x));
  }

  FeedArtifacts createFeedArtifacts(FVector input) {
    FVector intermediateVector = (weights.multiplyVector(input)) + bias;
    return FeedArtifacts(
        intermediateVector.applied((x) => activationFunc.func(x)),
        intermediateVector.applied((x) => activationFunc.derivative(x)));
  }

  Map<String, dynamic> toJson() {
    return {
      "weights": weights.toJson(),
      "bias": bias.toJson(),
      "activation": activationFunc.type.toString()
    };
  }
}

class TfannNetwork {
  List<TfannLayer> layers = [];
  TfannNetwork(this.layers);
  TfannNetwork.full(List<int> layersDefinition,
      {ActivationFunction activation = activationAbsSigmoid}) {
    int nextInputs = layersDefinition[0];

    layersDefinition.skip(1).forEach((outputs) {
      layers.add(TfannLayer(nextInputs, outputs, activation));
      nextInputs = outputs;
    });
  }
  FVector feedForward(FVector input) {
    return layers.fold(input, (vec, layer) => layer.feedForward(vec));
  }

  // returns the propagated error of the first layer. it is good for chaining networks
  FVector train(TrainData data, {double learningRate = 0.04}) {
    FVector nextInputs = data.input;
    List<FeedArtifacts> artifacts = [
      FeedArtifacts(nextInputs, FVector.zero(nextInputs.length))
    ];
    for (TfannLayer l in layers) {
      artifacts.add(l.createFeedArtifacts(nextInputs));
      nextInputs = artifacts.last.activation;
    }
    FVector netOutput = nextInputs;
    FVector netErrors;
    if (data.output != null)
      netErrors = netOutput - data.output!;
    else
      netErrors = data.errors!;
    /*Vector meanSquareError =
        netErrors.mapToVector((value) => 0.5 * value * value);*/
    FVector previousDelta = netErrors;
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
    return previousDelta;
  }

  double calculateMeanSquareError(TrainData data) {
    assert(data.output != null);
    return (feedForward(data.input) - data.output!).squared().sumElements() /
        data.output!.nRows;
  }

  double calculateRootMeanSquareError(TrainData data) {
    return sqrt(calculateMeanSquareError(data));
  }

  double calculateMeanAbsoluteError(TrainData data) {
    assert(data.output != null);
    return (feedForward(data.input) - data.output!).abs().sumElements() /
        data.output!.nRows;
  }

  Future<void> save(String filename) async {
    var sink = File(filename).openWrite(mode: FileMode.writeOnly);
    sink.write(jsonEncode(layers.map((l) => l.toJson()).toList()));

    // Close the IOSink to free system resources.
    await sink.flush();
    sink.close();
    //return ret;
  }

  static TfannNetwork? fromFile(String filename) {
    String jsonString = File(filename).readAsStringSync();
    print(jsonString);
    var net = jsonDecode(jsonString);
    assert(net is List);
    assert((net as List).isNotEmpty);
    if (net is List) {
      assert(net.every((element) => element is Map));
      return TfannNetwork(net.map((e) => TfannLayer.fromJsonMap(e)).toList());
    }
    return null;
  }

  String compile() {
    StringBuffer stringBuffer = StringBuffer();
    int inputSize = layers[0].weights.nColumns;

    stringBuffer.write("import 'dart:typed_data';\n");
    stringBuffer.write("import 'dart:math';\n\n");
    stringBuffer.write(
        "double logisticSigmoid(double x) { return 2 / (1 + exp(-x)) - 1;}\n");
    stringBuffer
        .write("double absSigmoid(double x) { return x / (1 + x.abs());}\n");
    stringBuffer.write(
        "double tanh(double x) {  var e2x = exp(2 * x);    return (e2x - 1) / (e2x + 1); }\n");

    stringBuffer.write("double bell(double x) {      return exp(-0.5*x*x);}\n");
    stringBuffer.write("double gelu(double x) {      return 0.5*x*(1+tanh(0.7978845608028653558798921198687*(x+0.044715*x*x*x)));}\n\n\n");
    
  layers.asMap().forEach((i, layer) {
      int weightsWidth = layer.weights.nColumns;
      weightsWidth = roundUp4(weightsWidth) ~/ 2;

      stringBuffer.write("final List<Float32x4List> Lweight$i = [");

      stringBuffer.write(layer.weights.rowsData
          .map((row) =>
              "Int64List.fromList(${row.buffer.asInt64List(0, weightsWidth).toList()}).buffer.asFloat32x4List()")
          .join(", "));
      stringBuffer.write("];\n");

      stringBuffer.write(
          "final Float32x4List Lbias$i = Int64List.fromList(${layer.bias.columnData.buffer.asInt64List(0, ((layer.bias.length + 3) ~/ 4) * 2).toList()}).buffer.asFloat32x4List();\n");
    });
    stringBuffer
        .write("\n\nList<double> tfann_evaluate(List<double> inData) \n{\n");
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
      stringBuffer.write(
          "  for (int r = 0; r < ${layer.weights.nRows}; ++r)\n  {\n");
      stringBuffer.write("    Float32x4List weightRow = Lweight$i[r];\n");
      int columns4 = (layer.weights.nColumns + 3) ~/ 4;
      if (columns4 == 1) {
        stringBuffer.write("    Float32x4 sum = currentTensor[0]*weightRow[0];\n");
      } else {
        stringBuffer.write(
            "    Float32x4 sum = Float32x4.zero();\n    for (int i = 0; i < $columns4; ++i)\n    {     sum+=currentTensor[i]*weightRow[i];   }\n");
      }
      stringBuffer
          .write("    outputTensor[r] = sum.z ${layer.weights.nColumns>=2?'+ sum.y':''} ${layer.weights.nColumns>=3?'+ sum.x':''} ${layer.weights.nColumns>=4?'+ sum.w':''};\n  }\n");
      stringBuffer
          .write("  currentTensor = outputTensor.buffer.asFloat32x4List();\n");
      int biasDiv4 = (layer.bias.length + 3) ~/ 4;
      if (biasDiv4 > 1) {
        stringBuffer.write(
            "  for (int i = 0; i < ${(layer.bias.length + 3) ~/ 4}; ++i)\n");
      }
      stringBuffer.write(
          "    currentTensor[${biasDiv4 == 1 ? "0" : "i"}]+=Lbias$i[${biasDiv4 == 1 ? "0" : "i"}];\n");
      stringBuffer.write("  for (int i = 0; i < ${layer.bias.length}; ++i)\n");
      stringBuffer.write("    outputTensor[i]=${[
        'logisticSigmoid',
        'tanh',
        'absSigmoid',
        'bell',
        'gelu'
      ][layer.activationFunc.type.index]}(outputTensor[i]);\n");
    });
    stringBuffer.write(
        "  return currentTensor.buffer.asFloat32List(0,${layers.last.bias.length}).toList();\n}\n\n");

    return stringBuffer.toString();
  }
}
