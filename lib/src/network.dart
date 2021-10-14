import 'dart:convert';
import 'dart:typed_data';

import 'package:tfann/tfann.dart';
import 'package:tfann/src/train_set.dart';

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

  int get inputLength => weights.nColumns;
  int get outputLength => weights.nRows;
}

/// A structure of a complete fully-connected network.
class TfannNetwork {
  List<TfannLayer> layers = [];
  TfannNetwork(this.layers);
  TfannNetwork.full(
      List<int> layersDefinition, List<ActivationFunctionType> activation) {
    assert(layersDefinition.length == activation.length + 1);

    for (int i = 0; i < activation.length; ++i) {
      layers.add(TfannLayer(
          layersDefinition[i], layersDefinition[i + 1], activation[i]));
    }
  }

  /// Get the network's output vector from the input vector.
  FVector feedForward(FVector input) {
    assert(input.length == layers.first.inputLength);
    return layers.fold(input, (vec, layer) => layer.feedForward(vec));
  }

  /// Train network with a single training pair, for a single epoch.
  ///
  /// returns the propagated error of the first layer, which is good for chained networks.
  TrainArtifacts train(TrainSet trainSet,
      {double learningRate = 0.04,
      double maxErrClipAbove = 0.0,
      double skipIfErrBelow = 0.0,
      bool Function(FVector)? skipIfOutput}) {
    assert(trainSet.input.length == layers.first.inputLength);
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
      assert(trainSet.output.length == layers.last.outputLength);
      netErrors = netOutput - trainSet.output;
    } else {
      assert((trainSet as TrainSetInputError).error.length ==
          layers.last.outputLength);
      netErrors = (trainSet as TrainSetInputError).error;
    }
    if (skipIfOutput?.call(netOutput) ?? false) {
      return TrainArtifacts(netErrors, FVector.zero(layers.first.inputLength));
    }
    FVector normalizedErrors = netErrors;
    if (maxErrClipAbove > 0.0) {
      double norm = normalizedErrors.abs().largestElement();
      if (norm < skipIfErrBelow) {
        return TrainArtifacts(
            netErrors, FVector.zero(layers.first.inputLength));
      }
      //double norm = normalizedErrors.squared().sumElements();
      if (norm > maxErrClipAbove) {
        normalizedErrors = netErrors.scaled(maxErrClipAbove / norm);
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
  FVector calculateMeanSquareError(List<TrainSetInputOutput> data) {
    var sumVector = FVector.zero(layers.last.outputLength);
    data.forEach((trainSet) {
      sumVector += (feedForward(trainSet.input) - trainSet.output).squared();
    });
    return sumVector..scale(1.0 / data.length);
  }

  /// Given a single training pair, calculate the network's mean-absolute-error per output node
  FVector calculateMeanAbsoluteError(List<TrainSetInputOutput> data) {
    var sumVector = FVector.zero(layers.last.outputLength);
    data.forEach((trainSet) {
      sumVector += (feedForward(trainSet.input) - trainSet.output).abs();
    });
    return sumVector..scale(1.0 / data.length);
  }

  /// Returns Json object.
  ///
  /// To convert it to Json string, use 'jsonEncode'
  Object toJson() {
    return layers.map((l) => l.toJson()).toList();
  }

  /// Saves the network to file.
  Future<void> save(String filename) async {
    var jsonString = jsonEncode(toJson());
    var sink = File(filename).openWrite(mode: FileMode.writeOnly);
    sink.write(jsonString);

    // Close the IOSink to free system resources.
    await sink.flush();
    sink.close();
  }

  static TfannNetwork? fromFile(String filename) {
    String jsonString = File(filename).readAsStringSync();
    return TfannNetwork.fromJson(jsonDecode(jsonString));
  }

  /// Creates TfannNetwork from Json object.
  ///
  /// Use 'jsonDecode' to convert a Json into Object.
  factory TfannNetwork.fromJson(dynamic jsonObject) {
    assert(jsonObject is List);
    assert((jsonObject as List).isNotEmpty);
    if (jsonObject is List) {
      assert(jsonObject.every((element) => element is Map));
      return TfannNetwork(
          jsonObject.map((e) => TfannLayer.fromJsonMap(e)).toList());
    }
    throw ArgumentError(
        "cannot construct a TfannNetwork from this json object");
  }
}
