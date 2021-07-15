import 'dart:convert';

import 'package:ann/ann.dart';
import 'package:ann/src/train_data.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:quiver/core.dart';
import 'activation_function.dart';
import 'dart:math';
import 'dart:io';

class RandomSupply {
  static final Random rng = Random();
  static double get nextReal => rng.nextDouble() * 2 - 1;
}

class FeedArtifacts {
  final Vector activation;
  final Vector derivative;
  FeedArtifacts(this.activation, this.derivative);
}

class Layer {
  Matrix weights;
  Vector bias;
  ActivationFunction activationFunc;
  Layer.fromJsonMap(Map json)
      : bias = Vector.fromJson(json["bias"]),
        weights = Matrix.fromJson(json["weights"]),
        activationFunc = mapActivationFunction[json["activation"]]!;

  Layer(int inputs, int outputs, this.activationFunc)
      : bias = Vector.fromList(
            List.generate(outputs, (index) => RandomSupply.nextReal)),
        weights = Matrix.fromList(List.generate(outputs,
            (o) => List.generate(inputs, (i) => RandomSupply.nextReal)));
  Vector feedForward(Vector input) {
    return ((weights * input).toVector() + bias)
        .mapToVector((x) => activationFunc.func(x));
  }

  FeedArtifacts createFeedArtifacts(Vector input) {
    Vector intermediateVector = (weights * input).toVector() + bias;
    return FeedArtifacts(
        intermediateVector.mapToVector((x) => activationFunc.func(x)),
        intermediateVector.mapToVector((x) => activationFunc.derivative(x)));
  }

  Map<String, dynamic> toJson() {
    return {"weights": weights.toJson(), "bias": bias.toJson(), "activation": activationFunc.type.toString()};
  }
}

class Network {
  List<Layer> layers = [];
  Network(this.layers);
  Network.full(List<int> layersDefinition,
      {ActivationFunction activation = activationAbsSigmoid}) {
    int nextInputs = layersDefinition[0];

    layersDefinition.skip(1).forEach((outputs) {
      layers.add(Layer(nextInputs, outputs, activation));
      nextInputs = outputs;
    });
  }
  Vector feedForward(Vector input) {
    return layers.fold(input, (vec, layer) => layer.feedForward(vec));
  }

  void train(TrainData data, {double learningRate = 0.04}) {
    Vector nextInputs = data.input;
    List<FeedArtifacts> artifacts = [
      FeedArtifacts(nextInputs, Vector.zero(nextInputs.length))
    ];
    for (Layer l in layers) {
      artifacts.add(l.createFeedArtifacts(nextInputs));
      nextInputs = artifacts.last.activation;
    }
    Vector netOutput = nextInputs;
    Vector netErrors = netOutput - data.output;
    /*Vector meanSquareError =
        netErrors.mapToVector((value) => 0.5 * value * value);*/
    Vector previousDelta = netErrors;
    List<Vector> layerDelta = [];
    for (int i = layers.length - 1; i >= 0; --i) {
      Vector currentDelta = (artifacts[i + 1].derivative * previousDelta);
      layerDelta.add(currentDelta);

      previousDelta = (layers[i].weights.transpose() * currentDelta).toVector();
    }

    var arti = artifacts.iterator;
    for (Layer l in layers) {
      l.bias -= layerDelta.last * learningRate;
      arti.moveNext();

      l.weights -= Matrix.column(layerDelta.last.toList()) *
          Matrix.row(arti.current.activation.toList()) *
          learningRate;
      layerDelta.removeLast();
    }
  }

  Future<void> save(String filename) async {
    var sink = File(filename).openWrite(mode: FileMode.writeOnly);
    sink.write(jsonEncode(layers.map((l) => l.toJson()).toList()));

    // Close the IOSink to free system resources.
    await sink.flush();
    sink.close();
    //return ret;
  }

  static Optional<Network> fromFile(String filename) {
    String jsonString = File(filename).readAsStringSync();
    print(jsonString);
    var net = jsonDecode(jsonString);
    assert(net is List);
    assert((net as List).isNotEmpty);
    if (net is List) {
      assert(net.every((element) => element is Map));
      return Optional.of(
          Network(net.map((e) => Layer.fromJsonMap(e)).toList()));
    }
    return Optional.absent();
  }
}
