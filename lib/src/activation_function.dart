import 'dart:math' as math;

enum ActivationType { logistic, tanh, abs }

abstract class ActivationFunction {
  const ActivationFunction();
  double func(double x);
  double derivative(double x);
  ActivationType get type;
}

class ActivationAbsSigmoid extends ActivationFunction {
  const ActivationAbsSigmoid({this.alpha = 1});
  final double alpha;
  @override
  double func(double x) {
    return x / (alpha + x.abs());
  }

  @override
  double derivative(double x) {
    double abs_plus_alpha = alpha + x.abs();
    return alpha / (abs_plus_alpha * abs_plus_alpha);
  }
  @override
  ActivationType get type => ActivationType.abs;
}

const ActivationAbsSigmoid activationAbsSigmoid = const ActivationAbsSigmoid();

class ActivationTanh extends ActivationFunction {
  const ActivationTanh();
  @override
  double func(double x) {
    var e2x = math.exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  }

  @override
  double derivative(double x) {
    var ex = math.exp(x);
    var sech = (2 * ex) / (ex * ex + 1);
    return sech * sech;
  }
  @override
  ActivationType get type => ActivationType.tanh;
}

const ActivationTanh activationTanh = const ActivationTanh();

class ActivationLogisticSigmoid extends ActivationFunction {
  const ActivationLogisticSigmoid();
  @override
  double func(double x) {
    return 2 / (1 + math.exp(-x)) - 1;
  }

  @override
  double derivative(double x) {
    var emx = math.exp(-x);
    return 2 * emx / ((1 + emx) * (1 + emx));
  }
  @override
  ActivationType get type => ActivationType.logistic;
}

const ActivationLogisticSigmoid activationLogisticSigmoid =
    const ActivationLogisticSigmoid();

final mapActivationFunction =<String,ActivationFunction>{
ActivationType.abs.toString(): activationAbsSigmoid,
ActivationType.logistic.toString(): activationLogisticSigmoid,
ActivationType.tanh.toString(): activationTanh,
};
