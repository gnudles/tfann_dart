import 'dart:math' as math;

enum ActivationFunctionType { logistic, tanh, abs, bell, gelu }

double tanh(double x) {
  var e2x = math.exp(2 * x);
  return (e2x - 1) / (e2x + 1);
}

double tanhDeriv(double x) {
  var ex = math.exp(x);
  var sech = (2 * ex) / (ex * ex + 1);
  return sech * sech;
}

double sech(double x) {
  return 2.0 / (math.exp(x) + math.exp(-x));
}

double sinh(double x) {
  return (math.exp(x) - math.exp(-x)) / 2.0;
}

const SQRT_TWO_DIV_PI = 0.7978845608028653558798921198687;

class ActivationFunction {
  const ActivationFunction(this.type, this.lowerLimit, this.upperLimit,
      {required this.func, required this.derivative});
  final double Function(double) func;
  final double Function(double) derivative;
  final ActivationFunctionType type;
  final double upperLimit;
  final double lowerLimit;
}

double geluFunc(double x) {
  return 0.5 * x * (1 + tanh(SQRT_TWO_DIV_PI * (x + 0.044715 * x * x * x)));
}

double geluDeriv(double x) {
  double triple_x = x * x * x;
  double exp_x = SQRT_TWO_DIV_PI * x + 0.0356774 * triple_x;
  double exp_part = math.exp(exp_x);
  double exp_part_minus = math.exp(-exp_x);
  double sech_part = 2.0 / (exp_part + exp_part_minus);
  double tanh_part = 0.5 * (exp_part - exp_part_minus) * sech_part;
  return 0.5 +
      (0.398942 * x + 0.0535161 * triple_x) * sech_part * sech_part +
      0.5 * tanh_part;
}

const ActivationFunction activationGELU = ActivationFunction(
    ActivationFunctionType.gelu, -0.2, double.infinity,
    func: geluFunc, derivative: geluDeriv);

double bellFunc(double x) {
  return math.exp(-0.5 * x * x);
}

double bellDeriv(double x) {
  return -x * math.exp(-0.5 * x * x);
}

const ActivationFunction activationBell = ActivationFunction(
    ActivationFunctionType.bell, 0.0, 1.0,
    func: bellFunc, derivative: bellDeriv);

double absSigmoidFunc(double x) {
  return x / (1 + x.abs());
}

double absSigmoidDeriv(double x) {
  double abs_plus_one = 1 + x.abs();
  return 1 / (abs_plus_one * abs_plus_one);
}

const ActivationFunction activationAbsSigmoid = ActivationFunction(
    ActivationFunctionType.abs, -1.0, 1.0,
    func: absSigmoidFunc, derivative: absSigmoidDeriv);
const ActivationFunction activationTanh = ActivationFunction(
    ActivationFunctionType.tanh, -1.0, 1.0,
    func: tanh, derivative: tanhDeriv);

double logisticFunc(double x) {
  return 2 / (1 + math.exp(-x)) - 1;
}

double logisticDeriv(double x) {
  var emx = math.exp(-x);
  return 2 * emx / ((1 + emx) * (1 + emx));
}

const ActivationFunction activationLogisticSigmoid = ActivationFunction(
    ActivationFunctionType.logistic, -1.0, 1.0,
    func: logisticFunc, derivative: logisticDeriv);

final mapActivationFunction = <ActivationFunctionType, ActivationFunction>{
  ActivationFunctionType.abs: activationAbsSigmoid,
  ActivationFunctionType.logistic: activationLogisticSigmoid,
  ActivationFunctionType.tanh: activationTanh,
  ActivationFunctionType.bell: activationBell,
  ActivationFunctionType.gelu: activationGELU,
};

var activationTypeFromString = Map.fromEntries(
    ActivationFunctionType.values.map((e) => MapEntry(e.toString(), e)));
