import 'dart:math' as math;

import 'dart:typed_data';

enum ActivationFunctionType { logistic, tanh, abs, bell, gelu, lelq, slq }

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
      {required this.func,
      required this.derivative,
      this.funcSIMD,
      this.derivativeSIMD});
  final double Function(double) func;
  final double Function(double) derivative;
  final Float32x4 Function(Float32x4)? funcSIMD;
  final Float32x4 Function(Float32x4)? derivativeSIMD;
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

//fast bell
//0.4*(1-4*x*x)/(1+x*x)+0.6
//or
//0.2*(1-16*x*x)/(1+4*x*x)+0.8
double absSigmoidFunc(double x) {
  return x / (1 + x.abs());
}

double absSigmoidDeriv(double x) {
  double abs_plus_one = 1 + x.abs();
  return 1 / (abs_plus_one * abs_plus_one);
}

final Float32x4 ones = Float32x4.splat(1);
Float32x4 absSigmoidFuncX4(Float32x4 x) {
  return x / (ones + x.abs());
}

Float32x4 absSigmoidDerivX4(Float32x4 x) {
  Float32x4 abs_plus_one = (ones + x.abs());
  return (abs_plus_one * abs_plus_one).reciprocal();
}

const ActivationFunction activationAbsSigmoid = ActivationFunction(
    ActivationFunctionType.abs, -1.0, 1.0,
    func: absSigmoidFunc,
    derivative: absSigmoidDeriv,
    funcSIMD: absSigmoidFuncX4,
    derivativeSIMD: absSigmoidDerivX4);

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

double lelqFunc(double x) {
  if (x > 4) return 1 + 0.25 * x;
  if (x > -2) return 0.5 * x;
  return 0.0625 * x - 0.875;
}
Float32x4 lelqFuncSimd(Float32x4 x) {
  Int32x4 greater4 = x.greaterThan(Float32x4.splat(4)); 
  Float32x4 branch1Result = x.scale(0.25) + Float32x4.splat(1);
  Int32x4 lessThanMinus2 = x.lessThanOrEqual(Float32x4.splat(-2));
  Float32x4 branch3Result = x.scale(0.0625) - Float32x4.splat(0.875);  
  
  return greater4.select(
      branch1Result,
      lessThanMinus2.select(
          branch3Result,x.scale(0.5)));
}

double lelqDeriv(double x) {
  if (x > 4) return 0.25;
  if (x > -2) return 0.5;
  return 0.0625;
}

Float32x4 lelqDerivSimd(Float32x4 x) {
  Int32x4 greater4 = x.greaterThan(Float32x4.splat(4)); 
  
  Int32x4 lessThanMinus2 = x.lessThanOrEqual(Float32x4.splat(-2));
  
  return greater4.select(
      Float32x4.splat(0.25),
      lessThanMinus2.select(
          Float32x4.splat(0.0625),Float32x4.splat(0.5)));
}

const ActivationFunction activationLELQ = ActivationFunction(
    ActivationFunctionType.lelq, double.negativeInfinity, double.infinity,
    func: lelqFunc, derivative: lelqDeriv, funcSIMD:lelqFuncSimd, derivativeSIMD: lelqDerivSimd);

double slqFunc(double x) {
  x += 0.45353;
  if (x > 4) return 1 + 0.25 * x;
  if (x > -2) {
    var x2 = x * x;
    var x3 = x2 * x;
    return (-11 / 576) * x3 + (7 / 96) * x2 + (7 / 12) * x - 5 / 18;
  }
  return 0.0625 * x - 0.875;
}

double slqDeriv(double x) {
  x += 0.45353;
  if (x > 4) return 0.25;
  if (x > -2) return (-33 / 576) * x * x + (7 / 48) * x + (7 / 12);
  return 0.0625;
}

Float32x4 slqFuncSimd(Float32x4 x) {
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
}


Float32x4 slqDerivSimd(Float32x4 x) {
  x += Float32x4.splat(0.45353);
  Int32x4 greater4 = x.greaterThan(Float32x4.splat(4));
  Int32x4 greater2 = x.greaterThan(Float32x4.splat(-2));
  Float32x4 x2 = x * x;
  return greater4.select( Float32x4.splat(0.25), greater2.select(x2.scale(-33 / 576)  + x.scale(7 / 48)  + Float32x4.splat(7 / 12)  , Float32x4.splat(0.0625)));
}

const ActivationFunction activationSLQ = ActivationFunction(
    ActivationFunctionType.slq, double.negativeInfinity, double.infinity,
    func: slqFunc, derivative: slqDeriv, funcSIMD: slqFuncSimd, derivativeSIMD: slqDerivSimd);

final mapActivationFunction = <ActivationFunctionType, ActivationFunction>{
  ActivationFunctionType.abs: activationAbsSigmoid,
  ActivationFunctionType.logistic: activationLogisticSigmoid,
  ActivationFunctionType.tanh: activationTanh,
  ActivationFunctionType.bell: activationBell,
  ActivationFunctionType.gelu: activationGELU,
  ActivationFunctionType.lelq: activationLELQ,
  ActivationFunctionType.slq: activationSLQ,
};

var activationTypeFromString = Map.fromEntries(
    ActivationFunctionType.values.map((e) => MapEntry(e.toString(), e)));
