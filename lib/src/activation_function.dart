import 'dart:math' as math;

import 'dart:typed_data';

enum ActivationFunctionType {
  /// the logistic sigmoid. ([0,1] bounds)
  logistic,

  /// tanh. ([-1,1] bounds)
  tanh,

  /// abs sigmoid defined as x/(1+abs(x)). ([-1,1] bounds)
  abs,

  /// bell curve defined as e^(-0.5*x*x). ([0,1] bounds)
  bell,

  /// that slow gelu function...
  gelu,

  /// unbounded S shaped curve made from three line segments.
  uscls,

  /// unbounded S shaped curve made from two line segments connected by cubic curve.
  uscsls,

  /// unbounded ascending curve made from two lines connected by quadric curve.
  uacsls,

  /// fast bell shaped function.
  fastBell
}

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
final Float32x4 _SIMD0_5 = Float32x4.splat(0.5);
final Float32x4 _SIMD0_25 = Float32x4.splat(0.25);
final Float32x4 _SIMD0_0625 = Float32x4.splat(0.0625);
final Float32x4 _SIMDm1_5 = Float32x4.splat(-1.5);
final Float32x4 _SIMD0_140625 = Float32x4.splat(0.140625);
final Float32x4 _SIMD1 = Float32x4.splat(1);
final Float32x4 _SIMD4 = Float32x4.splat(4);
final Float32x4 _SIMDm2 = Float32x4.splat(-2);
final Float32x4 _SIMD0_875 = Float32x4.splat(0.875);

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

double fastBellFunc(double x) {
  var x2 = x * x;
  if (x2 <= 0.25) return 1 - 2 * x2;
  return (1 - x2) / (8 * x2) + 1 / 8.0;
}

double fastBellDeriv(double x) {
  var x2 = x * x;
  if (x2 <= 0.25) return -4 * x;
  return -1 / (4 * x2 * x);
}

const ActivationFunction activationFastBell = ActivationFunction(
    ActivationFunctionType.fastBell, 0.0, 1.0,
    func: fastBellFunc, derivative: fastBellDeriv);

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
  return 1 / (1 + math.exp(-x));
}

double logisticDeriv(double x) {
  var emx = math.exp(-x);
  return emx / ((1 + emx) * (1 + emx));
}

const ActivationFunction activationLogisticSigmoid = ActivationFunction(
    ActivationFunctionType.logistic, 0.0, 1.0,
    func: logisticFunc, derivative: logisticDeriv);

///UACSLS unbounded ascending curve smoothen line segments
double uacslsFunc(double x) {
  var qx = x * 0.25;
  if (x >= 0) return qx;
  if (x > -1.5) return qx * qx + qx;
  return 0.0625 * x - 0.140625;
}



Float32x4 uacslsFuncSimd(Float32x4 x) {
  Float32x4 qx = x.scale(0.25);
  Int32x4 greaterZero = x.greaterThanOrEqual(Float32x4.zero());
  Int32x4 greaterM3div2 = x.greaterThan(_SIMDm1_5);
  return greaterZero.select(
      qx,
      greaterM3div2.select(
          qx * qx + qx, x.scale(0.0625) - _SIMD0_140625));
}

double uacslsDeriv(double x) {
  if (x >= 0) return 0.25;
  if (x > -1.5) return x * 0.125 + 0.25;
  return 0.0625;
}

Float32x4 uacslsDerivSimd(Float32x4 x) {
  Int32x4 greaterZero = x.greaterThanOrEqual(Float32x4.zero());
  Int32x4 greaterM3div2 = x.greaterThan(_SIMDm1_5);
  return greaterZero.select(
      _SIMD0_25,
      greaterM3div2.select(
          x.scale(0.125) + _SIMD0_25, _SIMD0_0625));
}

const ActivationFunction activationUACSLS = ActivationFunction(
  ActivationFunctionType.uacsls,
  double.negativeInfinity,
  double.infinity,
  func: uacslsFunc,
  derivative: uacslsDeriv,
  funcSIMD: uacslsFuncSimd,
  derivativeSIMD: uacslsDerivSimd
);

///USCLS unbounded S curve line segments
double usclsFunc(double x) {
  if (x > 4) return 1 + 0.25 * x;
  if (x > -2) return 0.5 * x;
  return 0.0625 * x - 0.875;
}

Float32x4 usclsFuncSimd(Float32x4 x) {
  Int32x4 greater4 = x.greaterThan(_SIMD4);
  Float32x4 branch1Result = x.scale(0.25) + _SIMD1;
  Int32x4 lessThanMinus2 = x.lessThanOrEqual(_SIMDm2);
  Float32x4 branch3Result = x.scale(0.0625) - _SIMD0_875;

  return greater4.select(
      branch1Result, lessThanMinus2.select(branch3Result, x.scale(0.5)));
}

double usclsDeriv(double x) {
  if (x > 4) return 0.25;
  if (x > -2) return 0.5;
  return 0.0625;
}

Float32x4 usclsDerivSimd(Float32x4 x) {
  Int32x4 greater4 = x.greaterThan(_SIMD4);

  Int32x4 lessThanMinus2 = x.lessThanOrEqual(_SIMDm2);

  return greater4.select(_SIMD0_25,
      lessThanMinus2.select(_SIMD0_0625, _SIMD0_5));
}

const ActivationFunction activationUSCLS = ActivationFunction(
    ActivationFunctionType.uscls, double.negativeInfinity, double.infinity,
    func: usclsFunc,
    derivative: usclsDeriv,
    funcSIMD: usclsFuncSimd,
    derivativeSIMD: usclsDerivSimd);

double uscslsFunc(double x) {
  x += 0.45353;
  if (x > 4) return 1 + 0.25 * x;
  if (x > -2) {
    var x2 = x * x;
    var x3 = x2 * x;
    return (-11 / 576) * x3 + (7 / 96) * x2 + (7 / 12) * x - 5 / 18;
  }
  return 0.0625 * x - 0.875;
}

double uscslsDeriv(double x) {
  x += 0.45353;
  if (x > 4) return 0.25;
  if (x > -2) return (-33 / 576) * x * x + (7 / 48) * x + (7 / 12);
  return 0.0625;
}

Float32x4 uscslsFuncSimd(Float32x4 x) {
  x += Float32x4.splat(0.45353);
  Int32x4 greater4 = x.greaterThan(_SIMD4);
  Float32x4 x2 = x * x;

  Float32x4 branch1Result = x.scale(0.25) + _SIMD1;
  Float32x4 x3 = x2 * x;

  Int32x4 lessThanMinus2 = x.lessThanOrEqual(_SIMDm2);
  Float32x4 branch3Result = x.scale(0.0625) - _SIMD0_875;

  return greater4.select(
      branch1Result,
      lessThanMinus2.select(
          branch3Result,
          x3.scale(-11 / 576) +
              x2.scale(7 / 96) +
              x.scale(7 / 12) -
              Float32x4.splat(5 / 18)));
}

Float32x4 uscslsDerivSimd(Float32x4 x) {
  x += Float32x4.splat(0.45353);
  Int32x4 greater4 = x.greaterThan(_SIMD4);
  Int32x4 greater2 = x.greaterThan(_SIMDm2);
  Float32x4 x2 = x * x;
  return greater4.select(
      Float32x4.splat(0.25),
      greater2.select(
          x2.scale(-33 / 576) + x.scale(7 / 48) + Float32x4.splat(7 / 12),
          _SIMD0_0625));
}

///USCSLS unbounded S curve smoothen line segments
const ActivationFunction activationUSCSLS = ActivationFunction(
    ActivationFunctionType.uscsls, double.negativeInfinity, double.infinity,
    func: uscslsFunc,
    derivative: uscslsDeriv,
    funcSIMD: uscslsFuncSimd,
    derivativeSIMD: uscslsDerivSimd);

final mapActivationFunction = <ActivationFunctionType, ActivationFunction>{
  ActivationFunctionType.abs: activationAbsSigmoid,
  ActivationFunctionType.logistic: activationLogisticSigmoid,
  ActivationFunctionType.tanh: activationTanh,
  ActivationFunctionType.bell: activationBell,
  ActivationFunctionType.gelu: activationGELU,
  ActivationFunctionType.uscls: activationUSCLS,
  ActivationFunctionType.uscsls: activationUSCSLS,
  ActivationFunctionType.uacsls: activationUACSLS,
  ActivationFunctionType.fastBell: activationFastBell,
};

var activationTypeFromString = Map.fromEntries(
    ActivationFunctionType.values.map((e) => MapEntry(e.toString(), e)));
