import 'dart:math' as math;

import 'dart:typed_data';

enum ActivationFunctionType {
  /// the logistic sigmoid. ([0,1] bounds)
  logistic,

  /// tanh. ([-1,1] bounds)
  ///
  /// It is sort of scaled version of the logistic sigmoid
  tanh,

  /// abs sigmoid defined as x/(1+abs(x)). ([-1,1] bounds)
  abs,

  /// bell curve defined as e^(-0.5*x*x). ([0,1] bounds)
  bell,

  /// unbounded S shaped curve made from three line segments.
  uscls,

  /// unbounded S shaped curve made from two line segments connected by cubic curve.
  uscsls,

  /// unbounded ascending curve made from two lines connected by quadric curve.
  uacsls,

  /// fast bell shaped function.
  fastBell,

  /// divline sigmoid ([-1,1] bounds)
  divlineSigmoid,

  /// the identity line ->  f(x)=x
  line,

  funnyHat,

  cubicSigmoid,

  /// f(x) = x*x/4
  squartered
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
final Float32x4 _SIMD0 = Float32x4.zero();
final Float32x4 _SIMD0_75 = Float32x4.splat(0.75);
final Float32x4 _SIMD0_5 = Float32x4.splat(0.5);
final Float32x4 _SIMD0_25 = Float32x4.splat(0.25);
final Float32x4 _SIMD0_125 = Float32x4.splat(0.125);
final Float32x4 _SIMD0_375 = Float32x4.splat(0.375);
final Float32x4 _SIMD0_625 = Float32x4.splat(0.625);
final Float32x4 _SIMD0_0625 = Float32x4.splat(0.0625);
final Float32x4 _SIMD0_03 = Float32x4.splat(0.03);
final Float32x4 _SIMD0_65625 = Float32x4.splat(0.65625);
final Float32x4 _SIMD0_065 = Float32x4.splat(0.065);
final Float32x4 _SIMD0_185 = Float32x4.splat(0.185);
final Float32x4 _SIMD0_104 = Float32x4.splat(0.104);
final Float32x4 _SIMD0_208 = Float32x4.splat(0.208);
final Float32x4 _SIMD0_704 = Float32x4.splat(0.704);
final Float32x4 _SIMDm0_8 = Float32x4.splat(-0.8);
final Float32x4 _SIMDm1_5 = Float32x4.splat(-1.5);
final Float32x4 _SIMD0_28125 = Float32x4.splat(0.28125);
final Float32x4 _SIMD1 = Float32x4.splat(1);
final Float32x4 _SIMD1_47 = Float32x4.splat(1.47);
final Float32x4 _SIMD1_6 = Float32x4.splat(1.6);
final Float32x4 _SIMD4 = Float32x4.splat(4);
final Float32x4 _SIMD8 = Float32x4.splat(8);
final Float32x4 _SIMDm2 = Float32x4.splat(-2);
final Float32x4 _SIMD0_875 = Float32x4.splat(0.875);
final Float32x4 _SIMD0_4 = Float32x4.splat(0.4);
final Float32x4 _SIMDm0_16 = Float32x4.splat(-0.16);

final Float32x4List _SimdSignMaskVector = Float32x4List.fromList(List.generate(
    16,
    (index) => Float32x4(
        (index & 1) != 0 ? -1.0 : 1.0,
        (index & 2) != 0 ? -1.0 : 1.0,
        (index & 4) != 0 ? -1.0 : 1.0,
        (index & 8) != 0 ? -1.0 : 1.0)));

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

/*
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
*/
double bellFunc(double x) {
  return math.exp(-2 * x * x);
}

double bellDeriv(double x) {
  return -4 * x * math.exp(-2 * x * x);
}

const ActivationFunction activationBell = ActivationFunction(
    ActivationFunctionType.bell, 0.0, 1.0,
    func: bellFunc, derivative: bellDeriv);

double divlineSigmoidFunc(double x) {
  var absX = x.abs();
  if (absX <= 0.75) {
    return x;
  }
  var ftxmo = absX * 4 - 1;
  return x.sign * (1 - 1 / (ftxmo * ftxmo));
}

double divlineSigmoidDeriv(double x) {
  var absX = x.abs();
  if (absX <= 0.75) {
    return 1;
  }
  var ftxmo = absX * 4 - 1;
  return (8 / (ftxmo * ftxmo * ftxmo));
}

Float32x4 divlineSigmoidFuncSimd(Float32x4 x) {
  var absX = x.abs();
  var ftxmo = absX.scale(4) - _SIMD1;
  return absX.greaterThan(_SIMD0_75).select(
      _SimdSignMaskVector[x.signMask] * (_SIMD1 - (ftxmo * ftxmo).reciprocal()),
      x);
}

Float32x4 divlineSigmoidDerivSimd(Float32x4 x) {
  var absX = x.abs();
  var ftxmo = absX.scale(4) - _SIMD1;
  return absX
      .greaterThan(_SIMD0_75)
      .select((_SIMD8 / (ftxmo * ftxmo * ftxmo)), _SIMD1);
}

const ActivationFunction activationFastSigmoid = ActivationFunction(
    ActivationFunctionType.divlineSigmoid, -1.0, 1.0,
    func: divlineSigmoidFunc,
    derivative: divlineSigmoidDeriv,
    funcSIMD: divlineSigmoidFuncSimd,
    derivativeSIMD: divlineSigmoidDerivSimd);

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

Float32x4 fastBellFuncSimd(Float32x4 x) {
  var x2 = x * x;
  return x2
      .greaterThan(_SIMD0_25)
      .select((_SIMD1 - x2) / (x2.scale(8)) + _SIMD0_125, _SIMD1 - x2.scale(2));
}

Float32x4 fastBellDerivSimd(Float32x4 x) {
  var x2 = x * x;
  var xm4 = x.scale(-4);
  return x2.greaterThan(_SIMD0_25).select((x2 * xm4).reciprocal(), xm4);
}

const ActivationFunction activationFastBell = ActivationFunction(
    ActivationFunctionType.fastBell, 0.0, 1.0,
    func: fastBellFunc,
    derivative: fastBellDeriv,
    funcSIMD: fastBellFuncSimd,
    derivativeSIMD: fastBellDerivSimd);

double squarteredFunc(double x) =>x*x/4;
Float32x4 squarteredFuncSimd(Float32x4 x) =>x*x.scale(0.25);
double squarteredDeriv(double x) =>x/2;
Float32x4 squarteredDerivSimd(Float32x4 x) =>x.scale(0.5);
const ActivationFunction activationSquartered = ActivationFunction(
    ActivationFunctionType.squartered, 0.0, double.infinity,
    func: squarteredFunc,
    derivative: squarteredDeriv,
    funcSIMD: squarteredFuncSimd,
    derivativeSIMD: squarteredDerivSimd);

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

Float32x4 absSigmoidFuncSimd(Float32x4 x) {
  return x / (_SIMD1 + x.abs());
}

Float32x4 absSigmoidDerivSimd(Float32x4 x) {
  Float32x4 abs_plus_one = (_SIMD1 + x.abs());
  return (abs_plus_one * abs_plus_one).reciprocal();
}

const ActivationFunction activationAbsSigmoid = ActivationFunction(
    ActivationFunctionType.abs, -1.0, 1.0,
    func: absSigmoidFunc,
    derivative: absSigmoidDeriv,
    funcSIMD: absSigmoidFuncSimd,
    derivativeSIMD: absSigmoidDerivSimd);

double cubicSigmoidFunc(double x) {
  if (x.abs() >= 1) return 0.03 * x + x.sign * 0.96;
  return -0.48 * x * x * x + 1.47 * x;
}

Float32x4 cubicSigmoidFuncSimd(Float32x4 x) {
  return x.abs().greaterThanOrEqual(_SIMD1).select(
      x.scale(0.03) + _SimdSignMaskVector[x.signMask].scale(0.96),
      x.scale(1.47) - x * x * x.scale(0.48));
}

double cubicSigmoidDeriv(double x) {
  if (x.abs() >= 1) return 0.03;
  return -1.44 * x * x + 1.47;
}

Float32x4 cubicSigmoidDerivSimd(Float32x4 x) {
  return x
      .abs()
      .greaterThanOrEqual(_SIMD1)
      .select(_SIMD0_03, _SIMD1_47 - x * x.scale(1.44));
}

const ActivationFunction activationCubicSigmoid = ActivationFunction(
    ActivationFunctionType.cubicSigmoid,
    double.negativeInfinity,
    double.infinity,
    func: cubicSigmoidFunc,
    derivative: cubicSigmoidDeriv,
    funcSIMD: cubicSigmoidFuncSimd,
    derivativeSIMD: cubicSigmoidDerivSimd);

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
  var qx = x * 0.5;
  if (x >= 0) return qx;
  if (x > -1.5) return 0.5 * qx * qx + qx;
  return 0.125 * x - 0.28125;
}

Float32x4 uacslsFuncSimd(Float32x4 x) {
  Float32x4 qx = x.scale(0.5);
  Int32x4 greaterZero = x.greaterThanOrEqual(Float32x4.zero());
  Int32x4 greaterM3div2 = x.greaterThan(_SIMDm1_5);
  return greaterZero.select(
      qx,
      greaterM3div2.select(
          qx * qx.scale(0.5) + qx, x.scale(0.125) - _SIMD0_28125));
}

double uacslsDeriv(double x) {
  if (x >= 0) return 0.5;
  if (x > -1.5) return x * 0.25 + 0.5;
  return 0.125;
}

Float32x4 uacslsDerivSimd(Float32x4 x) {
  Int32x4 greaterZero = x.greaterThanOrEqual(Float32x4.zero());
  Int32x4 greaterM3div2 = x.greaterThan(_SIMDm1_5);
  return greaterZero.select(
      _SIMD0_5, greaterM3div2.select(x.scale(0.25) + _SIMD0_5, _SIMD0_125));
}

const ActivationFunction activationUACSLS = ActivationFunction(
    ActivationFunctionType.uacsls, double.negativeInfinity, double.infinity,
    func: uacslsFunc,
    derivative: uacslsDeriv,
    funcSIMD: uacslsFuncSimd,
    derivativeSIMD: uacslsDerivSimd);

///USCLS unbounded S curve line segments
double usclsFunc(double x) {
  if (x > 1) return 0.375 + 0.125 * x;
  if (x > -1.5) return 0.5 * x;
  return 0.0625 * x - 0.65625;
}

Float32x4 usclsFuncSimd(Float32x4 x) {
  Int32x4 greater1 = x.greaterThan(_SIMD1);
  Float32x4 branch1Result = x.scale(0.125) + _SIMD0_375;
  Int32x4 lessThanMinus1_5 = x.lessThanOrEqual(_SIMDm1_5);
  Float32x4 branch3Result = x.scale(0.0625) - _SIMD0_65625;

  return greater1.select(
      branch1Result, lessThanMinus1_5.select(branch3Result, x.scale(0.5)));
}

double usclsDeriv(double x) {
  if (x > 1) return 0.125;
  if (x > -1.5) return 0.5;
  return 0.0625;
}

Float32x4 usclsDerivSimd(Float32x4 x) {
  Int32x4 greater1 = x.greaterThan(_SIMD1);

  Int32x4 lessThanMinus1_5 = x.lessThanOrEqual(_SIMDm1_5);

  return greater1.select(
      _SIMD0_125, lessThanMinus1_5.select(_SIMD0_0625, _SIMD0_5));
}

const ActivationFunction activationUSCLS = ActivationFunction(
    ActivationFunctionType.uscls, double.negativeInfinity, double.infinity,
    func: usclsFunc,
    derivative: usclsDeriv,
    funcSIMD: usclsFuncSimd,
    derivativeSIMD: usclsDerivSimd);

double uscslsFunc(double x) {
  if (x >= 1.6) return 0.065 * x + 0.704;
  if (x > -0.8) {
    var x2 = x * x;
    var x3 = x2 * x;
    return 0.125 * (x2 - x3) + 0.625 * x;
  }
  return 0.185 * x - 0.208;
}

Float32x4 uscslsFuncSimd(Float32x4 x) {
  Int32x4 greater1_6 = x.greaterThan(_SIMD1_6);
  Float32x4 x2 = x * x;

  Float32x4 branch1Result = x.scale(0.065) + _SIMD0_704;
  Float32x4 x3 = x2 * x;

  Int32x4 lessThanMinus0_8 = x.lessThanOrEqual(_SIMDm0_8);
  Float32x4 branch3Result = x.scale(0.185) - _SIMD0_208;

  return greater1_6.select(
      branch1Result,
      lessThanMinus0_8.select(
          branch3Result, (x2 - x3).scale(0.125) + x.scale(0.625)));
}

double uscslsDeriv(double x) {
  if (x >= 1.6) return 0.065;
  if (x > -0.8) return -0.375 * x * x + 0.25 * x + 0.625;
  return 0.185;
}

Float32x4 uscslsDerivSimd(Float32x4 x) {
  Int32x4 greater1_6 = x.greaterThan(_SIMD1_6);
  Float32x4 x2 = x * x;

  Int32x4 lessThanMinus0_8 = x.lessThanOrEqual(_SIMDm0_8);

  return greater1_6.select(
      _SIMD0_065,
      lessThanMinus0_8.select(
          _SIMD0_185, x.scale(0.25) - x2.scale(0.375) + _SIMD0_625));
}

///USCSLS unbounded S curve smoothen line segments
const ActivationFunction activationUSCSLS = ActivationFunction(
    ActivationFunctionType.uscsls, double.negativeInfinity, double.infinity,
    func: uscslsFunc,
    derivative: uscslsDeriv,
    funcSIMD: uscslsFuncSimd,
    derivativeSIMD: uscslsDerivSimd);

double simpleLineFunc(double x) => x;
double simpleLineDeriv(double x) => 1;
Float32x4 simpleLineFuncSimd(Float32x4 x) => x;
Float32x4 simpleLineDerivSimd(Float32x4 x) => _SIMD1;
const ActivationFunction activationLine = ActivationFunction(
    ActivationFunctionType.line, double.negativeInfinity, double.infinity,
    func: simpleLineFunc,
    derivative: simpleLineDeriv,
    funcSIMD: simpleLineFuncSimd,
    derivativeSIMD: simpleLineDerivSimd);


double funnyHatFunc(double x) {
  double x2=x*x;
  if (x>=0)
  {
    if(x>=1.6)
    {
      return -0.16*x+0.104;
    }
    return 0.5*x*x2-1.25*x2+1;
  }
  if(x<=-2)
  {
    return 0.4*x+1;
  }
  return 1-0.1*x*x2-0.4*x2;
}
Float32x4 funnyHatFuncSimd(Float32x4 x) {
  var x2 = x*x;
  var g0 = x.greaterThan(_SIMD0);
  var x3 = x2*x;
  var g1_6 =x.greaterThanOrEqual(_SIMD1_6);
  var gm2 =x.greaterThan(_SIMDm2);
  return g0.select(g1_6.select(x.scale(-0.16)+_SIMD0_104, x3.scale(0.5)-x2.scale(1.25)+_SIMD1), gm2.select(_SIMD1-x3.scale(0.1)-x2.scale(0.4), x.scale(0.4)+_SIMD1));
}
double funnyHatDeriv(double x) {
  double x2=x*x;
  if (x>=0)
  {
    if(x>=1.6)
    {
      return -0.16;
    }
    return 1.5*x2-2.5*x;
  }
  if(x<=-2)
  {
    return 0.4;
  }
  return -0.3*x2-0.8*x;
}

Float32x4 funnyHatDerivSimd(Float32x4 x) {
  var x2 = x*x;
  var g0 = x.greaterThan(_SIMD0);
  var g1_6 =x.greaterThanOrEqual(_SIMD1_6);
  var gm2 =x.greaterThan(_SIMDm2);
  return g0.select(g1_6.select(_SIMDm0_16, x2.scale(1.5)-x.scale(2.5)), gm2.select(x2.scale(-0.3)+x.scale(-0.8), _SIMD0_4));
}
const ActivationFunction activationFunnyHat = ActivationFunction(
    ActivationFunctionType.funnyHat, double.negativeInfinity, 1,
    func: funnyHatFunc,
    derivative: funnyHatDeriv,
    funcSIMD: funnyHatFuncSimd,
    derivativeSIMD: funnyHatDerivSimd);


final mapActivationFunction = <ActivationFunctionType, ActivationFunction>{
  ActivationFunctionType.logistic: activationLogisticSigmoid,
  ActivationFunctionType.tanh: activationTanh,
  ActivationFunctionType.abs: activationAbsSigmoid,
  ActivationFunctionType.bell: activationBell,
  ActivationFunctionType.uscls: activationUSCLS,
  ActivationFunctionType.uscsls: activationUSCSLS,
  ActivationFunctionType.uacsls: activationUACSLS,
  ActivationFunctionType.fastBell: activationFastBell,
  ActivationFunctionType.divlineSigmoid: activationFastSigmoid,
  ActivationFunctionType.cubicSigmoid: activationCubicSigmoid,
  ActivationFunctionType.line: activationLine,
  ActivationFunctionType.funnyHat: activationFunnyHat,
  ActivationFunctionType.squartered: activationSquartered,
};

var activationTypeFromString = Map.fromEntries(
    ActivationFunctionType.values.map((e) => MapEntry(e.toString(), e)));
