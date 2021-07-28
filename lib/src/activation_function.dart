import 'dart:math' as math;

enum ActivationType { logistic, tanh, abs, bell, gelu }

double tanh(double x)
{
  var e2x = math.exp(2 * x);
  return (e2x - 1) / (e2x + 1);
}

double sech(double x)
{
  return 2.0/(math.exp(x)+math.exp(-x));
}
double sinh(double x)
{
  return (math.exp(x)-math.exp(-x))/2.0;
}
const SQRT_TWO_DIV_PI = 0.7978845608028653558798921198687;
abstract class ActivationFunction {
  const ActivationFunction();
  double func(double x);
  double derivative(double x);
  ActivationType get type;
}

class ActivationGELU extends ActivationFunction {
  const ActivationGELU();
  @override
  double func(double x) {
    
    return 0.5*x*(1+tanh(SQRT_TWO_DIV_PI*(x+0.044715*x*x*x)));
  }

  @override
  double derivative(double x) {
    double triple_x = x*x*x;
    double exp_x = SQRT_TWO_DIV_PI*x + 0.0356774*triple_x;
    double exp_part = math.exp(exp_x);
    double exp_part_minus = math.exp(-exp_x);
    double sech_part = 2.0/(exp_part+exp_part_minus);
    double tanh_part = 0.5*(exp_part-exp_part_minus)*sech_part;
    return 0.5 + (0.398942*x + 0.0535161*triple_x)*sech_part*sech_part + 0.5 *tanh_part;
     
  }

  @override
  ActivationType get type => ActivationType.gelu;
}
const ActivationGELU activationGELU = const ActivationGELU();

class ActivationBell extends ActivationFunction {
  const ActivationBell();
  @override
  double func(double x) {
    
    return math.exp(-0.5*x*x);
  }

  @override
  double derivative(double x) {
    return -x*math.exp(-0.5*x*x);
  }

  @override
  ActivationType get type => ActivationType.bell;
}
const ActivationBell activationBell = const ActivationBell();

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

final mapActivationFunction = <String, ActivationFunction>{
  ActivationType.abs.toString(): activationAbsSigmoid,
  ActivationType.logistic.toString(): activationLogisticSigmoid,
  ActivationType.tanh.toString(): activationTanh,
  ActivationType.bell.toString():activationBell,
  ActivationType.gelu.toString():activationGELU,
};
