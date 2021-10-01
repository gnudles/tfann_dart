import 'network.dart';
import 'linalg.dart';
import 'activation_function.dart';

/// Returns a pure dart code that represents the function of this network.
String compileNetwork(TfannNetwork network,
    {String functionName = 'tfann_evaluate'}) {
  StringBuffer stringBuffer = StringBuffer();
  int inputSize = network.layers[0].weights.nColumns;
  var activationsSet = network.layers.map((e) => e.activationFunc.type).toSet();

  stringBuffer.write("import 'dart:typed_data';\n");
  stringBuffer.write("import 'dart:math';\n\n");
  stringBuffer.write('''final Float32x4 _SIMD0 = Float32x4.zero();
final Float32x4 _SIMD0_75 = Float32x4.splat(0.75);
final Float32x4 _SIMD0_5 = Float32x4.splat(0.5);
final Float32x4 _SIMD0_25 = Float32x4.splat(0.25);
final Float32x4 _SIMD0_125 = Float32x4.splat(0.125);
final Float32x4 _SIMD0_375 = Float32x4.splat(0.375);
final Float32x4 _SIMD0_625 = Float32x4.splat(0.625);
final Float32x4 _SIMD0_0625 = Float32x4.splat(0.0625);
final Float32x4 _SIMD0_03 = Float32x4.splat(0.03);
final Float32x4 _SIMD0_033 = Float32x4.splat(0.033);
final Float32x4 _SIMD0_65625 = Float32x4.splat(0.65625);
final Float32x4 _SIMD0_065 = Float32x4.splat(0.065);
final Float32x4 _SIMD0_185 = Float32x4.splat(0.185);
final Float32x4 _SIMD0_104 = Float32x4.splat(0.104);
final Float32x4 _SIMD0_208 = Float32x4.splat(0.208);
final Float32x4 _SIMD0_704 = Float32x4.splat(0.704);
final Float32x4 _SIMD0_7424 = Float32x4.splat(0.7424);
final Float32x4 _SIMDm0_8 = Float32x4.splat(-0.8);
final Float32x4 _SIMDm1_5 = Float32x4.splat(-1.5);
final Float32x4 _SIMD0_28125 = Float32x4.splat(0.28125);
final Float32x4 _SIMD1 = Float32x4.splat(1);
final Float32x4 _SIMD1_47 = Float32x4.splat(1.47);
final Float32x4 _SIMD1_6 = Float32x4.splat(1.6);
final Float32x4 _SIMD4 = Float32x4.splat(4);
final Float32x4 _SIMD8 = Float32x4.splat(8);
final Float32x4 _SIMDm2 = Float32x4.splat(-2);
final Float32x4 _SIMDm3_3 = Float32x4.splat(-3.3);
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
\n''');
  if (activationsSet.contains(ActivationFunctionType.logistic)) {
    stringBuffer.write(
        "double logisticSigmoid(double x) { return 1 / (1 + exp(-x));}\n");
  }
  if (activationsSet.contains(ActivationFunctionType.abs)) {
    stringBuffer
        .write("double absSigmoid(double x) { return x / (1 + x.abs());}\n");
    stringBuffer.write(
        "Float32x4 absSigmoidX4(Float32x4 x) =>   x / (_SIMD1 + x.abs());\n");
  }
  if (activationsSet.contains(ActivationFunctionType.tanh)) {
    stringBuffer.write(
        "double tanh(double x) {  var e2x = exp(2 * x);    return (e2x - 1) / (e2x + 1); }\n");
  }
  if (activationsSet.contains(ActivationFunctionType.bell)) {
    stringBuffer.write("double bell(double x) {      return exp(-2*x*x);}\n");
  }
  if (activationsSet.contains(ActivationFunctionType.uscls)) {
    stringBuffer.write('''double uscls(double x) {
  if (x > 1) return 0.375 + 0.125 * x;
  if (x > -1.5) return 0.5 * x;
  return 0.0625 * x - 0.65625;
}\n''');
    stringBuffer.write('''Float32x4 usclsX4(Float32x4 x) {
  Int32x4 greater1 = x.greaterThan(_SIMD1);
  Float32x4 branch1Result = x.scale(0.125) + _SIMD0_375;
  Int32x4 lessThanMinus1_5 = x.lessThanOrEqual(_SIMDm1_5);
  Float32x4 branch3Result = x.scale(0.0625) - _SIMD0_65625;

  return greater1.select(
      branch1Result, lessThanMinus1_5.select(branch3Result, x.scale(0.5)));
}\n\n''');
  }
  if (activationsSet.contains(ActivationFunctionType.uscsls)) {
    stringBuffer.write('''double uscsls(double x) {
  if (x >= 1.6) return 0.065 * x + 0.704;
  if (x > -0.8) {
    var x2 = x * x;
    var x3 = x2 * x;
    return 0.125 * (x2 - x3) + 0.625 * x;
  }
  return 0.185 * x - 0.208;
}


Float32x4 uscslsX4(Float32x4 x) {
  Int32x4 greater1_6 = x.greaterThan(_SIMD1_6);
  Float32x4 x2 = x * x;

  Float32x4 branch1Result = x.scale(0.065) + _SIMD0_704;
  Float32x4 x3 = x2 * x;

  Int32x4 lessThanMinus0_8 = x.lessThanOrEqual(_SIMDm0_8);
  Float32x4 branch3Result = x.scale(0.185) - _SIMD0_208;

  return greater1_6.select(
      branch1Result,
      lessThanMinus0_8.select(
          branch3Result,
           (x2 - x3).scale(0.125) +  x.scale(0.625)));
}\n\n''');
  }
  if (activationsSet.contains(ActivationFunctionType.uacsls)) {
    stringBuffer.write('''double uacsls(double x) {
  var qx = x * 0.5;
  if (x >= 0) return qx;
  if (x > -1.5) return 0.5*qx * qx + qx;
  return 0.125 * x - 0.28125;
}

Float32x4 uacslsX4(Float32x4 x) {
  Float32x4 qx = x.scale(0.5);
  Int32x4 greaterZero = x.greaterThanOrEqual(Float32x4.zero());
  Int32x4 greaterM3div2 = x.greaterThan(_SIMDm1_5);
  return greaterZero.select(
      qx, greaterM3div2.select(qx * qx.scale(0.5) + qx, x.scale(0.125) - _SIMD0_28125));
}\n\n''');
  }
  if (activationsSet.contains(ActivationFunctionType.fastBell)) {
    stringBuffer.write('''double fastBell(double x) {
  var x2 = x * x;
  if (x2 <= 0.25) return 1 - 2 * x2;
  return (1 - x2) / (8 * x2) + 1 / 8.0;
}\n\n''');
    stringBuffer.write('''
Float32x4 fastBellX4(Float32x4 x) {
  var x2 = x * x;
  return x2
      .greaterThan(_SIMD0_25)
      .select((_SIMD1 - x2) / (x2.scale(8)) + _SIMD0_125, _SIMD1 - x2.scale(2));
}
\n\n''');
  }
  if (activationsSet.contains(ActivationFunctionType.divlineSigmoid)) {
    stringBuffer.write('''double divlineSigmoid(double x)
{
  var absX = x.abs();
  if(absX<=0.75)
  {
    return x;
  }
  var ftxmo=absX*4-1;
  return x.sign *(1-1/(ftxmo*ftxmo));
}\n''');
    stringBuffer.write('''Float32x4 divlineSigmoidX4(Float32x4 x) {
  var absX = x.abs();
  var ftxmo = absX.scale(4) - _SIMD1;
  return absX
      .greaterThan(_SIMD0_75)
      .select(_SimdSignMaskVector[x.signMask] * (_SIMD1-(ftxmo*ftxmo).reciprocal()), x);
}\n\n''');
  }
  if (activationsSet.contains(ActivationFunctionType.cubicSigmoid)) {
    stringBuffer.write(
'''double cubicSigmoid(double x)
{
  if (x.abs()>=1)
    return 0.03*x+x.sign*0.96;
  return -0.48*x*x*x+1.47*x;
}

Float32x4 cubicSigmoidX4(Float32x4 x)
{
  return x.abs().greaterThanOrEqual(_SIMD1).select(x.scale(0.03)+_SimdSignMaskVector[x.signMask].scale(0.96), x.scale(1.47)-x*x*x.scale(0.48));
}\n\n''');
  }
  if(activationsSet.contains(ActivationFunctionType.line))
  {
    stringBuffer.write('double simpleLine(double x) => x;\n');
    stringBuffer.write('Float32x4 simpleLineX4(Float32x4 x) => x;\n\n');
  }
  if(activationsSet.contains(ActivationFunctionType.squartered))
  {
    stringBuffer.write('double squartered(double x) =>x*x/4;\n');
    stringBuffer.write('Float32x4 squarteredX4(Float32x4 x) =>x*x.scale(0.25);\n\n');
  }
  if(activationsSet.contains(ActivationFunctionType.funnyHat))
  {
    stringBuffer.write('''double funnyHat(double x) {
  double x2=x*x;
  if (x>=0)
  {
    if(x>=1.6)
    {
      return -0.16*x+0.104;
    }
    return 0.5*x*x2-1.25*x2+1;
  }
  if(x<=-3.3)
  {
    return 0.033*x-0.7424;
  }
  return 1-0.1*x*x2-0.5*x2;
}
Float32x4 funnyHatX4(Float32x4 x) {
  var x2 = x*x;
  var g0 = x.greaterThan(_SIMD0);
  var x3 = x2*x;
  var g1_6 =x.greaterThanOrEqual(_SIMD1_6);
  var gm3_3 =x.greaterThan(_SIMDm3_3);
  return g0.select(g1_6.select(x.scale(-0.16)+_SIMD0_104, x3.scale(0.5)-x2.scale(1.25)+_SIMD1), gm3_3.select(_SIMD1-x3.scale(0.1)-x2.scale(0.5), x.scale(0.033)-_SIMD0_7424));
}\n\n''');
  }

  network.layers.asMap().forEach((i, layer) {
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
  network.layers.asMap().forEach((i, layer) {
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
        "    outputTensor[r] = sum.x ${layer.weights.nColumns >= 2 ? '+ sum.y' : ''} ${layer.weights.nColumns >= 3 ? '+ sum.z' : ''} ${layer.weights.nColumns >= 4 ? '+ sum.w' : ''};\n  }\n");
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
      'usclsX4',
      'uscslsX4',
      'uacslsX4',
      'fastBellX4',
      'divlineSigmoidX4',
          'simpleLineX4',
          'funnyHatX4',
          'cubicSigmoidX4',
          'squarteredX4'
    ][layer.activationFunc.type.index];
    var currentFunc = [
      'logisticSigmoid',
      'tanh',
      'absSigmoid',
      'bell',
      'uscls',
      'uscsls',
      'uacsls',
      'fastBell',
      'divlineSigmoid',
          'simpleLine',
          'funnyHat',
          'cubicSigmoid',
          'squartered'
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
          stringBuffer
              .write("  currentTensor[0]=$currentX4Func(currentTensor[0]);\n");
        }
      }
      for (int i = 0; i < actRemain; ++i) {
        stringBuffer.write(
            "  outputTensor[${actFull4 * 4 + i}]=$currentFunc(outputTensor[${actFull4 * 4 + i}]);\n");
      }
    } else {
      stringBuffer.write("  for (int i = 0; i < ${layer.bias.length}; ++i)\n");
      stringBuffer
          .write("    outputTensor[i]=$currentFunc(outputTensor[i]);\n");
    }
  });
  stringBuffer.write(
      "  return currentTensor.buffer.asFloat32List(0,${network.layers.last.bias.length}).toList();\n}\n\n");

  return stringBuffer.toString();
}
