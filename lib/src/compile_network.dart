
import 'network.dart';
import 'linalg.dart';
import 'activation_function.dart';

/// Returns a pure dart code that represents the function of this network.
  String compileNetwork(TfannNetwork network, {String functionName = 'tfann_evaluate'}) {
    StringBuffer stringBuffer = StringBuffer();
    int inputSize = network.layers[0].weights.nColumns;
    var activationsSet = network.layers.map((e) => e.activationFunc.type).toSet();

    stringBuffer.write("import 'dart:typed_data';\n");
    stringBuffer.write("import 'dart:math';\n\n");
    if (activationsSet.contains(ActivationFunctionType.logistic))
      stringBuffer.write(
          "double logisticSigmoid(double x) { return 2 / (1 + exp(-x)) - 1;}\n");
    if (activationsSet.contains(ActivationFunctionType.abs)) {
      stringBuffer
          .write("double absSigmoid(double x) { return x / (1 + x.abs());}\n");
      stringBuffer.write("final Float32x4 onesX4 = Float32x4.splat(1);\n");
      stringBuffer.write(
          "Float32x4 absSigmoidX4(Float32x4 x) =>   x / (onesX4 + x.abs());\n");
    }
    if (activationsSet.contains(ActivationFunctionType.tanh))
      stringBuffer.write(
          "double tanh(double x) {  var e2x = exp(2 * x);    return (e2x - 1) / (e2x + 1); }\n");
    if (activationsSet.contains(ActivationFunctionType.bell))
      stringBuffer
          .write("double bell(double x) {      return exp(-0.5*x*x);}\n");
    if (activationsSet.contains(ActivationFunctionType.gelu))
      stringBuffer.write(
          "double gelu(double x) {      return 0.5*x*(1+tanh(0.7978845608028653558798921198687*(x+0.044715*x*x*x)));}\n");
    if (activationsSet.contains(ActivationFunctionType.uscls))
      stringBuffer.write(
          "double uscls(double x) {  if (x > 4) return 1 + 0.25 * x;  if (x > -2) return 0.5 * x;  return 0.0625 * x - 0.875; }\n");
    if (activationsSet.contains(ActivationFunctionType.uscsls)) {
      stringBuffer.write(
          "double uscsls(double x) {  x += 0.45353;  if (x > 4) return 1 + 0.25 * x;  if (x > -2) {    var x2 = x * x;    var x3 = x2 * x;    return (-11/576)*x3+(7/96)*x2+(7/12)*x-5/18;  }  return 0.0625 * x - 0.875;}\n");
      stringBuffer.write('''Float32x4 uscslsX4(Float32x4 x) {
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
}\n''');
    }
    if (activationsSet.contains(ActivationFunctionType.uacsls)) {
      stringBuffer.write('''double uacsls(double x) {
  var qx = x * 0.25;
  if (x >= 0) return qx;
  if (x > -1.5) return qx * qx + qx;
  return 0.0625 * x - 0.140625;
}''');
    }
    if (activationsSet.contains(ActivationFunctionType.fastBell)) {
      stringBuffer.write('''double fastBell(double x) {
  var x2 = x * x;
  if (x2 <= 0.25) return 1 - 2 * x2;
  return (1 - x2) / (8 * x2) + 1 / 8.0;
}''');
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
        '',
        '',
        'uscslsX4',
        '',
        ''
      ][layer.activationFunc.type.index];
      var currentFunc = [
        'logisticSigmoid',
        'tanh',
        'absSigmoid',
        'bell',
        'gelu',
        'uscls',
        'uscsls',
        'uacsls',
        'fastBell'
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
            stringBuffer.write(
                "  currentTensor[0]=$currentX4Func(currentTensor[0]);\n");
          }
        }
        for (int i = 0; i < actRemain; ++i) {
          stringBuffer.write(
              "  outputTensor[${actFull4 * 4 + i}]=$currentFunc(outputTensor[${actFull4 * 4 + i}]);\n");
        }
      } else {
        stringBuffer
            .write("  for (int i = 0; i < ${layer.bias.length}; ++i)\n");
        stringBuffer
            .write("    outputTensor[i]=$currentFunc(outputTensor[i]);\n");
      }
    });
    stringBuffer.write(
        "  return currentTensor.buffer.asFloat32List(0,${network.layers.last.bias.length}).toList();\n}\n\n");

    return stringBuffer.toString();
  }