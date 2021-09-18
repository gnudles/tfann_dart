import 'dart:convert';
import 'dart:math';

import 'package:tfann/tfann.dart';

void main() {
  {
    Random r = Random();
    final circle_net = TfannNetwork.full([
      2,
      4,
      4,
      1
    ], [
      ActivationFunctionType.squartered,
      ActivationFunctionType.uscsls,
      ActivationFunctionType.divlineSigmoid,
    ]);
    var criteria = (sr) => (sr < 0.6 && sr > 0.4) ? 1.0 : -1.0;
    for (int i = 0; i < 2000000; ++i) {
      var x = r.nextDouble() * 4 - 2;
      var y = r.nextDouble() * 4 - 2;
      var sr = x * x + y * y;
      circle_net.train(TrainSetInputOutput.lists([x, y], [criteria(sr)]),
          learningRate: exp(-1.5-5*r.nextDouble()));
    }
    int false_positive = 0;
    int false_negative = 0;
    int testLength = 50000;
    for (int i = 0; i < testLength; ++i) {
      var x = r.nextDouble() * 4 - 2;
      var y = r.nextDouble() * 4 - 2;
      var sr = x * x + y * y;
      var netResult = circle_net.feedForward(FVector.fromList([x, y])).single;
      if (criteria(sr)>0) if (netResult < 0.2) {
        false_negative++;
      }
      if (criteria(sr)<0) if (netResult > -0.2) {
        false_positive++;
      }
    }
    print(
        "False Positive: ${false_positive * 100 / testLength}%   False Negative: ${false_negative * 100 / testLength}%");
  }
}
