import 'dart:convert';
import 'dart:math';

import 'package:tfann/tfann.dart';

void main() {
  {
    Random r = Random();
    final circle_net = TfannNetwork.full([
      2,
      32,
      16,
      16,
      1
    ], [
      ActivationFunctionType.uscls,
      ActivationFunctionType.fastBell,
      ActivationFunctionType.uscls,
      ActivationFunctionType.fastBell
    ]);
    var criteria = (sr) => (sr < 1 && sr > 0.25) ? 1.0 : 0.0;
    for (int i = 0; i < 2000000; ++i) {
      var x = r.nextDouble() * 3 - 1.5;
      var y = r.nextDouble() * 3 - 1.5;
      var sr = x * x + y * y;
      circle_net.train(TrainSetInputOutput.lists([x, y], [criteria(sr)]),
          learningRate: 0.02);
    }
    int false_positive = 0;
    int false_negative = 0;
    int testLength = 50000;
    for (int i = 0; i < testLength; ++i) {
      var x = r.nextDouble() * 3 - 1.5;
      var y = r.nextDouble() * 3 - 1.5;
      var sr = x * x + y * y;
      var netResult = circle_net.feedForward(FVector.fromList([x, y])).single;
      if (sr < 1 && sr > 0.25) if (netResult < 0.8) {
        false_negative++;
      }
      if (!(sr < 1 && sr > 0.25)) if (netResult > 0.2) {
        false_positive++;
      }
    }
    print(
        "False Positive: ${false_positive * 100 / testLength}%   False Negative: ${false_negative * 100 / testLength}%");
  }
}
