import 'dart:convert';
import 'dart:math';

import 'package:tfann/tfann.dart';

void main() {
  {
    Random r = Random();
    final circle_net = TfannNetwork.full([
      2,
      16,
      8,
      8,
      1
    ], [
      ActivationFunctionType.funnyHat,
      ActivationFunctionType.uscsls,
      ActivationFunctionType.uscsls,
      ActivationFunctionType.divlineSigmoid,
    ]);
    var criteria = (sr) => (sr < 1.2 && sr > 0.4) ? 1.0 : -1.0;
    for (int i = 0; i < 2000000; ++i) {
      var x = r.nextDouble() * 4 - 2;
      var y = r.nextDouble() * 4 - 2;
      var sr = sqrt(x * x + y * y);
      circle_net.train(TrainSetInputOutput.lists([x, y], [criteria(sr)]),
          learningRate: 0.01);
    }
    int false_positive = 0;
    int false_negative = 0;
    int positive = 0;
    int negative = 0;
    int testLength = 50000;
    int positive_truth = 0;
    double threshold = 0.1;
    for (int i = 0; i < testLength; ++i) {
      var x = r.nextDouble() * 4 - 2;
      var y = r.nextDouble() * 4 - 2;
      var sr = sqrt(x * x + y * y);
      var netResult = circle_net.feedForward(FVector.fromList([x, y])).single;
      if (criteria(sr) == 1) {
        positive_truth++;
      }
      if (criteria(sr) > 0 && netResult < threshold) {
        false_negative++;
      }
      if (criteria(sr) < 0 && netResult > -threshold) {
        false_positive++;
      }
      if (netResult > threshold) {
        positive++;
      }
      if (netResult < -threshold) {
        negative++;
      }
      //print("$x $y $netResult");
    }
    print("Test ground truth: Positive: ${positive_truth * 100 / testLength}%    Negative: ${100 - positive_truth * 100 / testLength}%");
    print("Result: Positive: ${positive * 100 / testLength}%  Negative: ${negative * 100 / testLength}%");
    print("False Positive: ${false_positive * 100 / testLength}%   False Negative: ${false_negative * 100 / testLength}%");
  }
}
