import 'dart:typed_data';

import 'package:ml_linalg/vector.dart';

class TrainData {
  Vector input;
  Vector output;
  TrainData(this.input, this.output);
  TrainData.lists(List<double> inputList, List<double> outputList)
      : input = Vector.fromList(inputList),
        output = Vector.fromList(outputList);
}
