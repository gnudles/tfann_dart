import 'dart:typed_data';



import 'linalg.dart';

class TrainData {
  FVector input;
  FVector? output;
  FVector? errors;
  TrainData(this.input, this.output);
  TrainData.error(this.input, this.errors);
  TrainData.lists(List<double> inputList, List<double> outputList)
      : input = FVector.fromList(inputList),
        output = FVector.fromList(outputList);
}
