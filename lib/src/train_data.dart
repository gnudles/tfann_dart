import 'dart:typed_data';



import 'linalg.dart';

class TrainData {
  FVector input;
  FVector output;
  TrainData(this.input, this.output);
  TrainData.lists(List<double> inputList, List<double> outputList)
      : input = FVector.fromList(inputList),
        output = FVector.fromList(outputList);
}
