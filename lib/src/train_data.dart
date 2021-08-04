import 'dart:typed_data';

import 'linalg.dart';

abstract class TrainSet {
  FVector input;
  TrainSet(this.input);
}

class TrainSetInputError extends TrainSet {
  FVector error;
  TrainSetInputError(FVector input, this.error) : super(input);

  TrainSetInputError.lists(List<double> inputList, List<double> errorList)
      : error = FVector.fromList(errorList), super(FVector.fromList(inputList));
}

class TrainSetInputOutput extends TrainSet{
  FVector output;
  TrainSetInputOutput(FVector input, this.output) : super(input);
  TrainSetInputOutput.lists(List<double> inputList, List<double> outputList)
      : output = FVector.fromList(outputList), super(FVector.fromList(inputList));
}
